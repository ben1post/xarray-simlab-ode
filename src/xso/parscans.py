import importlib
import warnings
import xarray as xr
import xsimlab as xs
import time as tm
import numpy as np
from multiprocessing import Pool
from IPython.display import clear_output
import os
# (redirect_stdout / nullcontext previously used to suppress solver
# diagnostics; no longer needed since the solvers log through
# xso.solvers' logger instead of writing to stdout.)

# Suppress warnings that clutter output, especially common in scientific computing libraries
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global variables (must be defined outside functions for multiprocessing)
_global_model = None
_global_model_setup = None
_global_postprocess = None
_global_postprocess_kwargs = {}


def _validate_model_loading(module_name, model_name_str, model_setup_name_str,
                            postprocess_name=None):
    """
    Performs a pre-flight check in the main process to ensure
    the model, setup, and (optionally) postprocess callable can be
    loaded before starting workers.

    Raises ImportError, AttributeError, or TypeError if loading fails.
    """
    try:
        model_module = importlib.import_module(module_name)

        try:
            getattr(model_module, model_name_str)
        except AttributeError as e:
            print(f"\n[FATAL PRE-CHECK ERROR] Failed to find model object "
                  f"'{model_name_str}' in '{module_name}.py'.")
            suggestions = [o for o in dir(model_module)
                           if 'model' in o.lower() and not o.startswith('__')]
            if suggestions:
                print(f"Did you mean one of these? {suggestions}")
            raise e

        try:
            getattr(model_module, model_setup_name_str)
        except AttributeError as e:
            print(f"\n[FATAL PRE-CHECK ERROR] Failed to find setup object "
                  f"'{model_setup_name_str}' in '{module_name}.py'.")
            suggestions = [o for o in dir(model_module)
                           if 'setup' in o.lower() and not o.startswith('__')]
            if suggestions:
                print(f"Did you mean one of these? {suggestions}")
            raise e

        if postprocess_name is not None:
            try:
                pp = getattr(model_module, postprocess_name)
            except AttributeError as e:
                print(f"\n[FATAL PRE-CHECK ERROR] Failed to find postprocess "
                      f"callable '{postprocess_name}' in '{module_name}.py'.")
                suggestions = [o for o in dir(model_module)
                               if 'postprocess' in o.lower() or 'avg' in o.lower()
                               and not o.startswith('__')]
                if suggestions:
                    print(f"Did you mean one of these? {suggestions}")
                raise e
            if not callable(pp):
                raise TypeError(
                    f"'{postprocess_name}' in '{module_name}' is not callable."
                )

    except ImportError as e:
        print(f"\n[FATAL PRE-CHECK ERROR] Failed to import model file: "
              f"'{module_name}.py'.")
        print(f"Details: {e}")

        print("\n--- 💡 Debugging Import Error ---")
        try:
            cwd = os.getcwd()
            print(f"Python's Current Working Directory (CWD) is:")
            print(f"   {cwd}")
            print(f"\nIs this the folder where your '{module_name}.py' file is located?")

            expected_file_path = os.path.join(cwd, module_name + ".py")
            if not os.path.exists(expected_file_path):
                print(f"-> The file '{module_name}.py' was NOT found in the CWD.")
            else:
                print(f"-> The file '{module_name}.py' WAS found in the CWD (this is good!).")

        except Exception as debug_e:
            print(f"(Could not retrieve full debug info: {debug_e})")
        raise e

    except Exception as e:
        print(f"\n[FATAL PRE-CHECK ERROR] An unexpected error occurred while "
              f"loading '{module_name}'.")
        print(f"Details: {e}")
        raise e


def _validate_fixed_overrides(fixed_overrides, scan_param_names):
    """Raise if fixed_overrides collides with parameters being scanned."""
    if not fixed_overrides:
        return
    collisions = [p for p in scan_param_names if p in fixed_overrides]
    if collisions:
        raise ValueError(
            f"fixed_overrides contains parameter(s) also being scanned: "
            f"{collisions}. A parameter cannot be both scanned and held fixed."
        )


def _validate_linked_overrides(linked_overrides, param_values, scan_param_names,
                               fixed_overrides=None):
    """Validate linked_overrides for a 1D trajectory scan.

    Each linked parameter advances element-wise with the primary scan axis,
    so every linked array must match ``len(param_values)``. Linked keys may
    not name a scanned axis (that axis is already being varied) nor a
    ``fixed_overrides`` key (that would set the same parameter two ways).
    ``None`` is a no-op.
    """
    if not linked_overrides:
        return
    if not isinstance(linked_overrides, dict):
        raise TypeError(
            f"linked_overrides must be a dict mapping parameter name to a "
            f"sequence, got {type(linked_overrides).__name__}."
        )
    n = len(param_values)
    for key, values in linked_overrides.items():
        try:
            n_linked = len(values)
        except TypeError:
            raise TypeError(
                f"linked_overrides['{key}'] must be a sized sequence of "
                f"length {n}, got a non-sized {type(values).__name__}."
            )
        if n_linked != n:
            raise ValueError(
                f"linked_overrides['{key}'] has length {n_linked}, but "
                f"param_values has length {n}. Linked parameters must share "
                f"the primary scan axis element-wise."
            )
    scanned_collisions = [k for k in linked_overrides if k in scan_param_names]
    if scanned_collisions:
        raise ValueError(
            f"linked_overrides contains parameter(s) also being scanned: "
            f"{scanned_collisions}. A scanned axis cannot also be linked."
        )
    if fixed_overrides:
        fixed_collisions = [k for k in linked_overrides if k in fixed_overrides]
        if fixed_collisions:
            raise ValueError(
                f"linked_overrides and fixed_overrides share parameter(s): "
                f"{fixed_collisions}. A parameter cannot be both linked and "
                f"held fixed."
            )


def _worker_initializer(module_name, model_name_str, model_setup_name_str,
                        postprocess_name=None, postprocess_kwargs=None):
    """Initializes a worker process for multiprocessing.

    Dynamically imports the model, model_setup, and (optionally) a
    postprocess callable from the given module, storing them in the
    worker's global scope.

    Parameters
    ----------
    module_name : str
    model_name_str : str
    model_setup_name_str : str
    postprocess_name : str or None, optional
        Name of a callable in ``module_name`` with signature
        ``(ds, **kwargs) -> xr.Dataset`` applied to each run's output
        before it leaves the worker. If None, no post-processing.
    postprocess_kwargs : dict or None, optional
        Keyword arguments passed to the postprocess callable on every
        call. Set once per worker.
    """
    global _global_model, _global_model_setup
    global _global_postprocess, _global_postprocess_kwargs

    try:
        model_module = importlib.import_module(module_name)

        _global_model = getattr(model_module, model_name_str)
        _global_model_setup = getattr(model_module, model_setup_name_str)

        if postprocess_name is not None:
            pp = getattr(model_module, postprocess_name)
            if not callable(pp):
                raise TypeError(
                    f"'{postprocess_name}' in '{module_name}' is not callable."
                )
            _global_postprocess = pp
        else:
            _global_postprocess = None
        _global_postprocess_kwargs = postprocess_kwargs or {}

    except Exception as e:
        print(f"FATAL ERROR: Worker failed to initialize from '{module_name}'.")
        extra = f", postprocess='{postprocess_name}'" if postprocess_name else ""
        print(f"Attempted to load: model='{model_name_str}', "
              f"setup='{model_setup_name_str}'{extra}.")
        print(f"Check that these objects exist in your script. Details: {e}",
              flush=True)
        raise e


def _scalarize_param_for_coord(param_val):
    """
    Converts a parameter value (scalar or array) into a scalar
    for use as a xarray coordinate.
    """
    if np.isscalar(param_val):
        return param_val

    try:
        # Use np.max as a general rule for arrays.
        # This works for [0, 0, 0, 0.005] -> 0.005
        # This also works for [0.1, 0, 0] -> 0.1
        # Note: [0.1, 0] and [0, 0.1] will both map to 0.1
        # The user must ensure this scalar representation is unique
        # for their scan.
        return np.max(param_val)
    except Exception:
        # Fallback for other non-scalar types
        return str(param_val)


def _run_model_scan_point(task_data):
    """Executes a single simulation run within a worker process.

    Runs the solver, applies scan-param coordinates, standardizes the
    time coordinate, and (if configured) applies the user postprocess
    hook.

    Returns
    -------
    xarray.Dataset or None
        The (possibly post-processed) run output, or ``None`` if either
        the solve or the postprocess raised an exception. A ``None``
        sentinel causes the unpacking stage to fill the cell with NaN.
    """
    scan_param_dict = task_data['scan_param_dict']

    model = _global_model
    model_setup = _global_model_setup

    if model is None or model_setup is None:
        raise RuntimeError("Model objects were not initialized in this worker process.")

    try:
        with model:
            model_out = model_setup.xsimlab.update_vars(
                input_vars=scan_param_dict
            ).xsimlab.run()
    except Exception as e:
        print(f"[WARNING] Solver run failed for scan_param_dict={scan_param_dict}. "
              f"Cell will be NaN-filled. Details: {e}", flush=True)
        return None

    scalar_coords = {
        p_name: p_value for p_name, p_value in scan_param_dict.items()
        if np.isscalar(p_value)
    }
    model_out = model_out.assign_coords(scalar_coords)

    if 'time' in model_out.coords:
        model_out['time'] = model_out.time.round(9)

    if _global_postprocess is not None:
        try:
            model_out = _global_postprocess(model_out, **_global_postprocess_kwargs)
        except Exception as e:
            print(f"[WARNING] Postprocess failed for scan_param_dict={scan_param_dict}. "
                  f"Cell will be NaN-filled. Details: {e}", flush=True)
            return None

    return model_out


def _generate_iterable_tasks_1d(parameter, par_range,
                                initial_values_ds=None, iv_mapping=None, fixed_overrides=None,
                                linked_overrides=None):
    """Creates the list of task dictionaries for a 1D parameter scan.

    Injects initial values only if they are finite AND biologically plausible.

    ``linked_overrides`` (if given) maps parameter names to sequences the same
    length as ``par_range``; at index ``i`` each linked value is injected
    alongside the primary parameter, advancing element-wise with it (a
    trajectory through parameter space, not a cross-product).
    """
    tasks = []
    for i, val in enumerate(par_range):
        scan_dict = {parameter: val}
        if initial_values_ds is not None and iv_mapping is not None:
            sel_val = _scalarize_param_for_coord(val)
            iv_point = initial_values_ds.sel({parameter: sel_val}, method='nearest')
            for var_name, init_param_name in iv_mapping.items():
                if var_name in iv_point:
                    iv_value = iv_point[var_name].values

                    # --- FINAL CHECK ---
                    # Check 1: Is it finite?
                    is_finite = np.isfinite(iv_value).all()

                    # Check 2: Is it biologically plausible (non-negative)?
                    # Assumes variables with 'biomass' or 'value' (for Nutrient) should be >= 0
                    is_plausible = True
                    if 'biomass' in var_name or 'value' in var_name:
                        try:
                            # Use np.all() in case iv_value is an array
                            is_plausible = np.all(iv_value >= 0)
                        except TypeError:  # Handle non-numeric types just in case
                            is_plausible = False

                    if is_finite and is_plausible:
                        scan_dict[init_param_name] = iv_value
                    else:
                        pass  # Use model default if check fails
                    # --- END FINAL CHECK ---
                else:
                    print(f"Warning: '{var_name}' not found...")

        # Linked parameters advance element-wise with the primary axis.
        # Coerce to a plain Python float so update_vars and the coord logic
        # treat them as scalars (not 0-d numpy arrays).
        if linked_overrides:
            for k in linked_overrides:
                scan_dict[k] = float(linked_overrides[k][i])

        # Allow passing parameters to scan
        if fixed_overrides:
            scan_dict.update(fixed_overrides)

        tasks.append({'scan_param_dict': scan_dict})
    return tasks


def _nan_template_like(ds):
    """Return a copy of ``ds`` with all numeric data variables replaced by NaN.

    Non-numeric variables (string labels, object dtype) are preserved as-is
    — they encode structural/metadata info that is the same across cells.
    """
    out = ds.copy(deep=False)
    for name in list(out.data_vars):
        da = out[name]
        if np.issubdtype(da.dtype, np.number):
            out[name] = xr.DataArray(
                np.full(da.shape, np.nan, dtype=float),
                coords=da.coords, dims=da.dims, attrs=da.attrs,
            )
    return out


def _unpack_par_scan_1d(iterable_tasks, data, linked_overrides=None):
    """Combines the results from a 1D parameter scan into a single Dataset.

    ``None`` entries in ``data`` (failed cells) are replaced with a
    NaN-filled Dataset that matches the shape of the first successful
    cell, then re-coordinated so ``combine_by_coords`` aligns them.

    ``linked_overrides`` (if given) names parameters that vary per scan
    point. Their per-cell scalar copies are dropped and reattached as
    coordinates indexed along ``scan_param_name``, sourced from the
    authoritative arrays so they are correct and present even for failed
    (NaN-filled) cells. Each cell carries its own linked value, so they
    concatenate in lockstep with the scan axis regardless of ordering.
    """
    if not iterable_tasks or not data:
        return xr.Dataset()

    first_dict = iterable_tasks[0]['scan_param_dict']
    scan_param_name = list(first_dict.keys())[0]

    template = next((d for d in data if d is not None), None)
    if template is None:
        print("[WARNING] All scan cells failed; returning empty Dataset.")
        return xr.Dataset()

    linked_keys = list(linked_overrides) if linked_overrides else []

    dat_out = []
    for i, (dat, task) in enumerate(zip(data, iterable_tasks)):
        if dat is None:
            dat = _nan_template_like(template)
        # Linked params vary per cell. Drop the per-cell scalar copy (left
        # as a 0-d coord by _run_model_scan_point) -- it would collide as a
        # conflicting non-dimension coord in combine_by_coords -- then
        # reattach this cell's value as a length-1 coordinate along the scan
        # axis so it concatenates in lockstep with scan_param_name.
        if linked_keys:
            dat = dat.drop_vars(linked_keys, errors='ignore')
        scan_value = _scalarize_param_for_coord(
            task['scan_param_dict'][scan_param_name])
        dat = (dat.assign_coords({scan_param_name: scan_value})
                  .expand_dims(scan_param_name))
        if linked_keys:
            dat = dat.assign_coords({
                k: (scan_param_name, [float(linked_overrides[k][i])])
                for k in linked_keys
            })
        dat_out.append(dat)

    return xr.combine_by_coords(dat_out)


def _generate_iterable_tasks_2d(par1, par_range1, par2, par_range2,
                                initial_values_ds=None, iv_mapping=None, fixed_overrides=None):
    """Creates the nested task structure for a 2D parameter scan.

    Injects initial values only if they are finite AND biologically plausible.
    """
    outer_tasks = []
    for val1 in par_range1:
        inner_tasks = []
        for val2 in par_range2:
            scan_dict = {par1: val1, par2: val2}
            p1_sel_val = _scalarize_param_for_coord(val1)  # Assumes you have this helper
            p2_sel_val = _scalarize_param_for_coord(val2)  # Assumes you have this helper

            if initial_values_ds is not None and iv_mapping is not None:
                try:
                    iv_point = initial_values_ds.sel({par1: p1_sel_val, par2: p2_sel_val}, method='nearest')
                except KeyError:
                    iv_point = initial_values_ds.sel({par1: p1_sel_val, par2: p2_sel_val}, method='nearest')

                for var_name, init_param_name in iv_mapping.items():
                    if var_name in iv_point:
                        iv_value = iv_point[var_name].values

                        # --- FINAL CHECK ---
                        is_finite = np.isfinite(iv_value).all()
                        is_plausible = True
                        if 'biomass' in var_name or 'value' in var_name:
                            try:
                                is_plausible = np.all(iv_value >= 0)
                            except TypeError:
                                is_plausible = False

                        if is_finite and is_plausible:
                            scan_dict[init_param_name] = iv_value
                        else:
                            pass  # Use model default
                        # --- END FINAL CHECK ---
                    else:
                        print(f"Warning: '{var_name}' not found...")

            # Allow injection of parameters to scan:
            if fixed_overrides:
                scan_dict.update(fixed_overrides)

            inner_tasks.append({'scan_param_dict': scan_dict})
        outer_tasks.append((par1, val1, par2, inner_tasks))
    return outer_tasks


def _unpack_par_scan_2d(datasets):
    """Combines the nested results from a 2D parameter scan into a Dataset.

    ``None`` inner entries (failed cells) are replaced with a NaN-filled
    template built from the first successful cell found in any inner list.
    """
    template = None
    for _, _, _, inner_results_list in datasets:
        for _, ds in inner_results_list:
            if ds is not None:
                template = ds
                break
        if template is not None:
            break

    if template is None:
        print("[WARNING] All scan cells failed; returning empty Dataset.")
        return xr.Dataset()

    dats_out = []
    for par1, val1, par2, inner_results_list in datasets:
        dat_out = []
        scalar_val1 = _scalarize_param_for_coord(val1)

        for val2, ds in inner_results_list:
            if ds is None:
                ds = _nan_template_like(template)
            scalar_val2 = _scalarize_param_for_coord(val2)
            dat_out.append(
                ds.assign_coords({par1: scalar_val1, par2: scalar_val2})
                  .expand_dims(par1)
                  .expand_dims(par2)
            )

        data_combined_inner = xr.combine_by_coords(dat_out)
        dats_out.append(data_combined_inner)

    data_combined_2d = xr.combine_by_coords(dats_out)
    return data_combined_2d


def _run_inner_scan_with_progress(task_tuple):
    """Wraps the sequential execution of an inner 1D scan for a 2D scan.

    This function is executed by a worker in the *outer* parallel loop.
    It receives a tuple with the outer parameter's value and a list
    of inner tasks. It runs these inner tasks *sequentially* within
    the same worker, collects their results, and returns them along
    with the metadata.

    Parameters
    ----------
    task_tuple : tuple
        A tuple containing `(par1_name, par1_value, par2_name, inner_task_list)`,
        as generated by `_generate_iterable_tasks_2d`.

    Returns
    -------
    tuple
        A tuple containing `(par1_name, par1_value, par2_name, inner_results_list)`.
    """
    par1, val1, par2, inner_task_list = task_tuple

    # Since the worker is already initialized with the model, we run the inner tasks sequentially.
    inner_results = []

    for task_data in inner_task_list:
        # Get the actual parameter value for par2 (which could be an array)
        val2 = task_data['scan_param_dict'][par2]

        # Run the existing single-point function
        ds = _run_model_scan_point(task_data)

        # Append the TUPLE of (parameter_value, dataset_result)
        inner_results.append((val2, ds))

    # Return the metadata and results for unpacking
    return (par1, val1, par2, inner_results)


def _run_2d_parscan_core(model_file_name, param_name, param_values,
                         param_name2, param_values2, processes,
                         model_name, model_setup_name,
                         initial_values_ds, iv_mapping, fixed_overrides,
                         postprocess_name=None, postprocess_kwargs=None):
    """Manages the core execution and progress reporting for a 2D scan (solve_ivp)."""

    outer_tasks = _generate_iterable_tasks_2d(
        param_name, param_values, param_name2, param_values2,
        initial_values_ds, iv_mapping,
        fixed_overrides,
    )
    n_outer_points = len(param_values)

    nested_data = []
    start = tm.time()

    print(f"Starting 2D parallel scan of '{model_file_name}' ({param_name} x {param_name2}).")
    print(f"Outer loop ({n_outer_points} points) running in parallel with {processes} workers.")
    print(f"Inner loop ({len(param_values2)} points) running sequentially per worker.")
    print(f"Total points: {n_outer_points * len(param_values2)}.")

    try:
        with Pool(
                processes=processes,
                initializer=_worker_initializer,
                initargs=(model_file_name, model_name, model_setup_name,
                          postprocess_name, postprocess_kwargs,)
        ) as p:

            results_iterator = p.imap_unordered(_run_inner_scan_with_progress, outer_tasks)

            for i, result in enumerate(results_iterator, start=1):
                nested_data.append(result)

                p1_name = result[0]
                p1_val = result[1]

                clear_output(wait=True)

                print(f"PROGRESS: Completed {i}/{n_outer_points} outer points. ({p1_name} = {p1_val}).")

    except Exception as e:
        print(f"\n[ERROR] Parallel execution failed during 2D scan.")
        print(f"Details: {e}")
        return None, None

    end = tm.time()

    return nested_data, end - start


def run_xso_parscan(model_file_name, param_name, param_values, processes=20,
                    param_name2=None, param_values2=None,
                    model_name='model', model_setup_name='model_setup',
                    initial_values_ds=None, iv_mapping=None,
                    fixed_overrides=None, linked_overrides=None,
                    postprocess_name=None, postprocess_kwargs=None):
    """
    Executes a 1D or 2D parallel parameter scan (using solve_ivp).

    Parameters
    ----------
    ... (existing) ...
    linked_overrides : dict or None, optional
        Mapping of ``param_name -> sequence`` for a 1D *trajectory* scan:
        each linked parameter advances element-wise with ``param_values``
        rather than forming a cross-product (contrast ``param_name2``).
        Every sequence must have length ``len(param_values)``. At scan
        point ``i`` the applied inputs are ``{param_name: param_values[i]}``
        plus ``{k: linked_overrides[k][i] for k}`` plus ``fixed_overrides``.
        Linked keys must be disjoint from the scanned axes and from
        ``fixed_overrides``. Linked values are returned as coordinates
        indexed along ``param_name`` (present even for failed cells). Only
        supported for 1D scans (``param_name2 is None``).
    postprocess_name : str or None, optional
        Name of a callable in the user's model module applied to each
        run's output Dataset inside the worker, before the result is
        pickled back to the parent. Signature:
        ``postprocess(ds, **kwargs) -> xr.Dataset``. Typical use is to
        reduce the time dimension (e.g. averaging over a steady-state
        window) so large time series do not cross the process boundary.
        A canned ``avg_tail`` implementation is provided in
        ``xso.parscans``; re-export it from your model file
        (``from xso.parscans import avg_tail``) to reference
        it by name here.
    postprocess_kwargs : dict or None, optional
        Keyword arguments forwarded to the postprocess callable on every
        invocation. Set once per worker at pool init.

    Failed cells (solver exception, postprocess exception, or numerical
    blow-up producing NaN-filled output) are returned as NaN-filled
    sentinels with the same structure as successful cells, so one bad
    cell does not abort the whole scan.
    """

    print(f"--- Starting Parallel Scan (solve_ivp) ---")
    print(f"Validating model '{model_name}' and setup '{model_setup_name}' "
          f"from '{model_file_name}'"
          + (f" with postprocess '{postprocess_name}'"
             if postprocess_name else "")
          + "...")
    try:
        _validate_model_loading(model_file_name, model_name, model_setup_name,
                                postprocess_name=postprocess_name)
        print("Validation successful. Proceeding with scan.")
    except Exception:
        print("Scan aborted due to failed pre-flight check.")
        return None

    if initial_values_ds is not None:
        if iv_mapping is None:
            raise ValueError("iv_mapping must be provided if initial_values_ds is used.")
        print(f"Injecting initial values from dataset using mapping: {iv_mapping}")
    print(f"--------------------------------")

    scan_params = [param_name] + ([param_name2] if param_name2 is not None else [])
    _validate_fixed_overrides(fixed_overrides, scan_params)

    if linked_overrides and param_name2 is not None:
        raise NotImplementedError(
            "linked_overrides is only supported for 1D scans (param_name2 is "
            "None). Linking onto a 2D (param_name2) scan is not implemented."
        )
    _validate_linked_overrides(linked_overrides, param_values, scan_params,
                               fixed_overrides)

    if param_name2 is None:
        iter_scan = _generate_iterable_tasks_1d(
            param_name, param_values,
            initial_values_ds, iv_mapping,
            fixed_overrides,
            linked_overrides=linked_overrides,
        )
        n_points = len(param_values)

        print(f"Starting 1D parallel scan of '{model_file_name}' over "
              f"{n_points} points using {processes} workers...")

        start = tm.time()

        try:
            with Pool(
                    processes=processes,
                    initializer=_worker_initializer,
                    initargs=(model_file_name, model_name, model_setup_name,
                              postprocess_name, postprocess_kwargs,)
            ) as p:
                data = p.map(_run_model_scan_point, iter_scan)

        except Exception as e:
            print(f"\n[ERROR] Parallel execution failed.")
            print(f"Details: {e}")
            return None

        end = tm.time()
        out_ds = _unpack_par_scan_1d(iter_scan, data, linked_overrides=linked_overrides)
        print(f"1D Scan complete. Time taken: {round(end - start, 5)} seconds.")
        return out_ds

    else:
        if param_values2 is None:
            raise ValueError("param_values2 must be provided when param_name2 is specified.")

        nested_data, run_time = _run_2d_parscan_core(
            model_file_name, param_name, param_values,
            param_name2, param_values2, processes,
            model_name, model_setup_name,
            initial_values_ds, iv_mapping,
            fixed_overrides,
            postprocess_name=postprocess_name,
            postprocess_kwargs=postprocess_kwargs,
        )

        if nested_data is None:
            return None

        out_ds = _unpack_par_scan_2d(nested_data)
        print(f"\n2D Scan complete. Total Time taken: {round(run_time, 5)} seconds.")
        return out_ds


class StabilityAnalysisHook(xs.RuntimeHook):
    """
    Runtime hook class for capturing stability analysis results.

    This class-based approach allows for more control and state management
    compared to the function-based approach.

    Usage
    -----
    >>> hook = StabilityAnalysisHook()
    >>> with model:
    ...     output = model_setup.xsimlab.run(hooks=[hook])
    >>>
    >>> # Retrieve results
    >>> stability_data = hook.get_results()
    >>> if stability_data:
    ...     output.attrs['stability'] = stability_data
    """

    def __init__(self):
        super().__init__()
        self.stability_results = {}
        self._solver_found = False

    @xs.runtime_hook("initialize", "model", "post")
    def check_solver_type(self, model, context, state):
        """Check if the solver supports stability analysis at initialization."""
        try:
            backend = state.get(('Core', 'core'))
            if backend and hasattr(backend.solver, '__class__'):
                solver_name = backend.solver.__class__.__name__
                if 'Stability' in solver_name or 'Bifurcation' in solver_name:
                    self._solver_found = True
                    print(f"[HOOK] Detected {solver_name} - will capture stability results")
        except:
            pass

    @xs.runtime_hook("finalize", "model", "post")
    def capture_results(self, model, context, state):
        """Capture stability results after model finalization."""
        if not self._solver_found:
            return

        try:
            backend = state.get(('Core', 'core'))

            if backend:
                solver = backend.solver

                # Try different storage locations
                if hasattr(solver, 'stability_results'):
                    self.stability_results = solver.stability_results.copy()
                elif hasattr(backend, 'stability_metadata'):
                    self.stability_results = backend.stability_metadata.copy()

                if self.stability_results:
                    self._print_summary()

        except Exception as e:
            print(f"[HOOK WARNING] Could not capture stability results: {e}")

    def _print_summary(self):
        """Print a summary of captured stability results."""
        if 'stability' in self.stability_results:
            print(f"[HOOK] System stability: {self.stability_results['stability'].upper()}")
        if 'max_eigenvalue_real' in self.stability_results:
            print(f"[HOOK] Max eigenvalue real part: {self.stability_results['max_eigenvalue_real']:.4e}")

    def get_results(self):
        """Get the captured stability analysis results."""
        return self.stability_results.copy()

    def clear(self):
        """Clear stored results."""
        self.stability_results = {}
        self._solver_found = False


def _run_model_scan_stability(task_data):
    """Executes a single simulation run with stability analysis."""

    scan_param_dict = task_data['scan_param_dict']

    model = _global_model
    model_setup = _global_model_setup

    if model is None or model_setup is None:
        raise RuntimeError("Model objects were not initialized in this worker process.")

    # Create stability hook
    hook = StabilityAnalysisHook()

    # Solver-internal diagnostics now flow through the xso.solvers
    # logger rather than stdout, so per-cell output is silent by
    # default (no stdout redirection needed). Callers who want to see
    # convergence warnings can configure logging in the worker, e.g.
    # by setting XSO_LOG_LEVEL in the worker initializer.
    with model:
        model_out = model_setup.xsimlab.update_vars(
            input_vars=scan_param_dict
        ).xsimlab.run(hooks=[hook])

    scalar_coords = {
        p_name: p_value for p_name, p_value in scan_param_dict.items()
        if np.isscalar(p_value)
    }
    model_out = model_out.assign_coords(scalar_coords)

    stability_data = hook.get_results()

    try:
        # Try to get the results
        if stability_data and 'stability' in stability_data and 'max_eigenvalue_real' in stability_data:
            model_out['stability'] = stability_data['stability']
            model_out['max_eigenvalue'] = stability_data['max_eigenvalue_real']
            model_out['n_positive_eigenvalues'] = (
                (), stability_data.get('n_positive_eigenvalues', np.nan))
            model_out['n_complex_pairs'] = (
                (), stability_data.get('n_complex_pairs', np.nan))

            # Imag part of the eigenvalue with the largest real part
            eigs_real = stability_data.get('eigenvalues_real')
            eigs_imag = stability_data.get('eigenvalues_imag')
            if eigs_real is not None and eigs_imag is not None and len(eigs_real) > 0:
                idx_max = int(np.argmax(eigs_real))
                model_out['max_eigenvalue_imag'] = ((), float(eigs_imag[idx_max]))
            else:
                model_out['max_eigenvalue_imag'] = ((), np.nan)
        else:
            model_out['stability'] = ((), np.nan)
            model_out['max_eigenvalue'] = ((), np.nan)
            model_out['n_positive_eigenvalues'] = ((), np.nan)
            model_out['n_complex_pairs'] = ((), np.nan)
            model_out['max_eigenvalue_imag'] = ((), np.nan)

    except Exception as e:
        # Catch any other unexpected error during result extraction
        print(f"WARNING: Worker failed to extract stability results. Details: {e}")
        model_out['stability'] = ((), np.nan)
        model_out['max_eigenvalue'] = ((), np.nan)

    # Standardize time coordinate
    if 'time' in model_out.coords:
        model_out['time'] = model_out.time.round(9)

    return model_out


def _run_2d_stabilityscan_core(model_file_name, param_name, param_values, param_name2, param_values2, processes,
                               model_name, model_setup_name,
                               initial_values_ds, iv_mapping, fixed_overrides=None):
    """Manages the core execution and progress reporting for a 2D stability scan."""

    # Tasks for the outer parameter (P1)
    outer_tasks = _generate_iterable_tasks_2d(
        param_name, param_values, param_name2, param_values2,
        initial_values_ds, iv_mapping, fixed_overrides
    )
    n_outer_points = len(param_values)

    # Storage for results and timing
    nested_data = []
    start = tm.time()

    print(f"Starting 2D stability scan of '{model_file_name}' ({param_name} x {param_name2}).")
    print(f"Outer loop ({n_outer_points} points) running in parallel with {processes} workers.")
    print(f"Inner loop ({len(param_values2)} points) running sequentially per worker.")
    print(f"Total points: {n_outer_points * len(param_values2)}.")

    try:
        # Start Pool for the outer (P1) parallel scan
        with Pool(
                processes=processes,
                initializer=_worker_initializer,
                # --- MODIFIED LINE ---
                initargs=(model_file_name, model_name, model_setup_name,)
        ) as p:

            # Use p.imap_unordered for better progress tracking
            results_iterator = p.imap_unordered(_run_inner_stability_scan, outer_tasks)

            # --- Progress Tracking Loop ---
            for i, result in enumerate(results_iterator, start=1):
                nested_data.append(result)

                # Print progress for the outer loop
                p1_name = result[0]
                p1_val = result[1]

                # <-- REQUIRED JUPYTER CLEAR OUTPUT -->
                clear_output(wait=True)

                print(f"PROGRESS: Completed {i}/{n_outer_points} outer points. ({p1_name} = {p1_val}).")

    except Exception as e:
        print(f"\n[ERROR] Parallel execution failed during 2D stability scan.")
        print(f"Details: {e}")
        return None, None

    end = tm.time()

    return nested_data, end - start


def _run_inner_stability_scan(task_tuple):
    """Wraps the sequential execution of an inner 1D stability scan for a 2D scan."""
    par1, val1, par2, inner_task_list = task_tuple

    # Run the inner tasks sequentially with stability analysis
    inner_results = []

    for task_data in inner_task_list:
        # Get the actual parameter value for par2 (which could be an array)
        # We need this to pass back to the unpacker
        val2 = task_data['scan_param_dict'][par2]

        # Run the stability analysis version
        ds = _run_model_scan_stability(task_data)

        # Append the TUPLE of (parameter_value, dataset_result)
        inner_results.append((val2, ds))

    # Return the metadata and results for unpacking
    return (par1, val1, par2, inner_results)


def run_xso_stabilityscan(model_file_name, param_name, param_values, processes=20,
                          param_name2=None, param_values2=None,
                          model_name='model', model_setup_name='model_setup',
                          initial_values_ds=None, iv_mapping=None, fixed_overrides=None):
    """Executes a 1D or 2D parallel parameter scan with stability analysis for each run.

    This function performs the same parameter scan as run_xso_parscan but includes
    stability analysis at each parameter point using the NumericalStabilitySolver.
    Stability results are stored in the dataset attributes.

    Parameters
    ----------
    model_file_name : str
        The name of the Python file (module) containing the
        'model' and 'model_setup' objects (e.g., 'my_model').
    param_name : str
        The name of the primary parameter to scan (e.g., 'Component__variable').
    param_values : iterable
        Array or sequence of values for the primary parameter.
    processes : int, optional
        Number of parallel processes to use. Default is 20.
    param_name2 : str, optional
        The name of the second parameter for a 2D scan. Default is None.
    param_values2 : iterable, optional
        Array or sequence of values for the second parameter.
        Required if `param_name2` is given. Default is None.
    model_name : str, optional
        The string name of the model object to load (e.g., 'model').
    model_setup_name : str, optional
        The string name of the model_setup object to load (e.g., 'model_setup_ivp').
    initial_values_ds : xarray.Dataset, optional
        A dataset (e.g., from a previous `run_xso_parscan`) containing
        the initial values to use for this scan. The dataset must
        be indexed by the same scan parameters (e.g., `param_name`).
    iv_mapping : dict, optional
        A dictionary mapping variable names from `initial_values_ds`
        to the model's *initial value parameter names*.
        Example: {'Phytoplankton__value': 'Phytoplankton__value_init'}

    Returns
    -------
    xarray.Dataset
        The combined `xarray.Dataset` of the scan results with stability
        analysis stored in attributes. Returns `None` if the execution failed.
    """

    print(f"--- Starting Stability Scan ---")
    print(f"Validating model '{model_name}' and setup '{model_setup_name}' from '{model_file_name}'...")
    try:
        _validate_model_loading(model_file_name, model_name, model_setup_name)
        print("Validation successful. Proceeding with scan.")
    except Exception:
        print("Scan aborted due to failed pre-flight check.")
        return None

    if initial_values_ds is not None:
        if iv_mapping is None:
            raise ValueError("iv_mapping must be provided if initial_values_ds is used.")
        print(f"Injecting initial values from dataset using mapping: {iv_mapping}")
    print(f"--------------------------------")

    scan_params = [param_name] + ([param_name2] if param_name2 is not None else [])
    _validate_fixed_overrides(fixed_overrides, scan_params)

    if param_name2 is None:
        # --- 1D Scan Logic ---
        iter_scan = _generate_iterable_tasks_1d(
            param_name, param_values,
            initial_values_ds, iv_mapping,
            fixed_overrides
        )
        n_points = len(param_values)

        print(
            f"Starting 1D stability scan of '{model_file_name}' over {n_points} points using {processes} workers...")

        start = tm.time()

        try:
            with Pool(
                    processes=processes,
                    initializer=_worker_initializer,
                    initargs=(model_file_name, model_name, model_setup_name,)
            ) as p:
                data = p.map(_run_model_scan_stability, iter_scan)

        except Exception as e:
            print(f"\n[ERROR] Parallel execution failed.")
            print(f"Details: {e}")
            return None

        end = tm.time()
        out_ds = _unpack_par_scan_1d(iter_scan, data)
        print(f"1D Stability Scan complete. Time taken: {round(end - start, 5)} seconds.")
        return out_ds

    else:
        # --- 2D Scan Logic ---
        if param_values2 is None:
            raise ValueError("param_values2 must be provided when param_name2 is specified.")

        nested_data, run_time = _run_2d_stabilityscan_core(
            model_file_name, param_name, param_values, param_name2, param_values2, processes,
            model_name, model_setup_name,
            initial_values_ds, iv_mapping,
            fixed_overrides
        )

        if nested_data is None:
            return None

        # 3. Combine Results and Report
        out_ds = _unpack_par_scan_2d(nested_data)
        print(f"\n2D Stability Scan complete. Total Time taken: {round(run_time, 5)} seconds.")
        return out_ds



"""Canned postprocess hooks for use with ``run_xso_parscan``.

Each hook has signature ``(ds, **kwargs) -> xr.Dataset`` and is meant to
be referenced by name via the ``postprocess_name`` argument of
``run_xso_parscan``. To use one, re-export it from your model setup
module so the worker can locate it by attribute lookup::

    # cariaco_ssm_setup.py
    from xso.parscans_postprocess import avg_tail

    # run_2d_scan.py
    run_xso_parscan(
        model_file_name='cariaco_ssm_setup',
        ...,
        postprocess_name='avg_tail',
        postprocess_kwargs={'avg_window': 1000},
    )
"""


def avg_tail(ds, avg_window=1000):
    """Reduce every time-dimensioned variable to its mean over the last
    ``avg_window`` time steps, keeping ``time`` as a length-1 dim.

    Variables without a ``time`` dim (including string/object-dtype
    metadata variables from xsimlab input storage) are passed through
    unchanged.
    """
    if 'time' not in ds.dims:
        return ds

    final_t = ds['time'].values[-1]

    vars_with_time = [v for v in ds.data_vars if 'time' in ds[v].dims]
    vars_without_time = [v for v in ds.data_vars if 'time' not in ds[v].dims]

    if vars_with_time:
        tail = ds[vars_with_time].isel(time=slice(-avg_window, None))
        reduced = tail.mean('time', keep_attrs=True)
        reduced = reduced.expand_dims({'time': [final_t]})
    else:
        reduced = xr.Dataset()

    if vars_without_time:
        out = xr.merge([reduced, ds[vars_without_time]])
    else:
        out = reduced

    out.attrs = ds.attrs
    return out


# =============================================================================
# Generic streaming task pool (model-agnostic) -- added 2026-06-22.
#
# run_xso_parscan sweeps ONE model over a parameter grid (shared base setup +
# update_vars per cell). run_parallel_tasks is its sibling for the OTHER access
# pattern: an iterable of INDEPENDENT, self-contained tasks (each builds its own
# xso.setup and runs), dispatched across a process pool with results streaming
# back as they finish. It knows nothing about the model, scoring, or output
# schema -- the caller supplies an importable worker and an on_result hook.
#
# Use this when tasks differ by more than a parameter value (e.g. whole forcing
# arrays per era), so there is no single base setup to update_vars from.
#
# Solver diagnostics (instability / stiffness UserWarnings, convergence logs)
# stay a model-side concern: configure the `xso.solvers` logger at the top of
# your model module so each worker inherits it at import (see handoff S11). The
# driver deliberately does not reach into solver internals.
# =============================================================================
def _apply(packed):
    """Top-level worker wrapper run inside each pool process.

    Runs ``worker(*args)`` under try/except so one failure returns a sentinel
    (carrying the task index) instead of crashing the pool -- the same
    catch-and-return-sentinel contract as ``_run_model_scan_point`` returning
    ``None``, but the index lets the parent map an unordered result back to its
    task and build its own error record. Stays silent (no worker-side prints):
    the parent owns the console, so error lines never interleave across workers.

    Returns ``(index, ok, payload)`` -- on success ``payload`` is the worker's
    return value; on failure ``ok`` is False and ``payload`` is the exception
    repr.
    """
    index, worker, args = packed
    try:
        return index, True, worker(*args)
    except Exception as e:
        return index, False, repr(e)


def _validate_parallel_inputs(worker, tasks):
    """Pre-flight check in the main process, before any worker spawns.

    Mirrors the spirit of ``_validate_model_loading``: fail fast with a clear
    message rather than after pool start-up. Verifies the worker is callable and
    importable by reference (spawn/forkserver workers re-import it -- a worker
    defined in ``__main__`` or as a closure cannot be pickled), and that there
    is work to do.
    """
    if not callable(worker):
        raise TypeError(f"worker must be callable, got {type(worker).__name__}.")
    qualname = getattr(worker, "__qualname__", "")
    if getattr(worker, "__module__", None) == "__main__":
        raise ValueError(
            f"worker '{qualname}' is defined in __main__; spawn/forkserver "
            f"workers re-import it by reference and cannot resolve __main__ "
            f"functions. Move it into an importable module."
        )
    if "<locals>" in qualname:
        raise ValueError(
            f"worker '{qualname}' is a closure/local function and is not "
            f"picklable. Make it a module-level function."
        )
    if len(tasks) == 0:
        raise ValueError("tasks is empty -- nothing to run.")


def run_parallel_tasks(worker, tasks, processes=None, on_result=None,
                       progress=True, label="scan", tally_flags=None,
                       abort_after_errors=None, pin_threads=True):
    """Run an iterable of independent tasks across a process pool, streaming.

    The model-agnostic sibling of ``run_xso_parscan`` for heterogeneous,
    self-contained tasks. Results stream back via ``imap_unordered`` as each
    finishes, so progress + ETA are live and the caller can persist
    incrementally from the parent.

    Parameters
    ----------
    worker : callable
        An importable, module-level function (NOT a closure or ``__main__``
        function -- spawn/forkserver workers re-import it by reference). Called
        as ``worker(*task)``. Its defining module is imported once per worker
        (cached), so heavy model construction at module import is amortised
        across that worker's tasks -- the same amortisation
        ``_worker_initializer`` gives ``run_xso_parscan``.
    tasks : sequence of tuple
        Each task is the positional-argument tuple for ``worker``. Must be
        sized (``len`` is the progress denominator); a generator is materialised.
    processes : int, optional
        Pool size. Default ``os.cpu_count() - 1``.
    on_result : callable, optional
        Per-completion hook ``on_result(item, done, n)`` called in the PARENT
        process (single writer -- safe for incremental save). ``item`` is a dict
        ``{'index', 'ok', 'result'|'error'}``. Use it to score, assemble, and
        dump the caller's own records. Return value is ignored.
    progress : bool
        If True, print one plain line per completion (count, error + flag
        tallies, elapsed, ETA). Set False if the caller prints its own per-result
        line in ``on_result``. Errors are surfaced regardless of this flag.
    label : str
        Tag shown in the progress lines / summary.
    tally_flags : list of str, optional
        Keys to count on successful result dicts (e.g. ``['has_nan']``). Each
        truthy occurrence is counted and shown live + in the summary -- a running
        "how many cells blew up" without the driver knowing what the flag means.
    abort_after_errors : int, optional
        If the FIRST ``abort_after_errors`` completed tasks are ALL errors,
        terminate the pool and raise ``RuntimeError`` -- a systemic-failure guard
        (wrong import, bad setup) so a long scan does not burn the whole run on a
        bug. Off by default.
    pin_threads : bool
        If True, pin BLAS/OMP thread env to 1 before the pool starts so spawned
        workers import numpy single-threaded (avoids N workers x multithreaded
        BLAS oversubscription). Uses ``setdefault`` -- an explicit caller setting
        wins. Relies on the spawn/forkserver start method (workers re-import
        numpy); effectively a no-op under fork.

    Returns
    -------
    list of dict
        The ``item`` dicts in task/input order (index-aligned): raw results +
        error sentinels. Callers that persist via ``on_result`` can ignore this.
    """
    tasks = list(tasks)
    _validate_parallel_inputs(worker, tasks)

    if processes is None:
        processes = max(1, (os.cpu_count() or 2) - 1)
    if pin_threads:
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(_v, "1")

    n = len(tasks)
    tally_flags = list(tally_flags or [])
    packed = [(i, worker, args) for i, args in enumerate(tasks)]

    records = [None] * n                       # index-aligned -> stable input order
    flag_counts = {k: 0 for k in tally_flags}
    n_err = 0
    width = len(str(n))
    start = tm.time()

    print(f"--- run_parallel_tasks: {n} tasks, {processes} workers ({label}) ---",
          flush=True)

    with Pool(processes=processes) as p:
        for done, (index, ok, payload) in enumerate(
                p.imap_unordered(_apply, packed), start=1):
            if ok:
                item = {"index": index, "ok": True, "result": payload}
                if isinstance(payload, dict):
                    for k in tally_flags:
                        if payload.get(k):
                            flag_counts[k] += 1
            else:
                item = {"index": index, "ok": False, "error": payload}
                n_err += 1
                print(f"  [err] {label} task {index}: {payload}", flush=True)

            records[index] = item

            if on_result is not None:
                on_result(item, done, n)

            if progress:
                el = tm.time() - start
                eta = el / done * (n - done)
                extra = "".join(f" {k}={flag_counts[k]}" for k in tally_flags)
                print(f"  {done:>{width}}/{n} {label} | err={n_err}{extra} "
                      f"| {el / 60:.1f}m ETA {eta / 60:.1f}m", flush=True)

            if (abort_after_errors is not None and done == abort_after_errors
                    and n_err == done):
                p.terminate()
                raise RuntimeError(
                    f"run_parallel_tasks aborted: first {done} tasks all failed "
                    f"(systemic error, not a bad cell). Last error: "
                    f"{item.get('error')}. Fix the cause and re-run, or pass "
                    f"abort_after_errors=None to disable this guard."
                )

    el = tm.time() - start
    n_ok = n - n_err
    summary = "".join(f", {k}={flag_counts[k]}" for k in tally_flags)
    print(f"--- {label} done: {n_ok}/{n} ok, {n_err} errored{summary} "
          f"| {el / 60:.1f}m ---", flush=True)
    return records