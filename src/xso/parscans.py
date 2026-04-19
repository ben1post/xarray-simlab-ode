import importlib
import warnings
import xarray as xr
import xsimlab as xs
import time as tm
import numpy as np
from multiprocessing import Pool
from IPython.display import clear_output
import os
from contextlib import redirect_stdout, nullcontext

# Suppress warnings that clutter output, especially common in scientific computing libraries
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global variables (must be defined outside functions for multiprocessing)
_global_model = None
_global_model_setup = None


def _validate_model_loading(module_name, model_name_str, model_setup_name_str):
    """
    Performs a pre-flight check in the main process to ensure
    the model and setup objects can be loaded before starting workers.

    Raises ImportError or AttributeError if loading fails.
    """
    try:
        model_module = importlib.import_module(module_name)

        # Test-load the model
        try:
            getattr(model_module, model_name_str)
        except AttributeError as e:
            print(f"\n[FATAL PRE-CHECK ERROR] Failed to find model object '{model_name_str}' in '{module_name}.py'.")
            # Try to find suggestions
            suggestions = [o for o in dir(model_module) if 'model' in o.lower() and not o.startswith('__')]
            if suggestions:
                print(f"Did you mean one of these? {suggestions}")
            raise e  # Re-raise to stop execution

        # Test-load the model_setup
        try:
            getattr(model_module, model_setup_name_str)
        except AttributeError as e:
            print(
                f"\n[FATAL PRE-CHECK ERROR] Failed to find setup object '{model_setup_name_str}' in '{module_name}.py'.")
            # Try to find suggestions
            suggestions = [o for o in dir(model_module) if 'setup' in o.lower() and not o.startswith('__')]
            if suggestions:
                print(f"Did you mean one of these? {suggestions}")
            raise e  # Re-raise to stop execution

    except ImportError as e:
        # --- MODIFIED ERROR BLOCK ---
        print(f"\n[FATAL PRE-CHECK ERROR] Failed to import model file: '{module_name}.py'.")
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
        # --- END OF MODIFICATION ---
        raise e  # Re-raise to stop execution

    except Exception as e:
        print(f"\n[FATAL PRE-CHECK ERROR] An unexpected error occurred while loading '{module_name}'.")
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


def _worker_initializer(module_name, model_name_str, model_setup_name_str):
    """Initializes a worker process for multiprocessing.

    Dynamically imports the model and model_setup objects from the given
    module name and stores them in the worker's global scope
    (`_global_model`, `_global_model_setup`). This ensures each worker
    process has access to the model without serializing it.

    Parameters
    ----------
    module_name : str
        The name of the Python module (file) that contains the
        model and model_setup objects.
    model_name_str : str
        The string name of the model object to load (e.g., 'model').
    model_setup_name_str : str
        The string name of the model_setup object to load (e.g., 'model_setup_ivp').
    """
    global _global_model
    global _global_model_setup

    try:
        model_module = importlib.import_module(module_name)

        # Use getattr to dynamically fetch attributes by their string names
        _global_model = getattr(model_module, model_name_str)
        _global_model_setup = getattr(model_module, model_setup_name_str)

    except Exception as e:
        # Updated error message for better debugging
        print(f"FATAL ERROR: Worker failed to initialize model/setup from '{module_name}'.")
        print(f"Attempted to load: model='{model_name_str}', setup='{model_setup_name_str}'.")
        print(f"Check that these objects exist in your script. Details: {e}", flush=True)
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

    Retrieves the model from the worker's global scope. It then updates
    the model's input variables with the parameters from `task_data`
    and executes the simulation. The scan parameters are added to the
    output Dataset as coordinates, and the 'time' coordinate is standardized.

    Parameters
    ----------
    task_data : dict
        A dictionary containing the task information. Must have a
        key `'scan_param_dict'` which holds another dictionary
        mapping parameter names to their values for this specific run.

    Returns
    -------
    xarray.Dataset
        The result of the single model run, with scan parameters
        as coordinates.
    """
    scan_param_dict = task_data['scan_param_dict']

    model = _global_model
    model_setup = _global_model_setup

    if model is None or model_setup is None:
        raise RuntimeError("Model objects were not initialized in this worker process.")

    # Run the Model: Update variables with the scan point and execute
    with model:
        model_out = model_setup.xsimlab.update_vars(
            input_vars=scan_param_dict
        ).xsimlab.run()

    # Ensure all SCALAR scan parameters are added to the output coordinates
    # This avoids adding the non-scalar array, which causes the name conflict.
    scalar_coords = {
        p_name: p_value for p_name, p_value in scan_param_dict.items()
        if np.isscalar(p_value)
    }
    model_out = model_out.assign_coords(scalar_coords)

    # Post-processing: Standardize time coordinate for clean merging
    if 'time' in model_out.coords:
        model_out['time'] = model_out.time.round(9)

    return model_out


def _generate_iterable_tasks_1d(parameter, par_range,
                                initial_values_ds=None, iv_mapping=None, fixed_overrides=None):
    """Creates the list of task dictionaries for a 1D parameter scan.

    Injects initial values only if they are finite AND biologically plausible.
    """
    tasks = []
    for val in par_range:
        scan_dict = {parameter: val}
        if initial_values_ds is not None and iv_mapping is not None:
            iv_point = initial_values_ds.sel({parameter: val}, method='nearest')
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

        # Allow passing parameters to scan
        if fixed_overrides:
            scan_dict.update(fixed_overrides)

        tasks.append({'scan_param_dict': scan_dict})
    return tasks


def _unpack_par_scan_1d(iterable_tasks, data):
    """Combines the results from a 1D parameter scan into a single Dataset.

    Takes the list of completed tasks and the corresponding list of
    `xarray.Dataset` results. It assigns the scan parameter as a
    new coordinate and dimension for each dataset and then merges
    them all using `xr.combine_by_coords`.

    Parameters
    ----------
    iterable_tasks : list
        The list of task dictionaries that was used to generate the runs.
    data : list
        The list of `xarray.Dataset` objects returned by the worker processes.

    Returns
    -------
    xarray.Dataset
        A single Dataset containing all results combined along
        the new parameter dimension.
    """
    if not iterable_tasks or not data:
        return xr.Dataset()

    # Determine the scan parameter name from the first task
    first_dict = iterable_tasks[0]['scan_param_dict']
    scan_param_name = list(first_dict.keys())[0]

    dat_out = []
    for dat, task in zip(data, iterable_tasks):
        scan_value = task['scan_param_dict'][scan_param_name]

        # Assign the scan value as a coordinate and expand the dimension
        dat_out.append(
            dat.assign_coords({scan_param_name: scan_value}).expand_dims(scan_param_name)
        )

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
    """Combines the nested results from a 2D parameter scan.

    Takes the nested list of results (where each item corresponds
    to an outer loop value and contains a list of inner loop datasets).
    It iterates through both loops, assigns the two scan parameters
    as coordinates and dimensions, and combines everything into a
    single 2D `xarray.Dataset`.

    Parameters
    ----------
    datasets : list
        The nested list of results returned by `_run_inner_scan_with_progress`,
        matching the structure from `_generate_iterable_tasks_2d`.

    Returns
    -------
    xarray.Dataset
        A single Dataset containing all 2D scan results.
    """
    dats_out = []

    # Outer loop: Iterates over the first parameter (p1)
    # 'inner_datasets' is now a list of (val2, ds) tuples
    for par1, val1, par2, inner_results_list in datasets:
        dat_out = []

        # Get a scalar coordinate value for P1
        scalar_val1 = _scalarize_param_for_coord(val1)

        # Inner loop: Iterates over the second parameter (p2)
        for val2, ds in inner_results_list:  # Unpack the (val2, ds) tuple

            # Get a scalar coordinate value for P2
            scalar_val2 = _scalarize_param_for_coord(val2)

            # Combine the P1 and P2 coordinates and create the dimensions
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


def _run_2d_parscan_core(model_file_name, param_name, param_values, param_name2, param_values2, processes,
                         model_name, model_setup_name,
                         initial_values_ds, iv_mapping, fixed_overrides):  # <-- NEW ARGS
    """Manages the core execution and progress reporting for a 2D scan (solve_ivp)."""

    # Tasks for the outer parameter (P1)
    outer_tasks = _generate_iterable_tasks_2d(
        param_name, param_values, param_name2, param_values2,
        initial_values_ds, iv_mapping,
        fixed_overrides,
    )
    n_outer_points = len(param_values)

    # Storage for results and timing
    nested_data = []
    start = tm.time()

    print(f"Starting 2D parallel scan of '{model_file_name}' ({param_name} x {param_name2}).")
    print(f"Outer loop ({n_outer_points} points) running in parallel with {processes} workers.")
    print(f"Inner loop ({len(param_values2)} points) running sequentially per worker.")
    print(f"Total points: {n_outer_points * len(param_values2)}.")

    try:
        # Start Pool for the outer (P1) parallel scan
        with Pool(
                processes=processes,
                initializer=_worker_initializer,
                initargs=(model_file_name, model_name, model_setup_name,)
        ) as p:

            # Use p.imap_unordered for better progress tracking
            # This calls the original _run_inner_scan_with_progress
            results_iterator = p.imap_unordered(_run_inner_scan_with_progress, outer_tasks)

            # --- Progress Tracking Loop ---
            for i, result in enumerate(results_iterator, start=1):
                nested_data.append(result)

                # Print progress for the outer loop
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
                    fixed_overrides=None):
    """
    Executes a 1D or 2D parallel parameter scan (using solve_ivp).

    Can optionally be initialized from a dataset of steady states.
    ... (rest of docstring) ...
    """

    # --- Pre-flight check (unchanged) ---
    print(f"--- Starting Parallel Scan (solve_ivp) ---")
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
            fixed_overrides,
        )
        n_points = len(param_values)

        print(f"Starting 1D parallel scan of '{model_file_name}' over {n_points} points using {processes} workers...")

        start = tm.time()

        try:
            with Pool(
                    processes=processes,
                    initializer=_worker_initializer,
                    initargs=(model_file_name, model_name, model_setup_name,)
            ) as p:
                # Use the original _run_model_scan_point for IVP
                data = p.map(_run_model_scan_point, iter_scan)

        except Exception as e:
            print(f"\n[ERROR] Parallel execution failed.")
            print(f"Details: {e}")
            return None

        end = tm.time()
        out_ds = _unpack_par_scan_1d(iter_scan, data)
        print(f"1D Scan complete. Time taken: {round(end - start, 5)} seconds.")
        return out_ds

    else:
        # --- 2D Scan Logic ---
        if param_values2 is None:
            raise ValueError("param_values2 must be provided when param_name2 is specified.")

        nested_data, run_time = _run_2d_parscan_core(
            model_file_name, param_name, param_values,
            param_name2, param_values2, processes,
            model_name, model_setup_name,
            initial_values_ds, iv_mapping,
            fixed_overrides,
        )

        if nested_data is None:
            return None

        # 3. Combine Results and Report
        out_ds = _unpack_par_scan_2d(nested_data)
        print(f"\n2D Scan complete. Total Time taken: {round(run_time, 5)} seconds.")
        return out_ds


# Alternative: Class-based approach for more control
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

    # (This is the hard-coded suppression from your script)
    ctx_manager = redirect_stdout(open(os.devnull, 'w'))

    # Run the Model with stability hook *inside* the context manager
    with ctx_manager:
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
        else:
            # The hook ran but results are missing (e.g., due to default IVs failing)
            # Assign NaN to these coordinates so the point is marked as failed.
            model_out['stability'] = ((), np.nan)  # Using xarray tuple format for a scalar
            model_out['max_eigenvalue'] = ((), np.nan)

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
                               initial_values_ds, iv_mapping):
    """Manages the core execution and progress reporting for a 2D stability scan."""

    # Tasks for the outer parameter (P1)
    outer_tasks = _generate_iterable_tasks_2d(
        param_name, param_values, param_name2, param_values2,
        initial_values_ds, iv_mapping  # <-- PASS NEW ARGS
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
                          initial_values_ds=None, iv_mapping=None):
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

    if param_name2 is None:
        # --- 1D Scan Logic ---
        iter_scan = _generate_iterable_tasks_1d(
            param_name, param_values,
            initial_values_ds, iv_mapping  # <-- PASS NEW ARGS
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
            initial_values_ds, iv_mapping  # <-- PASS NEW ARGS
        )

        if nested_data is None:
            return None

        # 3. Combine Results and Report
        out_ds = _unpack_par_scan_2d(nested_data)
        print(f"\n2D Stability Scan complete. Total Time taken: {round(run_time, 5)} seconds.")
        return out_ds
