import importlib
import warnings
import xarray as xr
import time as tm
from multiprocessing import Pool
from IPython.display import clear_output

# Suppress warnings that clutter output, especially common in scientific computing libraries
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global variables (must be defined outside functions for multiprocessing)
_global_model = None
_global_model_setup = None


def _worker_initializer(module_name):
    """Initializes a worker process for multiprocessing.

    Dynamically imports the model and model_setup objects from the given
    module name and stores them in the worker's global scope
    (`_global_model`, `_global_model_setup`). This ensures each worker
    process has access to the model without serializing it.

    Parameters
    ----------
    module_name : str
        The name of the Python module (file) that contains the
        `model` and `model_setup` objects.
    """
    global _global_model
    global _global_model_setup

    try:
        model_module = importlib.import_module(module_name)
        _global_model = model_module.model
        _global_model_setup = model_module.model_setup

    except Exception as e:
        # Prints to the terminal if a worker fails to load the model
        print(f"FATAL ERROR: Worker failed to initialize model and model setup from '{module_name}'. "
              f"Check that naming in script is coherent (model, model_setup). Details: {e}", flush=True)
        raise e


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

    # Ensure all scan parameters are added to the output coordinates
    model_out = model_out.assign_coords(
        {p_name: p_value for p_name, p_value in scan_param_dict.items()}
    )

    # Post-processing: Standardize time coordinate for clean merging
    if 'time' in model_out.coords:
        model_out['time'] = model_out.time.round(9)

    return model_out


def _generate_iterable_tasks_1d(parameter, par_range):
    """Creates the list of task dictionaries for a 1D parameter scan.

    Generates a list where each item is a dictionary formatted for
    `_run_model_scan_point`. Each dictionary contains the parameter
    name and one value from the `par_range`.

    Parameters
    ----------
    parameter : str
        The name of the parameter being scanned.
    par_range : iterable
        An iterable (e.g., list, numpy array) of values for the parameter.

    Returns
    -------
    list
        A list of task dictionaries.
    """
    tasks = []
    for val in par_range:
        tasks.append({'scan_param_dict': {parameter: val}})
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


def _generate_iterable_tasks_2d(par1, par_range1, par2, par_range2):
    """Creates the nested task structure for a 2D parameter scan.

    Generates an "outer" list of tasks. Each item in this list corresponds
    to one value of the *first* parameter (`par1`) and contains all the
    necessary information for a *full 1D scan* of the *second*
    parameter (`par2`). This structure is designed for a "parallel
    outer loop, sequential inner loop" execution.

    Parameters
    ----------
    par1 : str
        The name of the first (outer) parameter.
    par_range1 : iterable
        An iterable of values for the first parameter.
    par2 : str
        The name of the second (inner) parameter.
    par_range2 : iterable
        An iterable of values for the second parameter.

    Returns
    -------
    list
        A list of tuples. Each tuple contains
        `(par1_name, par1_value, par2_name, inner_task_list)`.
    """
    # Outer iterable format:
    # (param1_name, param1_value, param2_name, [inner_task_dict_1, inner_task_dict_2, ...])
    outer_tasks = []
    for val1 in par_range1:
        # Create the iterable for the inner (p2) scan tasks
        # Each inner task now contains BOTH P1 and P2 values
        inner_tasks = [{'scan_param_dict': {par1: val1, par2: val2}} for val2 in par_range2]

        # Store metadata and inner tasks for processing
        outer_tasks.append(
            (par1, val1, par2, inner_tasks)
        )
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
    for par1, val1, par2, inner_datasets in datasets:
        dat_out = []

        # Inner loop: Iterates over the second parameter (p2)
        for ds in inner_datasets:

            if par2 not in ds.coords:
                raise ValueError(
                    f"Second scan parameter '{par2}' not found as a coordinate in inner scan output. Cannot unpack 2D scan results.")

            # Use .item() to safely extract the scalar value from the coordinate
            par2_value = ds.coords[par2].values.item()

            # Combine the P1 and P2 coordinates and create the dimensions
            dat_out.append(
                ds.assign_coords({par1: val1, par2: par2_value})
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
        # Run the existing single-point function
        inner_results.append(_run_model_scan_point(task_data))

    # Return the metadata and results for unpacking
    return (par1, val1, par2, inner_results)


def _run_2d_parscan_core(model_file_name, param_name, param_values, param_name2, param_values2, processes):
    """Manages the core execution and progress reporting for a 2D scan.

    Sets up the outer loop tasks and initializes the `multiprocessing.Pool`.
    It uses `imap_unordered` to process the outer loop tasks in parallel,
    which allows for real-time progress updates as each inner scan
    (a single outer task) completes. It prints progress to the console
    and handles exceptions.

    Parameters
    ----------
    model_file_name : str
        The name of the Python module to be loaded by workers.
    param_name : str
        Name of the first (outer) parameter.
    param_values : iterable
        Values for the first parameter.
    param_name2 : str
        Name of the second (inner) parameter.
    param_values2 : iterable
        Values for the second parameter.
    processes : int
        Number of parallel worker processes.

    Returns
    -------
    tuple
        A tuple containing `(nested_data, run_time)`. `nested_data` is
        the list of results, and `run_time` is the total execution time
        in seconds. Returns `(None, None)` on failure.
    """

    # Tasks for the outer parameter (P1)
    outer_tasks = _generate_iterable_tasks_2d(
        param_name, param_values, param_name2, param_values2
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
                initargs=(model_file_name,)
        ) as p:

            # Use p.imap_unordered for better progress tracking
            results_iterator = p.imap_unordered(_run_inner_scan_with_progress, outer_tasks)

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
        print(f"\n[ERROR] Parallel execution failed during 2D scan.")
        print(f"Details: {e}")
        return None, None

    end = tm.time()

    return nested_data, end - start


def run_xso_parscan(model_file_name, param_name, param_values, processes=20,
                    param_name2=None, param_values2=None):
    """Executes a 1D or 2D parallel parameter scan for an xsimlab model.

    This is the main user-facing function. It dynamically imports the
    specified model file. If only `param_name` is provided, it runs a 1D
    parallel scan. If `param_name2` is also provided, it runs a 2D scan
    (parallel outer loop, sequential inner loop). It handles setting up
    the multiprocessing pool, executing the tasks, and combining the
    results into a final `xarray.Dataset`.

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

    Returns
    -------
    xarray.Dataset
        The combined `xarray.Dataset` of the scan results.
        Returns `None` if the execution failed.
    """

    if param_name2 is None:
        # --- 1D Scan Logic ---
        iter_scan = _generate_iterable_tasks_1d(param_name, param_values)
        n_points = len(param_values)

        print(f"Starting 1D parallel scan of '{model_file_name}' over {n_points} points using {processes} workers...")

        start = tm.time()

        try:
            with Pool(
                    processes=processes,
                    initializer=_worker_initializer,
                    initargs=(model_file_name,)
            ) as p:
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
            model_file_name, param_name, param_values, param_name2, param_values2, processes
        )

        if nested_data is None:
            return None

        # 3. Combine Results and Report
        out_ds = _unpack_par_scan_2d(nested_data)
        print(f"\n2D Scan complete. Total Time taken: {round(run_time, 5)} seconds.")
        return out_ds