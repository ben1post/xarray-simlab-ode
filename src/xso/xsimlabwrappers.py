import xsimlab as xs
from xsimlab.variable import VarIntent

from collections import defaultdict

from xso.backendcomps import Backend, RunSolver, Time, create_time_component


def create(components, time_unit='d'):
    """Creates xsimlab Model instance, from dict of XSO components,
    automatically adding the necessary model backend, solver and time components.

    It is a simple wrapper of the xsimlab Model constructor, and returns a fully functional
    Xarray-simlab model object with the XSO core, Solver and Time components added.

    Parameters
    ----------
    components : dict
        Dictionary with component names as keys and classes (decorated with
        :func:`component`) as values.
    time_unit : str, optional
        Unit of time to be used in the model. Default is 'd' for days. This has to be
        supplied at model creation, since the time unit is written to the immutable
        metadata of the model object.

    Returns
    -------
    model : :class:`xsimlab.Model`
        Xarray-simlab model object with the XSO core, Solver and Time components added.
    """

    components.update({'Core': Backend, 'Solver': RunSolver, 'Time': create_time_component(time_unit)})
    return xs.Model(components)


def setup(solver, model, input_vars, output_vars=None, time=None):
    """Create a specific setup for model runs.

    This function wraps xsimlab's create_setup and adds a dummy clock parameter
    necessary for model execution. This convenient function creates a new
    :class:`xarray.Dataset` object with everything needed to run a model
    (i.e., input values, time steps, output variables to save at given times)
    as data variables, coordinates and attributes.

    Parameters
    ----------
    solver : :class:`xso.SolverABC` subclass
        Solver backend to be used at model runtime.
    model : :class:`xsimlab.Model`
        Create a simulation setup for this model.
    input_vars : dict, optional
        Dictionary with values given for model inputs. Entries of the
        dictionary may look like:
        - ``'foo': {'bar': value, ...}`` or
        - ``('foo', 'bar'): value`` or
        - ``'foo__bar': value``
        where ``foo`` is the name of a existing process in the model and
        ``bar`` is the name of an (input) variable declared in that process.
        Values are anything that can be easily converted to
        :class:`xarray.Variable` objects, e.g., single values, array-like,
        ``(dims, data, attrs)`` tuples or xarray objects.
        For array-like values with no dimension labels, xarray-simlab will look
        in ``model`` variables metadata for labels matching the number
        of dimensions of those arrays.
    output_vars : dict, optional
        Dictionary with model variable names to save as simulation output.
        Entries of the dictionary look similar than for ``input_vars``
        (see here above), except that here ``value`` must correspond
        to the dimension of a clock coordinate (i.e., new output values will
        be saved at each time given by the coordinate labels) or
        ``None`` (i.e., only one value will be saved at the end
        of the simulation).
    solver_kwargs : dict, optional
        Additional keyword arguments to pass to the solver backend. This is
        directly passed to the solving function and can be used to adjust parameters
        for solver backends that allow this, such as the IVPSolver backend.

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        A new Dataset object with model inputs as data variables or coordinates
        (depending on their given value) and clock coordinates.
        The names of the input variables also include the name of their process
        (i.e., 'foo__bar').
    """

    if time is None:
        raise Exception("Please supply (numpy) array of explicit timesteps to time keyword argument")

    input_vars.update({'Core__solver_type': solver,
                       'Time__time_input': time})

    # convenient option "ALL" and providing set of values that automatically are returned with dim None:
    if output_vars == "ALL" or output_vars is None:
        full_output_vars = defaultdict()
        for var in model._var_cache.values():
            try:
                if var['metadata']['intent'] is VarIntent.OUT:
                    if var['metadata']['attrs']['Phydra_store_out']:
                        full_output_vars[var['name']] = None
            except:
                pass
        output_vars = full_output_vars
    elif isinstance(output_vars, set):
        output_vars = {var: None for var in output_vars}

    if solver != "stepwise":
        # if a custom solver is used (e.g. odeint) timesteps are handled by that solver
        return xs.create_setup(model=model,
                               # supply a single time step to xsimlab clock
                               clocks={'clock': [time[0], time[1]]},
                               input_vars=input_vars,
                               output_vars=output_vars)
    else:
        # stepwise solver uses defined time as xsimlab clock
        return xs.create_setup(model=model,
                               clocks={'time_input': time},
                               input_vars=input_vars,
                               output_vars=output_vars)


def update_setup(model, old_setup, new_solver, new_time=None):
    """Change existing model setup to another solver type,
    with the possibility to update solver time as well.

    Provides a convenient wrapper for Xarray-simlab's :meth: `update_vars` and `update_clocks`.
    Currently it supports switching between the 'stepwise' solver and 'odeint' adaptive
    step-size solver.

    Parameters
    ----------
    model : :class:`xsimlab.Model`
        The model object that was used to create the model setup.
    old_setup : :class:`xarray.Dataset`
        The previous model setup Dataset, to be updated.
    new_solver : :class:`xso.SolverABC` subclass
        The new solver, that the model setup should be updated to be compatible with.

    Returns
    -------
    new_setup : :class:`xarray.Dataset`
        The new model setup Dataset, that is compatible to be run with the supplied solver.
    """

    if new_time is None:
        time = old_setup.Time__time.values
    else:
        time = new_time

    if new_solver != "stepwise":
        with model:
            setup1 = old_setup.xsimlab.update_vars(input_vars={'Core__solver_type': new_solver,
                                                               'Time__time_input': time})
            new_setup = setup1.xsimlab.update_clocks(clocks={'clock': [time[0], time[1]]}, master_clock='clock')
    else:
        with model:
            setup1 = old_setup.xsimlab.update_vars(input_vars={'Core__solver_type': new_solver,
                                                               'Time__time_input': time})  # ,
            new_setup = setup1.xsimlab.update_clocks(clocks={'clock': time}, master_clock='clock')

    return new_setup
