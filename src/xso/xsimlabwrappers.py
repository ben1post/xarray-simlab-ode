import json

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


def setup(solver, model, input_vars, output_vars=None, time=None,
          solver_kwargs=None):
    """Create a specific setup for model runs.

    This function wraps xsimlab's create_setup and adds a dummy clock parameter
    necessary for model execution. This convenient function creates a new
    :class:`xarray.Dataset` object with everything needed to run a model
    (i.e., input values, time steps, output variables to save at given times)
    as data variables, coordinates and attributes.

    Parameters
    ----------
    solver : str or :class:`xso.SolverABC` subclass
        Solver backend to be used at model runtime. Built-in names:
        ``'solve_ivp'``, ``'stepwise'``, ``'fsolve'``, ``'deriv'``,
        ``'stability'``, ``'hybrid_stability'``.
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
    output_vars : dict, set, str, or None, optional
        Selection of model variables to save as simulation output.

        - ``None`` (default) or ``"ALL"``: record every variable
          internally tagged for storage (state variables, fluxes,
          forcings, and ``setup_func`` parameters).
        - A ``set`` of variable names: record only the named variables,
          each at every clock tick.
        - A ``dict``: the xsimlab-native form, mapping variable name
          to a clock coordinate (or ``None`` for end-of-run only).
    time : numpy.ndarray
        Sequence of time steps at which to evaluate the model. **Required.**
    solver_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the underlying solver call
        (e.g. ``scipy.integrate.solve_ivp`` for ``'solve_ivp'``,
        ``scipy.optimize.fsolve`` for ``'fsolve'`` / ``'stability'`` /
        ``'hybrid_stability'``). Merged on top of each solver's
        class-level ``DEFAULT_SOLVER_KWARGS``; user-supplied keys win on
        collision. Solvers without an underlying scipy call
        (``'stepwise'``, ``'deriv'``) silently ignore the argument.

        Typical uses: switch to a stiff-system solver via
        ``solver_kwargs={'method': 'LSODA'}`` (also ``'BDF'``,
        ``'Radau'``); loosen or tighten tolerances via
        ``solver_kwargs={'rtol': 1e-3, 'atol': 1e-12}``; bound the
        adaptive step via ``solver_kwargs={'max_step': 0.1}``.

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

    # solver_kwargs is JSON-encoded to a string before being injected into
    # the xsimlab input dataset. xsimlab persists every input variable to
    # zarr at run time (see stores.write_input_xr_dataset), and zarr cannot
    # serialize arbitrary Python dicts; a string survives the round-trip
    # trivially. Backend.initialize calls json.loads to recover the dict.
    input_vars.update({'Core__solver_type': solver,
                       'Core__solver_kwargs': json.dumps(solver_kwargs or {}),
                       'Time__time_input': time})

    # convenient option "ALL" and providing set of values that automatically are returned with dim None:
    if output_vars == "ALL" or output_vars is None:
        full_output_vars = defaultdict()
        for var in model._var_cache.values():
            try:
                if var['metadata']['intent'] is VarIntent.OUT:
                    if var['metadata']['attrs']['xso_store_out']:
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


def update_setup(model, old_setup, new_solver, new_time=None,
                 new_solver_kwargs=None):
    """Change existing model setup to another solver type,
    with the possibility to update solver time as well.

    Provides a convenient wrapper for Xarray-simlab's :meth: `update_vars` and `update_clocks`.
    Currently, it supports switching between the 'stepwise' solver and 'odeint' adaptive
    step-size solver.

    Parameters
    ----------
    model : :class:`xsimlab.Model`
        The model object that was used to create the model setup.
    old_setup : :class:`xarray.Dataset`
        The previous model setup Dataset, to be updated.
    new_solver : str or :class:`xso.SolverABC` subclass
        The new solver, that the model setup should be updated to be compatible with.
    new_time : numpy.ndarray or None, optional
        New time array. Defaults to keeping the existing time array.
    new_solver_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the new solver's underlying
        call (see :func:`setup`). When ``None`` (default), the
        ``Core__solver_kwargs`` slot is **cleared** so the new solver
        falls back to its class-level ``DEFAULT_SOLVER_KWARGS``. This
        avoids the footgun where solver-specific kwargs from the
        previous setup (e.g. ``{'method': 'LSODA'}`` for ``solve_ivp``)
        leak into a different solver family (e.g. ``fsolve``) that does
        not accept them. To carry kwargs across an update, extract them
        from ``old_setup`` and pass them explicitly. Passing ``{}`` is
        equivalent to the default and is the explicit way to write
        "use defaults".

    Returns
    -------
    new_setup : :class:`xarray.Dataset`
        The new model setup Dataset, that is compatible to be run with the supplied solver.
    """

    if new_time is None:
        time = old_setup.Time__time.values
    else:
        time = new_time

    # Clear solver_kwargs by default (new_solver_kwargs=None -> {}) so
    # solver-specific kwargs from the previous solver cannot silently
    # reach a new solver family that does not accept them. See note in
    # setup(): the slot travels as a JSON-encoded string for zarr
    # serializability.
    new_kwargs = new_solver_kwargs if new_solver_kwargs is not None else {}
    input_vars = {
        'Core__solver_type': new_solver,
        'Core__solver_kwargs': json.dumps(new_kwargs),
        'Time__time_input': time,
    }

    if new_solver != "stepwise":
        with model:
            setup1 = old_setup.xsimlab.update_vars(input_vars=input_vars)
            new_setup = setup1.xsimlab.update_clocks(clocks={'clock': [time[0], time[1]]}, master_clock='clock')
    else:
        with model:
            setup1 = old_setup.xsimlab.update_vars(input_vars=input_vars)
            new_setup = setup1.xsimlab.update_clocks(clocks={'clock': time}, master_clock='clock')

    return new_setup

