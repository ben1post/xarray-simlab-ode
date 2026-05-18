import time as tm

from xso.model import Model
from xso.solvers import (SolverABC, IVPSolver, StepwiseSolver, FSolver,
                         DerivativeCalculator, NumericalStabilitySolver, HydridStabilitySolver)

_built_in_solvers = {'solve_ivp': IVPSolver, 'fsolve': FSolver, 'deriv': DerivativeCalculator,
                     'stepwise': StepwiseSolver,
                     'stability': NumericalStabilitySolver, 'hybrid_stability': HydridStabilitySolver}


class XSOCore:
    """Backend core class that initializes solver and model, and
    translates between the two for model construction and solving.

    This core class is necessary to allow an abstract model based on differential equations
    to be constructed, without being limited to the strict framework of xsimlab. Currently,
    all xsimlab variables are used as value stores, instead of meaningful components of the model.
    The XSOCore object is shared between all XSO components via a Xsimlab xs.any_object variable.

    Note: The logical division between Core and Solver might be unnecessary, but was chosen to simplify
    switching or modifying either component.
    """

    def __init__(self, solver, solver_kwargs=None):
        """
        Initializes XSO Model and XSO Solver, and stores solve start and end for diagnostics.

        Parameters
        ----------
        solver : {'stepwise', 'solve_ivp', 'fsolve', 'deriv', 'stability', 'hybrid_stability'} \
                 or subclass of SolverABC
           Solver name as str, has to be built into xso.
           Alternatively can be passed a custom subclass of xso.solver.SolverABC.
        solver_kwargs : dict or None, optional
            Extra keyword arguments forwarded to the underlying solver
            call (e.g. ``scipy.integrate.solve_ivp`` for ``'solve_ivp'``,
            ``scipy.optimize.fsolve`` for ``'fsolve'`` / ``'stability'``
            / ``'hybrid_stability'``). Merged on top of each solver's
            class-level ``DEFAULT_SOLVER_KWARGS``; user-supplied keys
            win on collision. Solvers without an underlying scipy call
            (``'stepwise'``, ``'deriv'``) silently ignore the argument.

            When ``solver`` is a pre-built :class:`SolverABC` instance
            and ``solver_kwargs`` is non-empty, the user-supplied kwargs
            are merged on top of the instance's existing
            ``solver_kwargs`` dict (setup-time wins).
        """
        self.solve_start = None
        self.solve_end = None

        solver_kwargs = dict(solver_kwargs) if solver_kwargs else {}

        if isinstance(solver, str):
            try:
                self.solver = _built_in_solvers[solver](solver_kwargs=solver_kwargs)
            except KeyError:
                raise KeyError(
                    "Solver name passed is not built-in. Please choose from: "
                    + ", ".join(repr(k) for k in _built_in_solvers)
                    + "."
                )
        elif isinstance(solver, SolverABC):
            # Setup-time kwargs win over kwargs baked into the instance.
            if solver_kwargs:
                existing = getattr(solver, 'solver_kwargs', None) or {}
                solver.solver_kwargs = {**existing, **solver_kwargs}
            elif not hasattr(solver, 'solver_kwargs'):
                # Backwards-compat for custom SolverABC subclasses that
                # predate solver_kwargs and don't call super().__init__.
                solver.solver_kwargs = {}
            self.solver = solver
        else:
            raise Exception("Solver argument passed to model is not built-in or subclass of SolverABC.")

        self.model = Model()

    def add_variable(self, label, initial_value=0):
        """Adding a variable to the model.

        Function that takes the variable label and initial value as input and
        calls the solver add_variable function. The output is assigned to the
        Model variables dict, from where the variable is returned. This method
        is called within component decorator function, where the output is
        assigned to the appropriate Xsimlab variable.

        Parameters
        ----------
        label : string
            This is the reference string to be used across the model.
        initial_value : numerical, optional
            The initial value of the variable within the model,
            a default value is 0, if none is supplied.
        """
        # the following step registers the variable within the framework
        self.model.variables[label] = self.solver.add_variable(label, initial_value, self.model)
        # return actual value store of variable to xsimlab framework
        return self.model.variables[label]

    def add_parameter(self, label, value):
        """Method to add a parameter with the model backend, via implemented function in solver."""
        self.model.parameters[label] = self.solver.add_parameter(label, value)

    def register_flux(self, label, flux, dims=None):
        """Method to register a flux with the model backend, via implemented function in Solver."""
        if label not in self.model.fluxes:
            # to store flux function:
            self.model.fluxes[label] = flux
            # to store flux value:
            self.model.flux_values[label] = self.solver.register_flux(label, flux, self.model, dims)
        else:
            raise Exception("Something is wrong, a unique flux label was registered twice")

        return self.model.flux_values[label]

    def add_flux(self, process_label, var_label, flux_label, negative=False, list_input=False):
        """Method to add a flux with the model backend, via implemented function in Solver."""
        # to store var - flux connection:
        label = process_label + '_' + flux_label
        flux_var_dict = {'label': label, 'negative': negative, 'list_input': list_input}

        self.model.fluxes_per_var[var_label].append(flux_var_dict)

    def add_forcing(self, label, forcing_func):
        """Method to register add forcing with the model backend, via implemented function in Solver."""
        self.model.forcing_func[label] = forcing_func
        self.model.forcings[label] = self.solver.add_forcing(label, forcing_func, self.model)
        return self.model.forcings[label]

    def assemble(self):
        """Method to assemble model upon full initialization, necessary for some solvers."""
        self.solver.assemble(self.model)
        # start measuring solve time:
        self.solve_start = tm.time()

    def solve(self, time_step):
        """Method to start model solve, calls appropriate function in Solver."""
        if self.model.time is None:
            raise Exception('Time needs to be supplied to Model before solve')
        self.solver.solve(self.model, time_step)

    def cleanup(self):
        """Method to remove temporary files after solving, necessary for some solvers."""
        # stop measuring solver time:
        self.solve_end = tm.time()
        # TODO: diagnostic print here
        # print(f"Model was solved in {round(self.solve_end - self.solve_start, 5)} seconds")
        self.solver.cleanup()
