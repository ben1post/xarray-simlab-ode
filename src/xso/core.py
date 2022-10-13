import time as tm
import numpy as np
import math

from xso.model import Model
from xso.solvers import SolverABC, ODEINTSolver, StepwiseSolver, GEKKOSolver

_built_in_solvers = {'odeint': ODEINTSolver, 'stepwise': StepwiseSolver}

# TODO!: Actually pass along math functions to Solver as needed,
#   how to do this though?
#   - allow passing custom math attribute and supply default
#   - if self.solver.add exists, use it, else: numpy

# TODO: Actually this could all be handled by Model class, right?
#   is there a need for this class actually?????


class XSOCore:
    """Core model object.

    Basic core class that collects solver object and model object and is shared between all model components.
    """

    def __init__(self, solver):
        """
        Parameters
        ----------
        solver : {'stepwise', 'odeint'} or subclass of SolverABC
           Solver name as str, has to be built into xso
           Alternatively can be passed a custom subclass of xso.solver.SolverABC
        """
        self.counter = 0  # this is a hack that is used in

        self.Model = Model()

        self.solve_start = None
        self.solve_end = None

        if isinstance(solver, str):
            self.Solver = _built_in_solvers[solver]()
        elif isinstance(solver, SolverABC):
            self.Solver = solver
        else:
            raise Exception("Solver argument passed to model is not built-in or subclass of SolverABC")

    def add_variable(self, label, initial_value=0):
        """Adding a variable to the model.

        Function that takes the state variable setup as input
        and returns the storage values.

        Parameters
        ----------
        label : string
           this is the reference string to be used across the model
        initial_value : numerical, optional
            the initial value of the variable within the model
            default value is 0, if none is supplied

        Returns
        _______
        xxx : type
            explanation
        """
        # the following step registers the variable within the framework
        self.Model.variables[label] = self.Solver.add_variable(label, initial_value, self.Model)
        # return actual value store of variable to xsimlab framework
        return self.Model.variables[label]

    def add_parameter(self, label, value):
        self.Model.parameters[label] = self.Solver.add_parameter(label, value)

    def register_flux(self, label, flux, dims=None):
        """"""
        if label not in self.Model.fluxes:
            # to store flux function:
            self.Model.fluxes[label] = flux
            # to store flux value:
            self.Model.flux_values[label] = self.Solver.register_flux(label, flux, self.Model, dims)
        else:
            raise Exception("Something is wrong, a unique flux label was registered twice")

        return self.Model.flux_values[label]

    def add_flux(self, process_label, var_label, flux_label, negative=False, list_input=False):
        # to store var - flux connection:
        label = process_label + '_' + flux_label
        flux_var_dict = {'label': label, 'negative': negative, 'list_input': list_input}

        self.Model.fluxes_per_var[var_label].append(flux_var_dict)

    def add_forcing(self, label, forcing_func):
        self.Model.forcing_func[label] = forcing_func
        self.Model.forcings[label] = self.Solver.add_forcing(label, forcing_func, self.Model)
        return self.Model.forcings[label]

    def assemble(self):
        self.Solver.assemble(self.Model)
        self.solve_start = tm.time()

    def solve(self, time_step):
        if self.Model.time is None:
            raise Exception('Time needs to be supplied to Model before solve')
        self.Solver.solve(self.Model, time_step)

    def cleanup(self):
        self.solve_end = tm.time()
        print(f"Model was solved in {round(self.solve_end - self.solve_start, 5)} seconds")
        self.Solver.cleanup()

    # math function wrappers:
    def exp(self, x):
        """ Exponential function that provides correct function for all supported solver types """
        if isinstance(self.Solver, GEKKOSolver):
            return self.Solver.gekko.exp(x)
        else:
            return np.exp(x)

    def sqrt(self, x):
        """ Square root function that provides correct function for all supported solver types"""
        if isinstance(self.Solver, GEKKOSolver):
            return self.Solver.gekko.sqrt(x)
        else:
            return np.sqrt(x)

    def log(self, x):
        if isinstance(self.Solver, GEKKOSolver):
            return self.Solver.gekko.log(x)
        else:
            return np.log(x)

    def product(self, x, axis=None):
        """ Product function that provides correct function for all supported solver types """
        try:
            return np.prod(x, axis=axis)
        except np.VisibleDeprecationWarning:
            return math.prod(x)

    def sum(self, x, axis=None):
        """ Sum function that provides correct function for all supported solver types """
        return np.sum(x, axis=axis)

    def max(self, x1, x2):
        """ """
        if isinstance(self.Solver, GEKKOSolver):
            return self.Solver.gekko.Param(np.maximum(x1, x2))
        else:
            return np.maximum(x1, x2)
