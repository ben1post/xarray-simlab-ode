import time as tm
import numpy as np
import math

from xso.model import Model
from xso.solvers import SolverABC, ODEINTSolver, StepwiseSolver

_built_in_solvers = {'odeint': ODEINTSolver, 'stepwise': StepwiseSolver}


# TODO: Actually this could all be handled by Model class, right?
#   is there a need for this class actually?????

#TODO
# SO essentially this whole Core class can be replaced by the solver
# but how do I check if the solver is available, where to provide that input?
# s
# so...



class XSOCore:
    """Backend core class
    - initializes solver and model
    - translates between the two for model solving

    """

    def __init__(self, solver):
        """
        Parameters
        ----------
        solver : {'stepwise', 'odeint'} or subclass of SolverABC
           Solver name as str, has to be built into xso
           Alternatively can be passed a custom subclass of xso.solver.SolverABC
        """
        self.solve_start = None
        self.solve_end = None

        if isinstance(solver, str):
            self.solver = _built_in_solvers[solver]()
        elif isinstance(solver, SolverABC):
            self.solver = solver
        else:
            raise Exception("Solver argument passed to model is not built-in or subclass of SolverABC")

        self.model = Model()

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
        self.model.variables[label] = self.solver.add_variable(label, initial_value, self.model)
        # return actual value store of variable to xsimlab framework
        return self.model.variables[label]

    def add_parameter(self, label, value):
        self.model.parameters[label] = self.solver.add_parameter(label, value)

    def register_flux(self, label, flux, dims=None):
        """"""
        if label not in self.model.fluxes:
            # to store flux function:
            self.model.fluxes[label] = flux
            # to store flux value:
            self.model.flux_values[label] = self.solver.register_flux(label, flux, self.model, dims)
        else:
            raise Exception("Something is wrong, a unique flux label was registered twice")

        return self.model.flux_values[label]

    def add_flux(self, process_label, var_label, flux_label, negative=False, list_input=False):
        """"""
        # to store var - flux connection:
        label = process_label + '_' + flux_label
        flux_var_dict = {'label': label, 'negative': negative, 'list_input': list_input}

        self.model.fluxes_per_var[var_label].append(flux_var_dict)

    def add_forcing(self, label, forcing_func):
        """"""
        self.model.forcing_func[label] = forcing_func
        self.model.forcings[label] = self.solver.add_forcing(label, forcing_func, self)
        return self.model.forcings[label]

    def assemble(self):
        """"""
        self.solver.assemble(self.model)
        self.solve_start = tm.time()

    def solve(self, time_step):
        """"""
        if self.model.time is None:
            raise Exception('Time needs to be supplied to Model before solve')
        self.solver.solve(self.model, time_step)

    def cleanup(self):
        """"""
        self.solve_end = tm.time()
        # TODO: diagnostic print here
        #print(f"Model was solved in {round(self.solve_end - self.solve_start, 5)} seconds")
        self.solver.cleanup()

