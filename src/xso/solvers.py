from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import math

from scipy.integrate import solve_ivp


def to_ndarray(value):
    """Helper function to always have at least 1d numpy array returned."""
    if isinstance(value, list):
        return np.array(value)
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.array([value])


class SolverABC(ABC):
    """Abstract base class of backend solver class,
    use subclass to solve model within the XSO framework.
    """

    @abstractmethod
    def add_variable(self, label, initial_value, model):
        """Method to reformat a variable object for use with solver,
        should return storage object of value."""
        pass

    @abstractmethod
    def add_parameter(self, label, value):
        """Method to reformat a parameter object for use with solver."""
        pass

    @abstractmethod
    def register_flux(self, label, flux, model, dims):
        """Method to reformat a flux function for use with solver,
        should return storage object of value."""
        pass

    @abstractmethod
    def add_forcing(self, label, flux, time):
        """Method to reformat a forcing function for use with solver,
        should return storage object of value."""
        pass

    @abstractmethod
    def assemble(self, model):
        """Method to initialize model."""
        pass

    @abstractmethod
    def solve(self, model, time_step):
        """Method to solve model, specific to solver chosen."""
        pass

    @abstractmethod
    def cleanup(self):
        """Method to clean up temporary storage or perform other actions
        after the model has been run."""
        pass

    class MathFunctionWrappers:
        """Default inner class providing mathematical function wrappers
        using numpy and math.

        This nested class can be modified in any implemented solver,
        if it requires specific math functions.

        Accessible within XSO components using self.m as defined in backendcomps.py class Backend.
         """

        # Constants:
        pi = np.pi  # pi constant
        e = np.e  # e constant

        def exp(x):
            """Exponential function"""
            return np.exp(x)

        # add np.errstate to ignore superfluous warnings, caused by solve_ivp solver
        @np.errstate(all='ignore')
        def sqrt(x):
            """Square root function"""
            return np.sqrt(x)

        def log(x):
            """Logarithmic function """
            return np.log(x)

        def product(x):  # no axis?
            """Product function"""
            return math.prod(x)

        def sum(x, axis=None):
            """ Sum function"""
            return np.sum(x, axis=axis)

        def min(x1, x2):
            """ Minimum function """
            return np.minimum(x1, x2)

        def max(x1, x2):
            """ Maximum function """
            return np.maximum(x1, x2)

        def abs(x):
            """ Absolute value function """
            return np.abs(x)

        def sin(x):
            """ Sine function """
            return np.sin(x)


class IVPSolver(SolverABC):
    """Solver backend using scipy.integrate.solve_ivp to solve model.

    SOLVE_IVP is a variable step-size solver for ordinary differential equations,
    included in the SciPy Python package.

    By default, it utilizes an explicit Runge-Kutta method of order 5(4).
    """

    def __init__(self):
        self.var_init = defaultdict()
        self.flux_init = defaultdict()

    @staticmethod
    def return_dims_and_array(value, model_time):
        """Helper function to expand numpy array to appropriate size
        for odeint solver based on value and model time.
        """
        if np.size(value) == 1:
            _dims = None
            full_dims = (np.size(model_time),)
        elif len(np.shape(value)) == 1:
            _dims = np.size(value)
            full_dims = (_dims, np.size(model_time))
        else:
            _dims = np.shape(value)
            full_dims = (*_dims, np.size(model_time))

        array_out = np.zeros(full_dims)
        return array_out, _dims

    def add_variable(self, label, initial_value, model):
        """Reformats variable to comply with solver and return storage array."""

        if model.time is None:
            raise Exception("To use ODEINT solver, model time needs to be supplied before adding variables")

        # store initial values of variables to pass to odeint function
        self.var_init[label] = to_ndarray(initial_value)

        array_out, dims = self.return_dims_and_array(initial_value, model.time)

        model.var_dims[label] = dims

        return array_out

    def add_parameter(self, label, value):
        """Returns parameter as numpy array."""
        return to_ndarray(value)

    def register_flux(self, label, flux, model, dims):
        """Method to reformat flux function with appropriate inputs and to proper size."""

        if model.time is None:
            raise Exception("To use ODEINT solver, model time needs to be supplied before adding fluxes")

        var_in_dict = defaultdict()
        for var, value in model.variables.items():
            var_in_dict[var] = self.var_init[var]
        for var, value in self.flux_init.items():
            var_in_dict[var] = value

        forcing_init = defaultdict()
        for key, func in model.forcing_func.items():
            forcing_init[key] = func(0)

        _flux_value = to_ndarray(flux(state=var_in_dict,
                                      parameters=model.parameters,
                                      forcings=forcing_init))
        self.flux_init[label] = _flux_value

        array_out, dims = self.return_dims_and_array(_flux_value, model.time)

        model.flux_dims[label] = dims

        return array_out

    def add_forcing(self, label, forcing_func, model):
        """Compute forcing for model time."""
        return forcing_func(model.time)

    def assemble(self, model):
        """Define full model dimensions after initialization."""

        for var_key, dim in model.var_dims.items():
            model.full_model_dims[var_key] = dim

        for flx_key, dim in model.flux_dims.items():
            model.full_model_dims[flx_key] = dim

        # TODO: diagnostic print here
        # print model repr for diagnostic purposes:
        # print("Model is assembled!")
        # print(model)

    def solve(self, model, time_step):
        """Solve model using scipy.integrate.solve_ivp, passing model_function, initial values and model.time.
        The model output is then assigned to the previously initialized storage arrays within xsimlab backend.
        """
        # convert all initial values to a 1D array:
        full_init = np.concatenate([[v for val in self.var_init.values() for v in val.ravel()],
                                    [v for val in self.flux_init.values() for v in val.ravel()]], axis=None)

        # solving model here:
        full_model_out = solve_ivp(model.model_function,
                                   t_span=[model.time[0], model.time[-1]],
                                   y0=full_init,
                                   t_eval=model.time)

        # round off 1e150-th decimal to remove floating point numerical errors
        state_rows = [row for row in np.around(full_model_out.y, decimals=150)]

        # unpack and reshape state array to appropriate dimensions:
        state_dict = defaultdict()
        index = 0
        for key, dims in model.full_model_dims.items():
            if dims is None:
                state_dict[key] = state_rows[index]
                index += 1
            elif isinstance(dims, int):
                val_list = []
                for i in range(dims):
                    val_list.append(state_rows[index])
                    index += 1
                state_dict[key] = np.array(val_list)
            else:
                full_dims = (*dims, np.size(model.time))
                _length = int(np.prod(dims))
                val_list = []
                for i in range(_length):
                    val_list.append(state_rows[index])
                    index += 1

                state_dict[key] = np.array(val_list).reshape(full_dims)

        # assign solved model state to value storage in xsimlab framework:
        for var_key, val in model.variables.items():
            val[...] = state_dict[var_key]

        for flux_key, val in model.flux_values.items():
            state = state_dict[flux_key]
            dims = model.full_model_dims[flux_key]

            difference = np.diff(state) / time_step

            if dims:
                if isinstance(dims, tuple):
                    val[...] = np.concatenate((difference[..., 0][..., np.newaxis], difference), axis=len(dims))
                else:
                    val[...] = np.hstack((difference[..., 0][:, None], difference))
            else:
                val[...] = np.hstack((difference[0], difference))

    def cleanup(self):
        """Empty cleanup method, not necessary for this solver."""
        pass


class StepwiseSolver(SolverABC):
    """Solver that can handle stepwise calculation built into xsimlab framework.

    Model output is computed step by step and assigned to the appropriate
    storage arrays in xsimlab backend."""

    def __init__(self):
        self.model_time = 0
        self.time_index = 0

        self.full_model_values = defaultdict()

    @staticmethod
    def return_dims_and_array(value, model_time):
        """Helper function to create arrays of appropriate size,
        and assign initial value(s) to first index
        """

        if np.size(value) == 1:
            _dims = None
            full_dims = (np.size(model_time),)
            array_out = np.zeros(full_dims)
            array_out[0] = value
        elif len(np.shape(value)) == 1:
            _dims = np.size(value)
            full_dims = (_dims, np.size(model_time))
            array_out = np.zeros(full_dims)
            array_out[:, 0] = value
        else:
            _dims = np.shape(value)
            full_dims = (*_dims, np.size(model_time))
            array_out = np.zeros(full_dims)
            array_out[..., 0] = value

        return array_out, _dims

    def add_variable(self, label, initial_value, model):
        """Method to reformat variable and return storage array."""
        array_out, _dims = self.return_dims_and_array(initial_value, model.time)
        model.var_dims[label] = _dims
        return array_out

    def add_parameter(self, label, value):
        """Method to reformat parameter and return array."""
        return to_ndarray(value)

    def register_flux(self, label, flux, model, dims):
        """Method to reformat flux function with appropriate inputs and to proper size."""

        var_in_dict = defaultdict()
        for var_key, value in model.variables.items():
            _dims = model.var_dims[var_key]
            if _dims is None:
                var_in_dict[var_key] = value[0]
            elif isinstance(_dims, int):
                var_in_dict[var_key] = value[:, 0]
            else:
                var_in_dict[var_key] = value[..., 0]

        for flx_key, value in model.flux_values.items():
            _dims = model.flux_dims[flx_key]
            if _dims is None:
                var_in_dict[flx_key] = value[0]
            elif isinstance(_dims, int):
                var_in_dict[flx_key] = value[:, 0]
            else:
                var_in_dict[flx_key] = value[..., 0]

        forcing_now = defaultdict()
        for key, func in model.forcing_func.items():
            forcing_now[key] = func(0)

        flux_init = to_ndarray(flux(state=var_in_dict,
                                    parameters=model.parameters,
                                    forcings=forcing_now))

        array_out, _dims = self.return_dims_and_array(flux_init, model.time)

        model.flux_dims[label] = _dims

        return array_out

    def add_forcing(self, label, forcing_func, model):
        """Compute forcing over model time and provide as array."""
        return forcing_func(model.time)

    def assemble(self, model):
        """Assemble full model dimension and order fluxes for proper unpacking."""
        for var_key, value in model.variables.items():
            _dims = model.var_dims[var_key]
            model.full_model_dims[var_key] = _dims
            self.full_model_values[var_key] = value

        for flx_key, value in model.flux_values.items():
            _dims = model.flux_dims[flx_key]
            model.full_model_dims[flx_key] = _dims
            self.full_model_values[flx_key] = value

        # TODO: diagnostic print here
        # finally print model repr for diagnostic purposes:
        # print("Model is assembled:")
        # print(model)

    def solve(self, model, time_step):
        """Solve model in a stepwise fashion, calling this function at each time step."""

        self.model_time += time_step
        self.time_index += 1

        model_forcing = defaultdict()
        for key, func in model.forcing_func.items():
            # retrieve pre-computed forcing:
            model_forcing[key] = model.forcings[key][self.time_index]

        model_state = []
        for key, val in self.full_model_values.items():
            if model.full_model_dims[key]:
                model_state.append(val[..., self.time_index - 1])
            else:
                model_state.append(val[self.time_index - 1])

        # flatten list for model function:
        flat_model_state = np.concatenate(model_state, axis=None)

        state_out = model.model_function(current_state=flat_model_state, forcing=model_forcing)

        # unpack flat state:
        state_dict = model.unpack_flat_state(state_out)

        for key, val in model.variables.items():
            if model.full_model_dims[key]:
                val[..., self.time_index] = val[..., self.time_index - 1] + state_dict[key] * time_step
            else:
                val[self.time_index] = val[self.time_index - 1] + state_dict[key] * time_step

        for key, val in model.flux_values.items():
            if model.full_model_dims[key]:
                val[..., self.time_index] = state_dict[key]
            else:
                val[self.time_index] = state_dict[key]

    def cleanup(self):
        """Empty cleanup method, not necessary for this solver."""
        pass
