from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import math

from scipy.integrate import odeint


def to_ndarray(value):
    """ helper function to always have at least 1d numpy array returned """
    if isinstance(value, list):
        return np.array(value)
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.array([value])


class SolverABC(ABC):
    """ abstract base class of backend solver class,
    use subclass to solve model within the Phydra framework

    TODO: add all necessary abstract methods and add quick & dirty docs
    """

    @abstractmethod
    def add_variable(self, label, initial_value, model):
        pass

    @abstractmethod
    def add_parameter(self, label, value):
        pass

    @abstractmethod
    def register_flux(self, label, flux, model, dims):
        pass

    @abstractmethod
    def add_forcing(self, label, flux, time):
        pass

    @abstractmethod
    def assemble(self, model):
        pass

    @abstractmethod
    def solve(self, model, time_step):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    class MathFunctionWrappers:
        """Default inner class providing mathematical function wrappers
        using numpy and math.

        This nested class can be modified in any implemented solver,
        if it requires specific math functions.

        Accessible within XSO components using self.m as defined in backendcomps.py class Backend.
         """

        def exp(x):
            """Exponential function"""
            return np.exp(x)

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



class ODEINTSolver(SolverABC):
    """ Solver that can handle odeint solving of Model """

    def __init__(self):
        self.var_init = defaultdict()
        self.flux_init = defaultdict()

    def return_dims_and_array(self, value, model_time):
        """ """
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
        """ this returns storage container """

        if model.time is None:
            raise Exception("To use ODEINT solver, model time needs to be supplied before adding variables")

        # store initial values of variables to pass to odeint function
        self.var_init[label] = to_ndarray(initial_value)

        array_out, dims = self.return_dims_and_array(initial_value, model.time)

        model.var_dims[label] = dims

        return array_out

    def add_parameter(self, label, value):
        """ """
        return to_ndarray(value)

    def register_flux(self, label, flux, model, dims):
        """ this returns storage container """

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
        """ """
        return forcing_func(model.time)

    def assemble(self, model):
        """ """

        for var_key, dim in model.var_dims.items():
            model.full_model_dims[var_key] = dim

        for flx_key, dim in model.flux_dims.items():
            model.full_model_dims[flx_key] = dim

        # TODO: diagnostic print here
        # print model repr for diagnostic purposes:
        #print("Model is assembled:")
        #print(model)

    def solve(self, model, time_step):
        """ """

        full_init = np.concatenate([[v for val in self.var_init.values() for v in val.ravel()],
                                    [v for val in self.flux_init.values() for v in val.ravel()]], axis=None)

        full_model_out = odeint(model.model_function, full_init, model.time)

        state_rows = [row for row in full_model_out.T]

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
        """ """
        pass


class StepwiseSolver(SolverABC):
    """ Solver that can handle stepwise calculation built into xarray-simlab framework """

    def __init__(self):
        self.model_time = 0
        self.time_index = 0

        self.full_model_values = defaultdict()

    def return_dims_and_array(self, value, model_time):
        """ """

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
        """ """
        array_out, _dims = self.return_dims_and_array(initial_value, model.time)

        model.var_dims[label] = _dims

        return array_out

    def add_parameter(self, label, value):
        """ """
        return to_ndarray(value)

    def register_flux(self, label, flux, model, dims):

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
        """ """
        return forcing_func(model.time)

    def assemble(self, model):
        # assemble dimensions now to separate and order fluxes and values correctly for unpacking (computed separately)

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
        #print("Model is assembled:")
        #print(model)

    def solve(self, model, time_step):
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

        state_out = model.model_function(flat_model_state, forcing=model_forcing)

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
        pass
