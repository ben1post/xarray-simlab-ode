from collections import defaultdict
import numpy as np

# from itertools import zip_longest as zip


def return_dim_ndarray(value):
    """ helper function to always have at least 1d numpy array returned """
    if isinstance(value, list):
        return np.array(value)
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.array([value])


class PhydraModel:
    """Backend model class
    - collects all things relevant to the model instance (i.e. variables, parameters, ...)
    - can be solved by passing it to the SolverABC class (that's where conversion (if necessary) happens)
    """

    def __init__(self):

        self.time = None

        self.variables = defaultdict()
        self.parameters = defaultdict()

        self.forcing_func = defaultdict()
        self.forcings = defaultdict()

        self.fluxes = defaultdict()
        self.flux_values = defaultdict()
        self.fluxes_per_var = defaultdict(list)

        self.var_dims = defaultdict()
        self.flux_dims = defaultdict()
        self.full_model_dims = defaultdict()

    def __repr__(self):
        return (f"Model contains: \n"
                f"Variables:{[var for var in self.variables]} \n"
                f"Parameters:{[par for par in self.parameters]} \n"
                f"Forcings:{[forc for forc in self.forcings]} \n"
                f"Fluxes:{[flx for flx in self.fluxes]} \n"
                f"Full Model Dimensions:{[(state,dim) for state,dim in self.full_model_dims.items()]} \n")

    def unpack_flat_state(self, flat_state):
        """ """

        state_dict = defaultdict()
        index = 0
        for key, dims in self.full_model_dims.items():
            if dims is None:
                state_dict[key] = flat_state[index]
                index += 1
            elif isinstance(dims, int):
                state_dict[key] = flat_state[index:index+dims]
                index += dims
            else:
                _length = np.prod(dims)
                state_dict[key] = flat_state[index:index+_length].reshape(dims)  # np.array( )
                index += _length

        return state_dict

    def model_function(self, current_state, time=None, forcing=None):
        """ general model function that matches fluxes to state variables

        :param current_state:
        :param time: argument is necessary for odeint solve
        :param forcing:
        :return:
        """

        state = self.unpack_flat_state(current_state)

        # Return forcings for time point:
        if time is not None:
            forcing_now = defaultdict()
            for key, func in self.forcing_func.items():
                forcing_now[key] = func(time)
            forcing = forcing_now
        elif forcing is None:
            forcing = self.forcings

        # Compute fluxes:
        flux_values = defaultdict()
        fluxes_out = []
        for flx_label, flux in self.fluxes.items():
            _value = return_dim_ndarray(flux(state=state, parameters=self.parameters, forcings=forcing))
            flux_values[flx_label] = _value
            fluxes_out.append(_value)
            if flx_label in state:
                state.update({flx_label: _value})

        # Route list input fluxes:
        list_input_fluxes = defaultdict(list)
        for flux_var_dict in self.fluxes_per_var["list_input"]:
            flux_label, negative, list_input = flux_var_dict.values()

            flux_val = flux_values[flux_label]
            flux_dims = self.full_model_dims[flux_label]

            list_var_dims = []
            for var in list_input:
                _dim = self.full_model_dims[var]
                list_var_dims.append(_dim or 1)

            if len(list_input) == flux_dims:
                for var, flux in zip(list_input, flux_val):
                    if negative:
                        list_input_fluxes[var].append(-flux)
                    else:
                        list_input_fluxes[var].append(flux)
            elif sum(list_var_dims) == flux_dims:
                _dim_counter = 0
                for var, dims in zip(list_input, list_var_dims):
                    flux = flux_val[_dim_counter:_dim_counter+dims]
                    _dim_counter += dims
                    if negative:
                        list_input_fluxes[var].append(-flux)
                    else:
                        list_input_fluxes[var].append(flux)
            else:
                raise Exception("ERROR: list input vars dims and flux output dims do not match")

        # Assign fluxes to variables:
        state_out = []
        for var_label, value in self.variables.items():
            var_fluxes = []
            dims = self.full_model_dims[var_label]
            flux_applied = False
            if var_label in self.fluxes_per_var:
                flux_applied = True
                for flux_var_dict in self.fluxes_per_var[var_label]:
                    flux_label, negative, list_input = flux_var_dict.values()

                    if dims:
                        _flux = flux_values[flux_label]
                    else:
                        _flux = np.sum(flux_values[flux_label])

                    if negative:
                        var_fluxes.append(-_flux)
                    else:
                        var_fluxes.append(_flux)

            if var_label in list_input_fluxes:
                flux_applied = True
                for flux in list_input_fluxes[var_label]:
                    if dims:
                        _flux = flux
                    else:
                        _flux = np.sum(flux)
                    var_fluxes.append(_flux)

            if not flux_applied:
                dims = self.full_model_dims[var_label]
                if dims:
                    var_fluxes.append(np.array([0 for i in range(dims)]))
                else:
                    var_fluxes.append(0)

            state_out.append(np.sum(var_fluxes, axis=0))

        full_output = np.concatenate([[v for val in state_out for v in val.ravel()],
                                      [v for val in fluxes_out for v in val.ravel()]], axis=None)

        return full_output
