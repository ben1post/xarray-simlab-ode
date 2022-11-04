import xsimlab as xs

import attr
from attr import fields_dict

from collections import OrderedDict, defaultdict
from functools import wraps
import inspect
import numpy as np

from .variables import XSOVarType
from xso.backendcomps import FirstInit, SecondInit, ThirdInit, FourthInit, FifthInit

# TODO:
#   - add proper documentation to the functions below, this is key

def _create_variables_dict(process_cls):
    """Get all phydra variables declared in a component.
    Exclude attr.Attribute objects that are not xsimlab-specific.
    """
    return OrderedDict(
        (k, v) for k, v in fields_dict(process_cls).items() if "var_type" in v.metadata
    )


def _convert_2_xsimlabvar(var, intent='in',
                          var_dims=None, value_store=False, groups=None,
                          description_label='', attrs=True):
    """Converts XSO variables to Xarray-simlab variables to be used in the model backend

    Function receives variable as attr in _make_phydra_variable function and extracts
    description, dimensions and metadata, then passes it and additional arguments
    through Xarray-simlab's xs.variable function.

    Parameters
    ----------
    var : xxx
        XSO variable object defined in object decorated with xso.component()
    intent: str ('in' or 'out')
        passed along, defines variable as receiving input from another component
        or being initialized within this component

    Returns
    -------
    xs.variable
        attr class handled by Xarray-simlab, the functional foundation of XSO
    """
    var_description = var.metadata.get('description')
    if var_description:
        description_label = description_label + var_description

    if var_dims is None:
        var_dims = var.metadata.get('dims')

    #
    if value_store:
        if not var_dims:
            # if there is no dim supplied, we need 'time' as
            var_dims = 'time'
        elif 'time' in var_dims:
            pass
        elif isinstance(var_dims, str):
            var_dims = (var_dims, 'time')
        elif isinstance(var_dims, tuple):
            var_dims = var_dims + ('time',)
        elif isinstance(var_dims, list):
            _dims = []
            for dim in var_dims:
                if isinstance(dim, str):
                    _dims.append((dim, 'time'))
                else:
                    _dims.append((*dim, 'time'))
            var_dims = _dims
        else:
            raise ValueError("Failed to parse dims argument for variable of type:",
                             var.metadata["var_type"], "with description:", description_label,
                             "with dimensions:", var_dims)

    if var_dims is None:
        var_dims = ()

    if attrs:
        var_attrs = var.metadata.get('attrs')
    else:
        var_attrs = {}

    return xs.variable(intent=intent, dims=var_dims, groups=groups, description=description_label, attrs=var_attrs)


def _make_xso_variable(label, variable):
    """ """
    xs_var_dict = defaultdict()
    if variable.metadata.get('foreign') is True:
        list_input = variable.metadata.get('list_input')
        if list_input:
            var_dims = variable.metadata.get('dims')
            if var_dims is None:
                raise ValueError("Variable with list_input=True requires passing dimension to dims keyword argument")
            xs_var_dict[label] = _convert_2_xsimlabvar(var=variable, var_dims=var_dims,
                                                       description_label='label reference / ')
        else:
            xs_var_dict[label] = _convert_2_xsimlabvar(var=variable, var_dims=(),
                                                       description_label='label reference / ')
    elif variable.metadata.get('foreign') is False:
        xs_var_dict[label + '_label'] = _convert_2_xsimlabvar(var=variable, var_dims=(),
                                                              description_label='label / ')
        xs_var_dict[label + '_init'] = _convert_2_xsimlabvar(var=variable, description_label='initial value / ')
        xs_var_dict[label + '_value'] = _convert_2_xsimlabvar(var=variable, intent='out',
                                                              value_store=True,
                                                              description_label='output of variable value / ')
    return xs_var_dict


def _make_xso_parameter(label, variable):
    """ """
    xs_var_dict = defaultdict()
    xs_var_dict[label] = _convert_2_xsimlabvar(var=variable)
    return xs_var_dict


def _make_xso_forcing(label, variable):
    """ """
    xs_var_dict = defaultdict()
    if variable.metadata.get('foreign') is True:
        xs_var_dict[label] = _convert_2_xsimlabvar(var=variable, description_label='label reference / ')
    elif variable.metadata.get('foreign') is False:
        xs_var_dict[label + '_label'] = _convert_2_xsimlabvar(var=variable, description_label='label / ')
        xs_var_dict[label + '_value'] = _convert_2_xsimlabvar(var=variable, intent='out',
                                                              value_store=True,
                                                              description_label='output of forcing value / ')
    return xs_var_dict


def _make_xso_flux(label, variable):
    """ """
    xs_var_dict = defaultdict()
    xs_var_dict[label + '_value'] = _convert_2_xsimlabvar(var=variable, intent='out',
                                                          value_store=True,
                                                          description_label='output of flux value / ')
    group = variable.metadata.get('group')
    group_to_arg = variable.metadata.get('group_to_arg')

    if group and group_to_arg:
        raise Exception("A flux can be either added to group or take a group as argument, not both.")

    if group:
        xs_var_dict[label + '_label'] = _convert_2_xsimlabvar(var=variable, intent='out', groups=group, var_dims=(),
                                                              description_label='label reference with group / ',
                                                              attrs=False)
    if group_to_arg:
        xs_var_dict[group_to_arg] = xs.group(group_to_arg)

    return xs_var_dict


_make_xsimlab_vars = {
    XSOVarType.VARIABLE: _make_xso_variable,
    XSOVarType.FORCING: _make_xso_forcing,
    XSOVarType.PARAMETER: _make_xso_parameter,
    XSOVarType.FLUX: _make_xso_flux,
}


def _create_xsimlab_var_dict(cls_vars):
    """ """
    xs_var_dict = defaultdict()

    for key, var in cls_vars.items():
        var_type = var.metadata.get('var_type')
        var_dict = _make_xsimlab_vars[var_type](key, var)
        for xs_key, xs_var in var_dict.items():
            xs_var_dict[xs_key] = xs_var

    return xs_var_dict


def _create_forcing_dict(cls, var_dict):
    """ """
    forcings_dict = defaultdict()

    for key, var in var_dict.items():
        if var.metadata.get('var_type') is XSOVarType.FORCING:
            _forcing_setup_func = var.metadata.get('setup_func')
            if _forcing_setup_func is not None:
                forcings_dict[key] = getattr(cls, _forcing_setup_func)

    return forcings_dict


def _create_new_cls(cls, cls_dict, init_stage):
    """ """
    # TODO! this should either make use of native ordering algorithm
    #   or order processes based on supplied xso variables
    #   for now, not sure how to do, so will leave it as such
    if init_stage == 1:
        new_cls = type(cls.__name__, (FirstInit,), cls_dict)
    elif init_stage == 2:
        new_cls = type(cls.__name__, (SecondInit,), cls_dict)
    elif init_stage == 3:
        new_cls = type(cls.__name__, (ThirdInit,), cls_dict)
    elif init_stage == 4:
        new_cls = type(cls.__name__, (FourthInit,), cls_dict)
    elif init_stage == 5:
        new_cls = type(cls.__name__, (FifthInit,), cls_dict)
    else:
        raise Exception("Wrong init_stage supplied, needs to be 1, 2, 3, 4 or 5")
    return new_cls


def _initialize_process_vars(cls, vars_dict):
    """ """
    process_label = cls.label
    for key, var in vars_dict.items():
        var_type = var.metadata.get('var_type')

        if var_type is XSOVarType.VARIABLE:
            foreign = var.metadata.get('foreign')
            if foreign is True:
                _label = getattr(cls, key)
            elif foreign is False:
                _init = getattr(cls, key + '_init')
                _label = getattr(cls, key + '_label')
                setattr(cls, key + '_value', cls.core.add_variable(label=_label, initial_value=_init))

            flux_label = var.metadata.get('flux')
            # TODO: use dict mapping for flux + negative statements, instead of two separate args and lists
            flux_negative = var.metadata.get('negative')
            list_input = var.metadata.get('list_input')

            if flux_label:
                if isinstance(flux_label, list) and isinstance(flux_negative, list):
                    for _flx_label, _flx_negative in zip(flux_label, flux_negative):
                        if list_input:
                            cls.core.add_flux(process_label=cls.label, var_label="list_input", flux_label=_flx_label,
                                              negative=_flx_negative, list_input=_label)
                        else:
                            cls.core.add_flux(process_label=cls.label, var_label=_label, flux_label=_flx_label,
                                              negative=_flx_negative)
                elif isinstance(flux_label, list) or isinstance(flux_negative, list):
                    raise ValueError(f"Variable {_label} was assigned {flux_label} with negative arguments {flux_negative}, "
                                     f"both need to be supplied as list")
                else:
                    if list_input:
                        cls.core.add_flux(process_label=cls.label, var_label="list_input", flux_label=flux_label,
                                          negative=flux_negative, list_input=_label)
                    else:
                        cls.core.add_flux(process_label=cls.label, var_label=_label, flux_label=flux_label,
                                          negative=flux_negative)
        elif var_type is XSOVarType.PARAMETER:
            if var.metadata.get('foreign') is False:
                _par_value = getattr(cls, key)
                cls.core.add_parameter(label=process_label + '_' + key, value=_par_value)
            else:
                raise Exception("Currently Phydra does not support foreign=True for parameters -> TODO 4 v1")


def _create_flux_inputargs_dict(cls, vars_dict):
    """ """
    input_arg_dict = defaultdict(list)
    _check_duplicate_group_arg = []

    for key, var in vars_dict.items():
        var_type = var.metadata.get('var_type')
        if var_type is XSOVarType.VARIABLE:
            var_dim = var.metadata.get('dims')
            if var.metadata.get('foreign') is False:
                var_label = getattr(cls, key + '_label')
                input_arg_dict['vars'].append({'var': key, 'label': var_label, 'dim': var_dim})
            elif var.metadata.get('foreign') is True:
                var_label = getattr(cls, key)
                if var.metadata.get('list_input'):
                    var_label = np.array(var_label)  # force to array for easier type checking later
                    input_arg_dict['list_input_vars'].append({'var': key, 'label': var_label, 'dim': var_dim})
                else:
                    input_arg_dict['vars'].append({'var': key, 'label': var_label, 'dim': var_dim})

        elif var_type is XSOVarType.PARAMETER:
            par_dim = var.metadata.get('dims')
            # TODO: so it doesn't work with foreign parameters here yet!
            input_arg_dict['pars'].append({'var': key, 'label': cls.label + '_' + key, 'dim': par_dim})

        elif var_type is XSOVarType.FORCING:
            if var.metadata.get('foreign') is False:
                forc_label = getattr(cls, key + '_label')
            elif var.metadata.get('foreign') is True:
                forc_label = getattr(cls, key)
            else:
                raise ValueError("Wrong argument supplied to xso.foreign, can be True or False")
            input_arg_dict['forcs'].append({'var': key, 'label': forc_label})

        elif var_type is XSOVarType.FLUX:
            flx_dim = var.metadata.get('dims')
            group_to_arg = var.metadata.get('group_to_arg')
            if group_to_arg:
                if group_to_arg not in _check_duplicate_group_arg:
                    _check_duplicate_group_arg.append(group_to_arg)
                    group = list(getattr(cls, group_to_arg))  # convert generator to list for safer handling
                    input_arg_dict['group_args'].append({'var': group_to_arg, 'label': group, 'dim': flx_dim})

    return input_arg_dict


def _initialize_fluxes(cls, vars_dict):
    """ """
    for key, var in vars_dict.items():
        var_type = var.metadata.get('var_type')
        if var_type is XSOVarType.FLUX:
            flux_func = var.metadata.get('flux_func')
            flux_dim = var.metadata.get('dims')
            label = cls.label + '_' + flux_func.__name__

            if var.metadata.get('group'):
                setattr(cls, flux_func.__name__ + '_label', label)

            setattr(cls, key + '_value',
                    cls.core.register_flux(label=label, flux=cls.flux_decorator(flux_func), dims=flux_dim))


def _initialize_forcings(cls, forcing_dict):
    """ """
    for var, forc_input_func in forcing_dict.items():
        forc_label = getattr(cls, var + '_label')

        argspec = inspect.getfullargspec(forc_input_func)

        input_args = defaultdict()
        for arg in argspec.args:
            if arg != "self":
                input_args[arg] = getattr(cls, arg)

        forc_func = forc_input_func(cls, **input_args)
        setattr(cls, var + '_value',
                cls.core.add_forcing(label=forc_label, forcing_func=forc_func))


def component(cls=None, *, init_stage=3):
    """ component decorator
    that converts simple base class using phydra.backend.variables into fully functional xarray simlab process
    """

    def create_component(cls):

        attr_cls = attr.attrs(cls, repr=False)
        vars_dict = _create_variables_dict(attr_cls)
        forcing_dict = _create_forcing_dict(cls, vars_dict)

        new_cls = _create_new_cls(cls, _create_xsimlab_var_dict(vars_dict), init_stage)

        def flux_decorator(self, func):
            """ flux function decorator to unpack arguments """

            @wraps(func)
            def unpack_args(**kwargs):
                state = kwargs.get('state')
                parameters = kwargs.get('parameters')
                forcings = kwargs.get('forcings')

                input_args = {}

                for v_dict in self.flux_input_args['vars']:
                    if isinstance(v_dict['label'], list) or isinstance(v_dict['label'], np.ndarray):
                        input_args[v_dict['var']] = [state[label] for label in v_dict['label']]
                    else:
                        input_args[v_dict['var']] = state[v_dict['label']]

                for v_dict in self.flux_input_args['list_input_vars']:
                    input_args[v_dict['var']] = np.concatenate([state[label] for label in v_dict['label']],
                                                               axis=None)

                for v_dict in self.flux_input_args['group_args']:
                    states = [state[label] for label in v_dict['label']]
                    if len(states) == 1:
                        # unpack list to array, for easier handling of single group arg
                        input_args[v_dict['var']] = states[0]
                    else:
                        input_args[v_dict['var']] = [state[label] for label in v_dict['label']]

                for p_dict in self.flux_input_args['pars']:
                    input_args[p_dict['var']] = parameters[p_dict['label']]

                for f_dict in self.flux_input_args['forcs']:
                    input_args[f_dict['var']] = forcings[f_dict['label']]

                return func(self, **input_args)

            return unpack_args

        def initialize(self):
            """ """
            super(new_cls, self).initialize()

            _initialize_process_vars(self, vars_dict)

            self.flux_input_args = _create_flux_inputargs_dict(self, vars_dict)

            _initialize_fluxes(self, vars_dict)

            _initialize_forcings(self, forcing_dict)

        setattr(new_cls, 'flux_decorator', flux_decorator)
        setattr(new_cls, 'initialize', initialize)

        process_cls = xs.process(new_cls)

        # allow passing helper functions through to process class
        _forcing_input_functions = [value.__name__ for value in forcing_dict.values()]
        cls_dir = dir(cls)
        for attribute in cls_dir:
            if hasattr(cls, attribute) and callable(getattr(cls, attribute)):
                if not attribute.startswith("__") and attribute not in _forcing_input_functions:
                    # Allow setting custom attr method, to be used in component
                    setattr(process_cls, attribute, getattr(cls, attribute))

        return process_cls

    if cls:
        return create_component(cls)

    return create_component
