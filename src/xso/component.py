import xsimlab as xs

import attr
from attr import fields_dict

from collections import OrderedDict, defaultdict, Counter
from functools import wraps
import inspect
import numpy as np

from .variables import XSOVarType
from .backendcomps import FirstInit, SecondInit, ThirdInit, FourthInit, FifthInit


def _create_variables_dict(process_cls):
    """Get all phydra variables declared in a component.
    Exclude attr.Attribute objects that are not XSO specific.
    """
    return OrderedDict(
        (k, v) for k, v in fields_dict(process_cls).items() if "var_type" in v.metadata
    )


def _convert_2_xsimlabvar(var, intent='in',
                          var_dims=None, value_store=False, groups=None,
                          description_label='', attrs=True):
    """Converts XSO variables to xarray-simlab variables to be used in the model backend.

    Function receives variable as attr in _make_xsimlab_vars function and extracts
    description, dimensions and metadata, then passes it and additional arguments
    through xarray-simlab's xs.variable function.

    Parameters
    ----------
    var : attr._Make.Attribute
        XSO variable defined in object decorated with xso.component()
    intent : str ('in' or 'out')
        passed along, defines variable as receiving input from another component
        or being initialized within this component.
    var_dims : tuple or str
        Defines dimensionality of created xsimlab variable,
        can be singular string or tuple of strings.
    value_store : bool
        When true, the ouput of variable is stored to xsimlab variable. Adds 'time' dimension.
    groups : str
        When defined, the variable output can be referenced via xarray simlabs
        group variable in other XSO components.
    description_label : str
        Description stored with Xarray Dataset created by xsimlab.
    attrs : bool
        If true, attrs defined in variable metadata are passed along to xsimlab variable function.

    Returns
    -------
    xs.variable
        attr class handled by Xarray-simlab, the functional foundation of XSO
    """
    # get variable metadata
    var_description = var.metadata.get('description')
    if var_description:
        description_label = description_label + var_description

    if var_dims is None:
        var_dims = var.metadata.get('dims')

    # initialize dimensions, with time if value_store is true
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

    # return fully functional xarray-simlab variable
    return xs.variable(intent=intent, dims=var_dims, groups=groups,
                       description=description_label, attrs=var_attrs)


def _make_xso_variable(label, variable):
    """Checks for type of variable defined and calls _convert_2_xsimlabvar function
    accordingly. Returns dict with label and xsimlab variable as key/value pairs.
    """
    xs_var_dict = defaultdict()
    if variable.metadata.get('foreign') is True:
        list_input = variable.metadata.get('list_input')
        if list_input:
            var_dims = variable.metadata.get('dims')
            if var_dims is None:
                raise ValueError("Variable with list_input=True requires passing dimension to dims keyword argument")
            xs_var_dict[label] = _convert_2_xsimlabvar(var=variable, var_dims=var_dims,
                                                       description_label='label list / ')
        else:
            xs_var_dict[label] = _convert_2_xsimlabvar(var=variable, var_dims=(),
                                                       description_label='label reference / ')
    elif variable.metadata.get('foreign') is False:
        xs_var_dict[label + '_label'] = _convert_2_xsimlabvar(var=variable, var_dims=(),
                                                              description_label='label / ')
        xs_var_dict[label + '_init'] = _convert_2_xsimlabvar(var=variable, description_label='initial value / ')
        xs_var_dict[label] = _convert_2_xsimlabvar(var=variable, intent='out',
                                                   value_store=True,
                                                   description_label='output of variable / ')
    return xs_var_dict


def _make_xso_parameter(label, variable):
    """Checks for type of variable defined and calls _convert_2_xsimlabvar function
    accordingly. Returns dict with label and xsimlab variable as key/value pairs.
    """
    xs_var_dict = defaultdict()
    xs_var_dict[label] = _convert_2_xsimlabvar(var=variable, description_label='parameter / ')
    return xs_var_dict


def _make_xso_forcing(label, variable):
    """Checks for type of variable defined and calls _convert_2_xsimlabvar function
    accordingly. Returns dict with label and xsimlab variable as key/value pairs.
    """
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
    """Checks for type of variable defined and calls _convert_2_xsimlabvar function
    accordingly. Returns dict with label and xsimlab variable as key/value pairs.
    """
    xs_var_dict = defaultdict()
    xs_var_dict[label + '_value'] = _convert_2_xsimlabvar(var=variable, intent='out',
                                                          value_store=True,
                                                          description_label='output of flux value / ')
    group = variable.metadata.get('group')
    group_to_arg = variable.metadata.get('group_to_arg')

    if group:
        xs_var_dict[label + '_label'] = _convert_2_xsimlabvar(var=variable, intent='out', groups=group, var_dims=(),
                                                              description_label='label reference with group / ',
                                                              attrs=False)
    if group_to_arg:
        xs_var_dict[group_to_arg] = xs.group(group_to_arg)

    return xs_var_dict


def _make_xso_index(label, variable):
    """Checks for type of variable defined and calls _convert_2_xsimlabvar function
    accordingly. Returns dict with label and xsimlab variable as key/value pairs.
    """
    xs_var_dict = defaultdict()

    description_label = 'index / '
    # get variable metadata
    var_description = variable.metadata.get('description')
    if var_description:
        description_label = description_label + var_description

    var_dims = variable.metadata.get('dims')

    if var_dims is None:
        raise ValueError("Argument dims is not supplied. Index variable requires passing the labels of dimension to 'dims' keyword.")

    if label != var_dims:
        raise ValueError("The variable name has to be the same as the dimension it labels. This is a requirement of xarray-simlab.")

    if variable.metadata.get('attrs'):
        var_attrs = variable.metadata.get('attrs')
    else:
        var_attrs = {}

    xs_var_dict[label] = xs.index(dims=var_dims, description=description_label, attrs=var_attrs)
    xs_var_dict[label + '_index'] = _convert_2_xsimlabvar(var=variable, description_label='index / ')
    return xs_var_dict


_make_xsimlab_vars = {
    XSOVarType.VARIABLE: _make_xso_variable,
    XSOVarType.FORCING: _make_xso_forcing,
    XSOVarType.PARAMETER: _make_xso_parameter,
    XSOVarType.FLUX: _make_xso_flux,
    XSOVarType.INDEX: _make_xso_index,
}


def _create_xsimlab_var_dict(cls_vars):
    """Parses through attributes defined in xso.component decorated class
    and extracts those relevant for XSO.

    These are initialized as xsimlab variables
    and returned in dict to xso.component decorated class.
    """
    xs_var_dict = defaultdict()

    for key, var in cls_vars.items():
        var_type = var.metadata.get('var_type')
        var_dict = _make_xsimlab_vars[var_type](key, var)

        for xs_key, xs_var in var_dict.items():
            xs_var_dict[xs_key] = xs_var

    return xs_var_dict


def _create_forcing_dict(cls, var_dict):
    """Parses var_dict and extracts forcing setup function"""
    forcings_dict = defaultdict()

    for key, var in var_dict.items():
        if var.metadata.get('var_type') is XSOVarType.FORCING:
            _forcing_setup_func = var.metadata.get('setup_func')
            if _forcing_setup_func is not None:
                forcings_dict[key] = getattr(cls, _forcing_setup_func)

    return forcings_dict


def _create_index_dict(cls, var_dict):
    """Parses var_dict and extracts index setup function"""
    index_dict = defaultdict()

    for key, var in var_dict.items():
        if var.metadata.get('var_type') is XSOVarType.INDEX:
            index_dict[key] = var

    return index_dict


def _initialize_process_vars(cls, vars_dict):
    """Parses vars_dict of xso.component decorated class, and
    initializes defined variables and methods with model backend.
    """
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
                setattr(cls, key, cls.core.add_variable(label=_label, initial_value=_init))  # + '_value'

            flux_label = var.metadata.get('flux')
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
                    raise ValueError(
                        f"Variable {_label} was assigned {flux_label} with negative arguments {flux_negative}, "
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
                raise Exception("Sorry, currently XSO does not support foreign=True for parameters.")


def _create_flux_inputargs_dict(cls, vars_dict):
    """Creates a dictionary to parse the input arguments to a flux function."""
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
            # TODO: Implement foreign parameters here
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
    """Parses flux variables and methods in xso.component decorated class
    and registers them with the model backend.
    """
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
    """Parses xso.forcing variables and methods defined in xso.component decorated class
    and registers them with the model backend.
    """
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


def _initialize_indexes(cls, index_dict):
    """Initializes xso.index variables defined in xso.component decorated class."""
    for var, val in index_dict.items():
        # apply index value to XSO Index variable type
        index_value = getattr(cls, var + '_index')
        setattr(cls, var, index_value)



def _get_init_stage(vars_dict):
    """Returns the initialization stage of the component
    attempts to automatically determine init stage from implemented variable types and group arguments

    :param vars_dict: dictionary of variables in component

    :return: init stage
    """

    vars_list = []
    groups = 0
    groups_to_arg = 0

    for key, var in vars_dict.items():
        vars_list.append(var.metadata.get('var_type'))
        if var.metadata.get('group'):
            groups += 1
        if var.metadata.get('group_to_arg'):
            groups_to_arg += 1

    # count number of variables of each type
    count_vars = Counter(vars_list)

    if groups_to_arg > 0:
        init_stage_automated = "fifth"
    elif count_vars[XSOVarType.FLUX] > 0:
        init_stage_automated = "fourth"
    elif count_vars[XSOVarType.FORCING] > 0:
        init_stage_automated = "third"
    else:
        init_stage_automated = "second"

    return init_stage_automated


def _create_new_cls(cls, cls_dict, init_stage):
    """Method to initialize XSO component with appropriate parent class
    from xso.backendcomps, which defines initialisation stage and
    inherits from Context class.
    """

    if init_stage == "first":
        new_cls = type(cls.__name__, (FirstInit,), cls_dict)
    elif init_stage == "second":
        new_cls = type(cls.__name__, (SecondInit,), cls_dict)
    elif init_stage == "third":
        new_cls = type(cls.__name__, (ThirdInit,), cls_dict)
    elif init_stage == "fourth":
        new_cls = type(cls.__name__, (FourthInit,), cls_dict)
    elif init_stage == "fifth":
        new_cls = type(cls.__name__, (FifthInit,), cls_dict)
    else:
        raise Exception("There was an error with the sorting of processes. The automatic sorting failed.")
    return new_cls


def component(cls=None):
    """A component decorator that adds everything needed to use the class
    as a XSO component. It is a wrapper for the xarray-simlab process decorator.

    A component represents a logical unit with the model, and usually implements:

    - A set of XSO variables, defined as class attributes, e.g. xso.variable, xso.parameter,
    xso.forcing or xso.flux.
    - One or more methods that can be functions defining a xso.flux or a xso.forcing.

    Parameters
    __________
    cls : class, optional
        Allows applying this decorator either as @xso.component or @xso.component(*args).

    Returns
    _______
    cls : class
        The decorated class that is a fully functional xso.component.
    """

    def create_component(cls):
        """Function to construct new class and return xarray-simlab process"""
        attr_cls = attr.attrs(cls, repr=False)
        vars_dict = _create_variables_dict(attr_cls)
        forcing_dict = _create_forcing_dict(cls, vars_dict)
        index_dict = _create_index_dict(cls, vars_dict)

        # implement a basic automatic process ordering
        init_stage_automated = _get_init_stage(vars_dict)

        new_cls = _create_new_cls(cls, _create_xsimlab_var_dict(vars_dict), init_stage_automated)

        def flux_decorator(self, func):
            """XSO flux function decorator to unpack arguments"""

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
            """Defines xarray-simlab process `initialize` method
            that is executed at model runtime.
            """
            super(new_cls, self).initialize()

            _initialize_process_vars(self, vars_dict)

            self.flux_input_args = _create_flux_inputargs_dict(self, vars_dict)

            _initialize_fluxes(self, vars_dict)

            _initialize_forcings(self, forcing_dict)

            _initialize_indexes(self, index_dict)


        setattr(new_cls, 'flux_decorator', flux_decorator)
        setattr(new_cls, 'initialize', initialize)

        # constructed class is passed through xsimlab process decorator:
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
