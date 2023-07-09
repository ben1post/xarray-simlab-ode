import attr

from enum import Enum


class XSOVarType(Enum):
    VARIABLE = "variable"
    PARAMETER = "parameter"
    FORCING = "forcing"
    FLUX = "flux"
    INDEX = "index"


def variable(foreign=False, flux=None, negative=False, list_input=False,
             dims=None, description='', attrs={}):
    """Create a state variable.

    This can be a local state variable for the component, or a reference to a state variable
    initialized in another component. A flux function can be applied to that variable.

    The variable stores a single state variable, if no dimension (argument: dims) is supplied,
    but can also define an array or matrix of state variables, depending on the argument.
    In the model it is always assigned to a single label, and can only be used in a flux
    with appropriate dimensionality.

    Parameters
    ----------
    foreign : boolean, optional
        Defines whether the variable is initialized and labeled in the component,
        or is simply a reference to a variable in another component.
    flux: str, optional
        Name of the flux function defined in this component, the result of which
        is applied to this state variable.
    negative : boolean, optional
        If true, the result of the flux function is substracted, if false,
        it is added to the variable.
    list_input : list, optional
        If it is a foreign variable, a list of labels of other state variables
        can be supplied, as a handy way of applying the same flux to multiple
        variables.
    dims : str or tuple or list, optional
        Dimension label(s) of the variable. An empty tuple
        corresponds to a scalar variable (default), a string or a 1-length
        tuple corresponds to a 1-d variable and a n-length tuple corresponds to
        a n-d variable. A list of str or tuple items may also be provided if
        the variable accepts different numbers of dimensions.
    description : str, optional
        Short description of the variable.
    attrs : dict, optional
        Dictionnary of additional metadata (e.g., standard_name,
        units, math_symbol...).
    """
    attrs.update({'Phydra_store_out': True})

    metadata = {
        "var_type": XSOVarType.VARIABLE,
        "foreign": foreign,
        "negative": negative,
        "flux": flux,
        "list_input": list_input,
        "dims": dims,
        "attrs": attrs,
        "description": description,
    }

    return attr.attrib(metadata=metadata)


def forcing(foreign=False,
            setup_func=None, dims=(), description='', attrs={}):
    """Create a forcing variable.

    This can be a local forcing variable for the component, or a reference to a forcing variable
    initialized in another component. A setup function can be registered to supply the forcing
    value.

    The forcing can be of variable dimensionality.

    Parameters
    ----------
    foreign : boolean, optional
        Defines whether the variable is initialized and labeled in the component,
        or is simply a reference to a variable in another component.
    setup_func: str, optional
        Name of the forcing setup function defined in this component, the result of which
        is defines this forcing. This allows supplying a forcing as a mathematical function,
        which is dynamically computed at each time-step, instead of a fixed array of values.
    dims : str or tuple or list, optional
        Dimension label(s) of the forcing. An empty tuple
        corresponds to a scalar variable (default), a string or a 1-length
        tuple corresponds to a 1-d variable and a n-length tuple corresponds to
        a n-d variable. A list of str or tuple items may also be provided if
        the variable accepts different numbers of dimensions.
    description : str, optional
        Short description of the forcing.
    attrs : dict, optional
        Dictionnary of additional metadata (e.g., standard_name,
        units, math_symbol...).
    """

    attrs.update({'Phydra_store_out': True})

    metadata = {
        "var_type": XSOVarType.FORCING,
        "foreign": foreign,
        "setup_func": setup_func,
        "dims": dims,
        "attrs": attrs,
        "description": description,
    }

    return attr.attrib(metadata=metadata)


def parameter(foreign=False, dims=(), description='', attrs=None):
    """Create a parameter.

    This can be a local parameter for the component, or a reference to a parameter
    initialized in another component.

    The parameter can be of variable dimensionality.

    Parameters
    ----------
    foreign : boolean, optional
        Defines whether the parameter is initialized and labeled in the component,
        or is simply a reference to a variable in another component.
    dims : str or tuple or list, optional
        Dimension label(s) of the forcing. An empty tuple
        corresponds to a scalar variable (default), a string or a 1-length
        tuple corresponds to a 1-d variable and a n-length tuple corresponds to
        a n-d variable. A list of str or tuple items may also be provided if
        the variable accepts different numbers of dimensions.
    description : str, optional
        Short description of the parameter.
    attrs : dict, optional
        Dictionnary of additional metadata (e.g., standard_name,
        units, math_symbol...).
    """
    metadata = {
        "var_type": XSOVarType.PARAMETER,
        "foreign": foreign,
        "dims": dims,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata)


def flux(flux_func=None, *, dims=(), group=None, group_to_arg=None, description='', attrs={}):
    """Create a flux function.

    This is a function decorator that registers a method within a component
    as a flux (i.e. term in the ODE). It ta

    The parameter can be of variable dimensionality.

    Parameters
    ----------
    flux_func : decorator argument
        Allows decorator to be used with and without arguments.
    dims : str or tuple or list, optional
        Dimension label(s) of the forcing. An empty tuple
        corresponds to a scalar variable (default), a string or a 1-length
        tuple corresponds to a 1-d variable and a n-length tuple corresponds to
        a n-d variable. A list of str or tuple items may also be provided if
        the variable accepts different numbers of dimensions.
    group : str, optional
        Output of flux is stored in xsimlab group variable, to be referenced
        in other fluxes, via the group_to_arg argument. This way the flux output
        can be routed to multiple other fluxes, for more complex mathematical constructs.
    group_to_arg : str, optional
        The string supplied is used as a reference for an xsimlab group variable.
        This group variable collects the output of fluxes in other components, that
        have defined the same string label for the group argument. The values thus
        collected are inserted into the function as a variable of the same name,
        for further computations.
    description : str, optional
        Short description of the flux.
    attrs : dict, optional
        Dictionnary of additional metadata (e.g., standard_name,
        units, math_symbol...).
    """

    def create_attrib(function):

        attrs.update({'Phydra_store_out': True})

        metadata = {
            "var_type": XSOVarType.FLUX,
            "flux_func": function,
            "group": group,
            "group_to_arg": group_to_arg,
            "dims": dims,
            "attrs": attrs,
            "description": description,
        }
        return attr.attrib(metadata=metadata)

    if flux_func:
        return create_attrib(flux_func)

    return create_attrib


def index(foreign=False, dims=(), description='', attrs=None):
    """Create an index.

    This has to be be a local index for the component.

    The index can be of variable dimensionality.

    Parameters
    ----------
    foreign : boolean, optional
        Defines whether the parameter is initialized and labeled in the component,
        or is simply a reference to a variable in another component.
    dims : str or tuple or list, optional
        Dimension label(s) of the forcing. An empty tuple
        corresponds to a scalar variable (default), a string or a 1-length
        tuple corresponds to a 1-d variable and a n-length tuple corresponds to
        a n-d variable. A list of str or tuple items may also be provided if
        the variable accepts different numbers of dimensions.
    description : str, optional
        Short description of the parameter.
    attrs : dict, optional
        Dictionnary of additional metadata (e.g., standard_name,
        units, math_symbol...).
    """
    metadata = {
        "var_type": XSOVarType.INDEX,
        "foreign": foreign,
        "dims": dims,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata)