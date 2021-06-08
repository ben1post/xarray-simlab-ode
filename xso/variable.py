import attr

from enum import Enum


class PhydraVarType(Enum):
    VARIABLE = "variable"
    PARAMETER = "parameter"
    FORCING = "forcing"
    FLUX = "flux"


def variable(foreign=False, flux=None, negative=False, list_input=False,
             dims=None, description='', attrs={}):

    attrs.update({'Phydra_store_out': True})

    metadata = {
        "var_type": PhydraVarType.VARIABLE,
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

    attrs.update({'Phydra_store_out': True})

    metadata = {
        "var_type": PhydraVarType.FORCING,
        "foreign": foreign,
        "setup_func": setup_func,
        "dims": dims,
        "attrs": attrs,
        "description": description,
    }

    return attr.attrib(metadata=metadata)


def parameter(foreign=False, dims=(), description='', attrs=None):

    metadata = {
        "var_type": PhydraVarType.PARAMETER,
        "foreign": foreign,
        "dims": dims,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata)


def flux(flux_func=None, *, dims=(), group=None, group_to_arg=None, description='', attrs={}):
    """ decorator arg setup allows to be applied to function with and without args """

    def create_attrib(function):

        attrs.update({'Phydra_store_out': True})

        metadata = {
            "var_type": PhydraVarType.FLUX,
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
