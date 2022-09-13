from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from .component import component
from .model import Model

from .variables import variable, parameter, forcing, flux

from .xsimlabwrappers import create, setup