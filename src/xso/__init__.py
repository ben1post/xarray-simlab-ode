# -----------------------------------------------------------------------------
# Upstream warning suppression
#
# The following FutureWarnings originate from xarray-simlab (our foundation),
# not from XSO itself, and fire on essentially every model run. They are
# suppressed here so that XSO users aren't spammed by them.
#
# When the upstream package is patched, the corresponding entries below
# should be removed.
#
# Tracked upstream at: https://github.com/benbovy/xarray-simlab
# -----------------------------------------------------------------------------
import warnings as _warnings

_warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message=r".*with name matching its dimension will not be automatically "
            r"converted into an `IndexVariable` object.*",
    module=r"xsimlab\..*",
)
_warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message=r".*dropping variables using `drop` is deprecated; use drop_vars.*",
    module=r"xsimlab\..*",
)

from .component import component

from .variables import variable, parameter, forcing, flux, index

from .xsimlabwrappers import create, setup
