"""Tests for xso.parameter with setup_func.

A setup_func parameter is computed once at model initialization from other
variables declared on the same component. The user cannot supply its value
via input_vars — the setup_func is authoritative. Arguments of the setup
function must match names of declared variables on the component; anything
else raises with a clear error.
"""

import numpy as np
import pytest
import xso


def test_local_setup_func_computes_value():
    """A setup_func with only local-parameter args resolves, is readable by
    fluxes, and the output reflects the computation (2 * base)."""

    @xso.component
    class Doubler:
        out = xso.variable(description='receives the doubled value',
                           flux='write_flux')
        base = xso.parameter(description='user-supplied base value')
        doubled = xso.parameter(setup_func='doubled_setup',
                                description='computed as 2 * base')

        def doubled_setup(self, base):
            return 2.0 * base

        @xso.flux
        def write_flux(self, out, base, doubled):
            return doubled

    model = xso.create({'doubler': Doubler})

    setup = xso.setup(
        solver='solve_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'doubler__out_label': 'out',
            'doubler__out_init': 0.0,
            'doubler__base': 3.0,
        },
    )

    out = setup.xsimlab.run(model=model)
    assert np.isclose(out['doubler__out'].values[-1], 6.0)


def test_setup_func_with_invalid_arg_raises():
    """A setup_func declaring an argument name that is not a declared XSO
    variable on the component must raise a clear ValueError at run time.

    This prevents the silent failure where an arg name coincidentally matches
    some attribute of the underlying xsimlab process class (e.g. 'core',
    'label') and would otherwise be resolved to an unrelated object via
    getattr.
    """

    @xso.component
    class BadSetup:
        out = xso.variable(description='state var', flux='f')
        good = xso.parameter(setup_func='bad_setup',
                             description='param whose setup_func has a bad arg')

        def bad_setup(self, core):
            return 1.0

        @xso.flux
        def f(self, out, good):
            return good

    model = xso.create({'bad_setup': BadSetup})

    setup = xso.setup(
        solver='solve_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'bad_setup__out_label': 'out',
            'bad_setup__out_init': 0.0,
        },
    )

    with pytest.raises(ValueError, match="'core'"):
        setup.xsimlab.run(model=model)