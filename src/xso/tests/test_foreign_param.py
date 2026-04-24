"""Tests for xso.parameter with broadcast=True / foreign=True.

A broadcast parameter is registered under a user-supplied label string
(via the source-side ``_label`` input slot), mirroring how xso.variable
foreign references work. Consumers reference it by passing the same label
string into their foreign parameter's slot. Passing a value instead of a
label raises a clear TypeError.
"""

import numpy as np
import pytest
import xso


def test_foreign_param_reads_source_value():
    """A foreign parameter resolves to the value of the broadcast parameter
    on the source component, via the user-supplied label string."""

    @xso.component
    class Source:
        value = xso.parameter(broadcast=True, description='broadcast value')

    @xso.component
    class Consumer:
        out = xso.variable(description='receives the foreign value',
                           flux='read_flux')
        src_val = xso.parameter(foreign=True,
                                description='foreign ref')

        @xso.flux
        def read_flux(self, out, src_val):
            return src_val

    model = xso.create({'src': Source, 'consumer': Consumer})

    setup = xso.setup(
        solver='solve_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'consumer__out_label': 'out',
            'consumer__out_init': 0.0,
            'src__value_label': 'my_value',
            'src__value': 7.0,
            'consumer__src_val': 'my_value',
        },
    )

    out = setup.xsimlab.run(model=model)
    assert np.isclose(out['consumer__out'].values[-1], 7.0)


def test_foreign_param_rejects_non_string_value():
    """Passing a numeric value into a foreign parameter's input slot — rather
    than a label string — must raise TypeError with a clear message pointing
    at the offending component and variable.
    """

    @xso.component
    class Src:
        value = xso.parameter(broadcast=True)

    @xso.component
    class Cons:
        out = xso.variable(flux='f')
        src_val = xso.parameter(foreign=True)

        @xso.flux
        def f(self, out, src_val):
            return src_val

    model = xso.create({'src': Src, 'cons': Cons})

    setup = xso.setup(
        solver='solve_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'cons__out_label': 'out',
            'cons__out_init': 0.0,
            'src__value_label': 'my_value',
            'src__value': 7.0,
            'cons__src_val': 7.0,  # WRONG — value instead of label
        },
    )

    with pytest.raises(TypeError, match="label STRING"):
        setup.xsimlab.run(model=model)