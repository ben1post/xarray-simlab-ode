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


def test_broadcast_param_used_by_own_flux():
    """A flux declared on the same component that declares a broadcast
    parameter can read that parameter directly. Regression test: the
    source-side flux-arg label must resolve to the user-supplied broadcast
    label, not the auto-generated '<Component>_<param>' label.
    """

    @xso.component
    class StateX:
        value = xso.variable(description='a state variable')

    @xso.component
    class SourceX:
        target = xso.variable(foreign=True, flux='grow', negative=False)
        p_shared = xso.parameter(broadcast=True, description='broadcast parameter')

        @xso.flux
        def grow(self, target, p_shared):
            return p_shared

    @xso.component
    class ConsumerX:
        target = xso.variable(foreign=True, flux='shrink', negative=True)
        p_ref = xso.parameter(foreign=True, description='foreign ref to broadcast param')

        @xso.flux
        def shrink(self, target, p_ref):
            return p_ref * target

    model = xso.create({'State': StateX, 'Source': SourceX, 'Consumer': ConsumerX})

    setup = xso.setup(
        solver='solve_ivp',
        model=model,
        time=np.arange(0, 5, 0.1),
        input_vars={
            'State': {'value_label': 'X', 'value_init': 1.0},
            'Source': {'target': 'X', 'p_shared': 0.5, 'p_shared_label': 'shared_p'},
            'Consumer': {'target': 'X', 'p_ref': 'shared_p'},
        },
    )

    out = setup.xsimlab.run(model=model)
    # dX/dt = 0.5 - 0.5 X, X(0) = 1 -> steady state X = 1
    assert np.isclose(float(out['State__value'].values[-1]), 1.0)


def test_setup_func_and_broadcast_rejected():
    """Declaring a parameter with both setup_func and broadcast=True raises
    at declaration time (when the component class body executes)."""

    with pytest.raises(ValueError, match="setup_func cannot currently be combined"):

        @xso.component
        class Bad:
            value = xso.parameter(broadcast=True, setup_func='compute')

            def compute(self):
                return 1.0


def test_duplicate_broadcast_label_rejected():
    """Two components declaring a broadcast parameter under the same label
    must raise when the model is run (parameter registered twice)."""

    @xso.component
    class SrcA:
        a = xso.parameter(broadcast=True)

    @xso.component
    class SrcB:
        b = xso.parameter(broadcast=True)

    @xso.component
    class Cons:
        out = xso.variable(flux='f')
        ref = xso.parameter(foreign=True)

        @xso.flux
        def f(self, out, ref):
            return ref

    model = xso.create({'srcA': SrcA, 'srcB': SrcB, 'cons': Cons})

    setup = xso.setup(
        solver='solve_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'cons__out_label': 'out',
            'cons__out_init': 0.0,
            'srcA__a_label': 'dup',
            'srcA__a': 1.0,
            'srcB__b_label': 'dup',  # same label as srcA -> collision
            'srcB__b': 2.0,
            'cons__ref': 'dup',
        },
    )

    with pytest.raises(Exception, match="registered twice"):
        setup.xsimlab.run(model=model)