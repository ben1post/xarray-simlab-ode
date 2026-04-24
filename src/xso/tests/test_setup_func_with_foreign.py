"""Tests for xso.parameter with setup_func that takes a foreign argument.

Combines the foreign-parameter and setup-func mechanisms: a derived parameter
whose setup_func pulls a foreign array from another component, then processes
it. Mirrors the phiPZ_setup pattern used in the Cariaco size-spectrum model.
"""

import numpy as np
import xso


def test_setup_func_with_foreign_array_arg():
    """A setup_func receives a foreign array parameter, transforms it, and the
    result is correctly passed into a flux."""

    @xso.component
    class Spectrum:
        value = xso.variable(dims='bin', description='per-bin state var')
        sizes = xso.parameter(broadcast=True, dims='bin',
                              description='per-bin sizes')
        bin = xso.index(dims='bin', description='bin index')

    @xso.component
    class Scaled:
        out = xso.variable(dims='bin', description='per-bin output',
                           flux='copy_flux')
        foreign_sizes = xso.parameter(foreign=True, dims='bin')
        factor = xso.parameter()
        scaled_sizes = xso.parameter(dims='bin', setup_func='scaled_setup')

        def scaled_setup(self, foreign_sizes, factor):
            return factor * np.asarray(foreign_sizes)

        @xso.flux(dims='bin')
        def copy_flux(self, out, scaled_sizes):
            return scaled_sizes

    model = xso.create({'spectrum': Spectrum, 'scaled': Scaled})

    sizes_input = np.array([1.0, 2.0, 3.0, 4.0])

    setup = xso.setup(
        solver='solve_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'spectrum__value_label': 'value',
            'spectrum__value_init': np.zeros(4),
            'spectrum__sizes_label': 'bin_sizes',
            'spectrum__sizes': sizes_input,
            'spectrum__bin_index': np.arange(4),
            'scaled__out_label': 'out',
            'scaled__out_init': np.zeros(4),
            'scaled__foreign_sizes': 'bin_sizes',
            'scaled__factor': 2.0,
        },
    )

    out = setup.xsimlab.run(model=model)
    assert np.allclose(out['scaled__out'].values[:, -1], 2.0 * sizes_input)