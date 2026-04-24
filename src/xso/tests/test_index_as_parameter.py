"""Tests for xso.index with as_parameter=True.

An index with as_parameter=True additionally registers its values as a
broadcast parameter, so other components can foreign-reference it as
numeric data. The attribute name on the component may differ from the
dimension name — the index labels the dimension named in its ``dims``
argument, while foreign references use the attribute name.
"""

import numpy as np
import xso


def test_index_as_parameter_split_names():
    """An index with as_parameter=True and an attribute name different from
    the dim name is foreign-referenceable under the attribute name, and the
    dimension coordinate of the output dataset carries the index values."""

    @xso.component
    class Spectrum:
        value = xso.variable(dims='phyto', description='per-bin state var')
        phyto_esd = xso.index(dims='phyto', as_parameter=True,
                              description='per-bin sizes')

    @xso.component
    class Scaled:
        out = xso.variable(dims='phyto', description='per-bin output',
                           flux='copy_flux')
        phyto_esd = xso.parameter(foreign=True, dims='phyto')
        factor = xso.parameter()
        scaled_esd = xso.parameter(dims='phyto', setup_func='scaled_setup')

        def scaled_setup(self, phyto_esd, factor):
            return factor * np.asarray(phyto_esd)

        @xso.flux(dims='phyto')
        def copy_flux(self, out, scaled_esd):
            return scaled_esd

    model = xso.create({'spectrum': Spectrum, 'scaled': Scaled})

    esd_values = np.array([1.0, 2.0, 3.0, 4.0])

    setup = xso.setup(
        solver='solve_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'spectrum__value_label': 'value',
            'spectrum__value_init': np.zeros(4),
            'spectrum__phyto_esd_index': esd_values,
            'spectrum__phyto_esd_label': 'phyto_esd',
            'scaled__out_label': 'out',
            'scaled__out_init': np.zeros(4),
            'scaled__phyto_esd': 'phyto_esd',
            'scaled__factor': 2.0,
        },
    )

    out = setup.xsimlab.run(model=model)
    assert np.allclose(out['scaled__out'].values[:, -1], 2.0 * esd_values)