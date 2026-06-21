"""Tests for sweeping array-valued parameters in 1D parscans.

Array params (e.g. a per-class ``dims='zoo'`` parameter) used to be
rejected by the 1D scan path: the raw array was assigned as the new-
dimension coordinate and ``expand_dims`` then failed. The 1D path now
mirrors the 2D path by scalarizing the swept value via
``_scalarize_param_for_coord`` (np.max) for both the output coordinate
and the ``initial_values_ds`` lookup.

These tests exercise the two changed helpers directly, avoiding the
multiprocessing Pool / import-by-name machinery of ``run_xso_parscan``.
"""

import numpy as np
import xarray as xr

from xso.parscans import (
    _scalarize_param_for_coord,
    _generate_iterable_tasks_1d,
    _unpack_par_scan_1d,
)


def test_scalarize_param_for_coord():
    """Scalars pass through; uniform arrays collapse to their value."""
    assert _scalarize_param_for_coord(0.3) == 0.3
    assert _scalarize_param_for_coord(np.full(5, 0.3)) == 0.3
    # np.max rule, as documented
    assert _scalarize_param_for_coord(np.array([0.0, 0.0, 0.005])) == 0.005


def test_unpack_1d_array_param_builds_scalar_coord():
    """A 1D scan over an array-valued param produces a scalar coordinate
    (np.max of each array) and a clean stacked Dataset — no expand_dims
    error."""
    param = 'Grazing__KsZ'
    arrays = [np.full(3, 0.1), np.full(3, 0.2)]

    iterable_tasks = [{'scan_param_dict': {param: a}} for a in arrays]
    data = [
        xr.Dataset({'biomass': ('time', [1.0, 2.0])}, coords={'time': [0, 1]})
        for _ in arrays
    ]

    out = _unpack_par_scan_1d(iterable_tasks, data)

    assert param in out.dims
    np.testing.assert_array_equal(out[param].values, [0.1, 0.2])
    # biomass stacked as (Grazing__KsZ, time)
    assert out['biomass'].sizes[param] == 2
    assert out['biomass'].sizes['time'] == 2


def test_unpack_1d_scalar_param_unchanged():
    """Regression: scalar swept params still stack correctly after the
    scalarization change."""
    param = 'Inflow__FN'
    values = [5.0, 10.0]

    iterable_tasks = [{'scan_param_dict': {param: v}} for v in values]
    data = [
        xr.Dataset({'biomass': ('time', [1.0, 2.0])}, coords={'time': [0, 1]})
        for _ in values
    ]

    out = _unpack_par_scan_1d(iterable_tasks, data)

    assert param in out.dims
    np.testing.assert_array_equal(out[param].values, [5.0, 10.0])


def test_generate_1d_array_param_with_initial_values():
    """The initial-values injection path scalarizes the array swept value
    for the .sel lookup (nearest on the scalar coordinate), and still
    passes the full array through as the model input."""
    param = 'Grazing__KsZ'
    arrays = [np.full(3, 0.1), np.full(3, 0.2)]

    initial_values_ds = xr.Dataset(
        {'Phytoplankton__biomass': (param, [5.0, 6.0])},
        coords={param: [0.1, 0.2]},
    )
    iv_mapping = {'Phytoplankton__biomass': 'Phytoplankton__biomass_init'}

    tasks = _generate_iterable_tasks_1d(
        param, arrays,
        initial_values_ds=initial_values_ds,
        iv_mapping=iv_mapping,
    )

    assert len(tasks) == 2

    # Full array is passed through as the model input (not scalarized).
    np.testing.assert_array_equal(
        tasks[0]['scan_param_dict'][param], np.full(3, 0.1))
    np.testing.assert_array_equal(
        tasks[1]['scan_param_dict'][param], np.full(3, 0.2))

    # Injected initial value resolved via nearest on the scalar coord
    # (max(0.1-array) -> 0.1 -> 5.0; max(0.2-array) -> 0.2 -> 6.0).
    assert float(tasks[0]['scan_param_dict']['Phytoplankton__biomass_init']) == 5.0
    assert float(tasks[1]['scan_param_dict']['Phytoplankton__biomass_init']) == 6.0
