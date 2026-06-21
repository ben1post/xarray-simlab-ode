"""Quick checks for linked_overrides in xso.parscans (1D trajectory scan).

Run:  python test_linked_overrides.py
  or: pytest test_linked_overrides.py

Exercises the three changed pure functions without a solver or the
multiprocessing pool, including the failed-cell and non-monotonic-axis
paths. This does NOT replace an end-to-end run of your real Inflow__FN
scan, but it covers where all the subtlety lives (the unpack drop +
reattach + combine_by_coords).
"""
import numpy as np
import xarray as xr

from xso.parscans import (
    _validate_linked_overrides,
    _generate_iterable_tasks_1d,
    _unpack_par_scan_1d,
)

# Deliberately non-monotonic so we prove linked values track the scan axis
# in lockstep even after combine_by_coords reorders it.
FN = np.array([0.4, 0.2, 0.9, 0.6])
de = np.clip(55.89 - 3.966 * FN, 19.0, 69.0)
T = np.clip(25.64 - 0.414 * FN, 22.0, 29.0)
LINKED = {'Inflow__de': de, 'Temperature__value': T}


def _fake_worker_cell(fn_i, de_i, t_i):
    """Mimic _run_model_scan_point output: a time-series data var plus the
    scan + linked params promoted to 0-d scalar coords (what assign_coords
    of scalar_coords produces in the worker).
    """
    ds = xr.Dataset(
        {'biomass': ('time', np.array([1.0, 2.0, 3.0]) + fn_i)},
        coords={'time': [0.0, 1.0, 2.0]},
    )
    return ds.assign_coords(
        {'Inflow__FN': fn_i, 'Inflow__de': de_i, 'Temperature__value': t_i}
    )


def test_validate():
    _validate_linked_overrides(LINKED, FN, ['Inflow__FN'], None)  # valid: no raise

    for bad, args, label in [
        ({'X': de[:-1]}, (FN, ['Inflow__FN'], None), 'length mismatch'),
        ({'Inflow__FN': de}, (FN, ['Inflow__FN'], None), 'scanned-axis collision'),
        ({'Inflow__de': de}, (FN, ['Inflow__FN'], {'Inflow__de': 50.0}),
         'fixed-overrides collision'),
    ]:
        try:
            _validate_linked_overrides(bad, *args)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"{label} should have raised")
    print("validate: OK")


def test_generate():
    tasks = _generate_iterable_tasks_1d(
        'Inflow__FN', FN, fixed_overrides={'K__half': 1.0},
        linked_overrides=LINKED,
    )
    assert len(tasks) == len(FN)
    for i, t in enumerate(tasks):
        d = t['scan_param_dict']
        assert list(d)[0] == 'Inflow__FN'              # primary key is first
        assert d['Inflow__de'] == float(de[i])
        assert isinstance(d['Inflow__de'], float)      # plain float, not 0-d np
        assert d['Temperature__value'] == float(T[i])
        assert d['K__half'] == 1.0                      # fixed carried through
    print("generate: OK")


def test_unpack_with_failed_cell():
    tasks = _generate_iterable_tasks_1d('Inflow__FN', FN, linked_overrides=LINKED)
    data = [_fake_worker_cell(FN[i], de[i], T[i]) for i in range(len(FN))]
    data[2] = None  # one failed cell

    scan = _unpack_par_scan_1d(tasks, data, linked_overrides=LINKED)

    # scan axis present (combine_by_coords sorts it)
    assert 'Inflow__FN' in scan.coords
    assert np.allclose(np.sort(scan['Inflow__FN'].values), np.sort(FN))

    # linked params are 1-D along Inflow__FN, correct EVEN at the failed cell
    for k, arr in LINKED.items():
        assert scan[k].dims == ('Inflow__FN',), f"{k} dims = {scan[k].dims}"
        for fn_i, v_i in zip(FN, arr):
            got = float(scan[k].sel(Inflow__FN=fn_i))
            assert np.isclose(got, v_i), f"{k} at FN={fn_i}: {got} != {v_i}"
        assert np.isfinite(scan[k].values).all(), f"{k} unexpectedly has NaN"

    # the data var is NaN-filled only at the failed cell
    assert np.isnan(scan['biomass'].sel(Inflow__FN=FN[2])).all()
    assert np.isfinite(scan['biomass'].sel(Inflow__FN=FN[0])).all()
    print("unpack (failed cell, non-monotonic axis): OK")


if __name__ == '__main__':
    test_validate()
    test_generate()
    test_unpack_with_failed_cell()
    print("\nALL PASSED")
