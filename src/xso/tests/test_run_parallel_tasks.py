"""Tests for ``run_parallel_tasks`` — the model-agnostic streaming task pool.

Two layers:

1. Pure unit tests of the new helpers (``_apply``, ``_validate_parallel_inputs``)
   with no multiprocessing, in the spirit of ``test_parscan_array_param.py``.
2. One real-pool integration test of the streaming / ``on_result`` / error-survival
   / abort orchestration. This intentionally breaks the suite's "avoid the Pool"
   convention because that loop is the actual new behaviour and cannot be reached
   otherwise. Workers live in ``_parallel_workers.py`` so spawn/forkserver pool
   workers import them cleanly (``xso.tests`` is a package).
"""

import pytest

from xso.parscans import _apply, _validate_parallel_inputs, run_parallel_tasks
from xso.tests import _parallel_workers as w


# --------------------------------------------------------------------------- #
# Pure unit tests — no Pool (matches the existing parscan-test convention).
# --------------------------------------------------------------------------- #
def test_apply_success():
    assert _apply((2, w.square, (5,))) == (2, True, {"x": 5, "sq": 25, "big": True})


def test_apply_catches_exception():
    index, ok, payload = _apply((7, w.always_fail, (1,)))
    assert index == 7 and ok is False
    assert "always 1" in payload          # exception repr, not a crash


def test_validate_rejects_non_callable():
    with pytest.raises(TypeError):
        _validate_parallel_inputs(42, [(1,)])


def test_validate_rejects_main_worker():
    class _MainWorker:                     # mimics a function defined in __main__
        __qualname__ = "w"
        __module__ = "__main__"

        def __call__(self, x):
            return x

    with pytest.raises(ValueError):
        _validate_parallel_inputs(_MainWorker(), [(1,)])


def test_validate_rejects_closure():
    def _make():
        def inner(x):
            return x
        return inner

    with pytest.raises(ValueError):        # __qualname__ contains '<locals>'
        _validate_parallel_inputs(_make(), [(1,)])


def test_validate_rejects_empty_tasks():
    with pytest.raises(ValueError):
        _validate_parallel_inputs(w.square, [])


def test_validate_accepts_module_level_worker():
    _validate_parallel_inputs(w.square, [(1,)])   # must not raise


# --------------------------------------------------------------------------- #
# Integration tests — real Pool (breaks the "avoid the pool" convention on
# purpose, to cover the streaming / on_result / error-survival / abort loop).
# --------------------------------------------------------------------------- #
def test_run_parallel_tasks_end_to_end(capsys):
    n = 6
    tasks = [(i,) for i in range(n)]
    seen = []

    records = run_parallel_tasks(
        w.square, tasks, processes=2,
        on_result=lambda item, done, total: seen.append(item["index"]),
        tally_flags=["big"], label="unit", pin_threads=False,
    )

    # records are index-aligned (input order), independent of completion order
    assert len(records) == n
    for i in range(n):
        assert records[i]["ok"] is True
        assert records[i]["index"] == i
        assert records[i]["result"]["sq"] == i * i

    # on_result fired exactly once per task, covering every index
    assert sorted(seen) == list(range(n))

    # tally + summary surfaced in the parent's output (big = x>=3 -> {3,4,5})
    out = capsys.readouterr().out
    assert "big=3" in out
    assert "6/6 ok" in out


def test_run_parallel_tasks_survives_one_failure(capsys):
    n = 5
    tasks = [(i,) for i in range(n)]       # fail_on raises only at x == 2
    records = run_parallel_tasks(
        w.fail_on, tasks, processes=2,
        progress=False, pin_threads=False, label="fail",
    )

    assert records[2]["ok"] is False and "boom at 2" in records[2]["error"]
    for i in (0, 1, 3, 4):
        assert records[i]["ok"] is True and records[i]["result"]["sq"] == i * i

    assert "[err]" in capsys.readouterr().out   # failure surfaced from the parent


def test_run_parallel_tasks_aborts_on_systemic_failure():
    tasks = [(i,) for i in range(8)]
    with pytest.raises(RuntimeError):
        run_parallel_tasks(
            w.always_fail, tasks, processes=2,
            abort_after_errors=2, progress=False, pin_threads=False,
        )
