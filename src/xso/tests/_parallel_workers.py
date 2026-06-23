"""Module-level workers for ``test_run_parallel_tasks``.

These must live in an importable module (``xso.tests`` is a package) rather
than in the test file's local scope, so that spawn/forkserver pool workers can
resolve them by reference — they cannot pickle closures or ``__main__``
functions. They are deliberately trivial and deterministic.
"""


def square(x):
    """Return a small result dict; ``big`` flags ``x >= 3`` for tally_flags tests."""
    return {"x": x, "sq": x * x, "big": x >= 3}


def fail_on(x, bad=2):
    """Raise for one input, succeed otherwise — exercises pool survival."""
    if x == bad:
        raise ValueError(f"boom at {x}")
    return {"x": x, "sq": x * x}


def always_fail(x):
    """Always raise — exercises ``abort_after_errors``."""
    raise RuntimeError(f"always {x}")
