"""Tests for the ``solver_kwargs`` plumbing on :func:`xso.setup`.

User-supplied ``solver_kwargs`` are merged on top of each solver's
class-level ``DEFAULT_SOLVER_KWARGS`` and forwarded to the underlying
scipy call (``solve_ivp`` for :class:`IVPSolver`, ``fsolve`` for the
steady-state solvers). Setup-time kwargs override kwargs baked into a
pre-built solver instance. Solvers without an underlying scipy call
(``stepwise``, ``deriv``) accept but silently ignore the argument.

The slot ``Core__solver_kwargs`` carries a JSON-encoded string at the
xsimlab level (so the input dataset can be persisted to zarr); the
encode happens inside :func:`xso.setup`, the decode inside
:meth:`Backend.initialize`. Tests that inspect the raw slot value must
decode with ``json.loads``.
"""

import json
import warnings as _warnings

import numpy as np
import pytest
import xso
from xso import solvers as _solvers


# -----------------------------------------------------------------------------
# Minimal model fixture
# -----------------------------------------------------------------------------

@xso.component
class _Linear:
    """Exponential decay: dX/dt = -rate * X."""
    var = xso.variable(description='state',
                       flux='decay', negative=True)
    rate = xso.parameter(description='decay rate')

    @xso.flux
    def decay(self, var, rate):
        return rate * var


def _build_setup(solver='solve_ivp', time=None, solver_kwargs=None):
    """Construct a tiny one-component model and matching setup."""
    model = xso.create({'X': _Linear})
    if time is None:
        time = np.arange(0.0, 2.0, 1.0)
    setup = xso.setup(
        solver=solver,
        model=model,
        time=time,
        input_vars={
            'X__var_label': 'x',
            'X__var_init': 1.0,
            'X__rate': 1.0,
        },
        solver_kwargs=solver_kwargs,
    )
    return model, setup


# -----------------------------------------------------------------------------
# Default behaviour preserved
# -----------------------------------------------------------------------------

def test_setup_without_solver_kwargs_works():
    """A setup that omits solver_kwargs runs end-to-end and behaves
    identically to the pre-feature default (RK45, rtol=1e-6, atol=1e-9)."""
    model, setup = _build_setup()
    with model:
        out = setup.xsimlab.run()
    # X starts at 1.0, decays at rate 1.0; X(1.0) = exp(-1) ≈ 0.3679.
    assert np.isclose(out['X__var'].values[-1], np.exp(-1.0), rtol=1e-4)


def test_core_slot_defaults_to_empty_json_dict():
    """When solver_kwargs is omitted, the Core__solver_kwargs slot in
    the setup dataset holds the JSON encoding of an empty dict (i.e.
    the string ``'{}'``) so that xsimlab can persist it to zarr."""
    _, setup = _build_setup()
    raw = str(setup['Core__solver_kwargs'].values.item())
    assert json.loads(raw) == {}


# -----------------------------------------------------------------------------
# Kwargs reach solve_ivp
# -----------------------------------------------------------------------------

def test_solver_kwargs_reach_solve_ivp(monkeypatch):
    """``solver_kwargs={'method': 'LSODA'}`` is observed by the actual
    solve_ivp call, and IVP defaults (rtol=1e-6, atol=1e-9) are still
    forwarded alongside it."""
    captured = {}
    real_solve_ivp = _solvers.solve_ivp

    def fake_solve_ivp(*args, **kwargs):
        captured.update(kwargs)
        return real_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(_solvers, 'solve_ivp', fake_solve_ivp)

    model, setup = _build_setup(solver_kwargs={'method': 'LSODA'})
    with model:
        setup.xsimlab.run()

    assert captured.get('method') == 'LSODA'
    assert captured.get('rtol') == 1e-6      # default preserved
    assert captured.get('atol') == 1e-9      # default preserved


def test_user_kwargs_override_defaults_per_key(monkeypatch):
    """Overriding only ``rtol`` leaves ``atol`` at the IVP default."""
    captured = {}
    real_solve_ivp = _solvers.solve_ivp   # capture before monkeypatching

    def fake_solve_ivp(*args, **kwargs):
        captured.update(kwargs)
        return real_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(_solvers, 'solve_ivp', fake_solve_ivp)

    model, setup = _build_setup(solver_kwargs={'rtol': 1e-3})
    with model:
        setup.xsimlab.run()

    assert captured.get('rtol') == 1e-3      # overridden
    assert captured.get('atol') == 1e-9      # default kept


# -----------------------------------------------------------------------------
# Instance kwargs vs setup-time kwargs (design choice 2)
# -----------------------------------------------------------------------------

def test_setup_kwargs_win_over_instance_kwargs():
    """When a pre-built SolverABC instance is passed to XSOCore along with
    setup-time solver_kwargs, setup-time kwargs override on key
    collisions while instance-only keys are preserved.

    Exercised at the XSOCore level rather than through xso.setup +
    xsimlab.run, because xsimlab persists every input variable to zarr
    and a SolverABC instance cannot be serialized there (an upstream
    limitation that predates the solver_kwargs feature). The XSOCore
    merge logic is what implements the documented "setup-time wins"
    contract, and that is what this test verifies.
    """
    from xso.core import XSOCore

    instance = _solvers.IVPSolver(
        solver_kwargs={'method': 'RK45', 'max_step': 0.5},
    )
    core = XSOCore(instance, solver_kwargs={'method': 'LSODA'})

    # The merged kwargs live on the solver instance.
    assert core.solver is instance
    assert core.solver.solver_kwargs == {
        'method': 'LSODA',   # setup-time wins on collision
        'max_step': 0.5,     # instance-only kept
    }


# -----------------------------------------------------------------------------
# Solvers without a scipy call accept but ignore kwargs
# -----------------------------------------------------------------------------

def test_stepwise_solver_ignores_kwargs():
    """Stepwise solver has no underlying scipy call. Passing
    solver_kwargs must not raise — they are stored on the instance for
    introspection and ignored at solve time."""
    model = xso.create({'X': _Linear})
    setup = xso.setup(
        solver='stepwise',
        model=model,
        time=np.arange(0.0, 5.0, 1.0),
        input_vars={
            'X__var_label': 'x',
            'X__var_init': 1.0,
            'X__rate': 1.0,
        },
        solver_kwargs={'method': 'this-key-would-crash-solve_ivp'},
    )
    with model:
        out = setup.xsimlab.run()
    # Decay over 4 steps of Δt=1, forward-Euler with rate=1 hits 0 at step 1.
    assert np.isfinite(out['X__var'].values).all()


# -----------------------------------------------------------------------------
# fsolve-side forwarding for the steady-state solver
# -----------------------------------------------------------------------------

def test_fsolve_kwargs_reach_fsolve(monkeypatch):
    """``solver='fsolve'`` with ``solver_kwargs={'xtol': ...}`` reaches
    scipy.optimize.fsolve via FSolver.solve."""
    captured = {}
    real_fsolve = _solvers.fsolve

    def fake_fsolve(*args, **kwargs):
        captured.update(kwargs)
        return real_fsolve(*args, **kwargs)

    monkeypatch.setattr(_solvers, 'fsolve', fake_fsolve)

    model = xso.create({'X': _Linear})
    setup = xso.setup(
        solver='fsolve',
        model=model,
        time=[0.0, 1.0],
        input_vars={
            'X__var_label': 'x',
            'X__var_init': 1.0,
            'X__rate': 1.0,
        },
        solver_kwargs={'xtol': 1e-10, 'maxfev': 500},
    )
    with model:
        setup.xsimlab.run()

    assert captured.get('xtol') == 1e-10
    assert captured.get('maxfev') == 500
    # full_output is set explicitly in FSolver.solve, not via kwargs.
    assert captured.get('full_output') is True


# -----------------------------------------------------------------------------
# update_setup carries new kwargs through
# -----------------------------------------------------------------------------

def _slot(setup):
    """Helper: decode the Core__solver_kwargs slot of a setup Dataset."""
    return json.loads(str(setup['Core__solver_kwargs'].values.item()))


def test_update_setup_rewrites_solver_kwargs():
    """update_setup with new_solver_kwargs rewrites the slot; omitting
    new_solver_kwargs preserves the previous value; passing an empty
    dict clears it.

    Note: new_time must be passed explicitly here. The default
    new_time=None branch of update_setup reads from old_setup.Time__time,
    a coord that only exists on output datasets (after xsimlab.run),
    not on bare setup datasets.
    """
    time = np.arange(0.0, 2.0, 1.0)
    model, setup = _build_setup(time=time, solver_kwargs={'method': 'RK45'})

    # Omitting new_solver_kwargs: previous kwargs preserved.
    setup_kept = xso.xsimlabwrappers.update_setup(
        model=model, old_setup=setup, new_solver='solve_ivp',
        new_time=time,
    )
    assert _slot(setup_kept) == {'method': 'RK45'}

    # Passing new_solver_kwargs: slot is rewritten.
    setup_new = xso.xsimlabwrappers.update_setup(
        model=model, old_setup=setup, new_solver='solve_ivp',
        new_time=time,
        new_solver_kwargs={'method': 'LSODA', 'rtol': 1e-4},
    )
    assert _slot(setup_new) == {'method': 'LSODA', 'rtol': 1e-4}

    # Passing an empty dict clears the kwargs.
    setup_cleared = xso.xsimlabwrappers.update_setup(
        model=model, old_setup=setup, new_solver='solve_ivp',
        new_time=time,
        new_solver_kwargs={},
    )
    assert _slot(setup_cleared) == {}


# -----------------------------------------------------------------------------
# Solver instance attribute is populated, even when omitted at setup
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Stiffness-diagnostic warning
# -----------------------------------------------------------------------------

def test_nfev_warning_fires_when_ratio_exceeded(monkeypatch):
    """When the step controller does more RHS evaluations per output
    point than ``IVPSolver.NFEV_WARN_RATIO``, a UserWarning is emitted
    suggesting tolerance / method adjustments via solver_kwargs."""
    # Force the next run to trip the warning regardless of how
    # well-behaved the test model is: drop the threshold to effectively
    # zero. Restore via monkeypatch teardown.
    monkeypatch.setattr(_solvers.IVPSolver, 'NFEV_WARN_RATIO', 0.0)
    # And re-arm the once-per-process flag (a previous test in this
    # session may have already tripped it).
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', False)

    model, setup = _build_setup()
    with pytest.warns(UserWarning, match="step controller is working hard"):
        with model:
            setup.xsimlab.run()


def test_nfev_warning_fires_at_most_once_per_process(monkeypatch):
    """The diagnostic warning is class-level once-per-process: after
    the first emission, subsequent runs in the same process are
    silent until ``_nfev_warned`` is reset (or the process forks)."""
    monkeypatch.setattr(_solvers.IVPSolver, 'NFEV_WARN_RATIO', 0.0)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', False)

    model, setup = _build_setup()

    # First run: the warning fires.
    with pytest.warns(UserWarning, match="step controller"):
        with model:
            setup.xsimlab.run()

    # Second run: same threshold, same conditions, but the class-level
    # flag has flipped — no stiffness warning should fire again.
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        with model:
            setup.xsimlab.run()

    stiffness_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and 'step controller' in str(w.message)
    ]
    assert stiffness_warnings == [], (
        "Stiffness warning should have fired at most once per process, "
        f"but fired again: {[str(w.message) for w in stiffness_warnings]}"
    )


def test_nfev_warning_quiet_on_well_behaved_model(monkeypatch):
    """At the default threshold, a trivially-simple model does not
    trip the stiffness warning."""
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', False)
    # Threshold left at the production default.

    model, setup = _build_setup()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        with model:
            setup.xsimlab.run()

    stiffness_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and 'step controller' in str(w.message)
    ]
    assert stiffness_warnings == [], (
        "Stiffness warning should not fire on a trivial exponential-decay "
        "model under default RK45 tolerances; got: "
        f"{[str(w.message) for w in stiffness_warnings]}"
    )


def test_solver_instance_has_solver_kwargs_attr():
    """Every built-in solver, when instantiated with no kwargs, still
    exposes a ``solver_kwargs`` attribute equal to ``{}`` — important
    for downstream code that may want to introspect it."""
    for name in ('solve_ivp', 'fsolve', 'deriv', 'stepwise', 'stability'):
        cls = _solvers.__dict__.get({
            'solve_ivp': 'IVPSolver',
            'fsolve': 'FSolver',
            'deriv': 'DerivativeCalculator',
            'stepwise': 'StepwiseSolver',
            'stability': 'NumericalStabilitySolver',
        }[name])
        inst = cls()
        assert hasattr(inst, 'solver_kwargs')
        assert inst.solver_kwargs == {}
