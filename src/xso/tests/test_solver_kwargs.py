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
import logging
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


def test_update_setup_clears_solver_kwargs_by_default():
    """update_setup clears the solver_kwargs slot when
    new_solver_kwargs is omitted, so solver-specific kwargs from the
    previous solver cannot leak into a different solver family. To
    carry kwargs across an update, the caller passes them explicitly.

    Note: new_time must be passed explicitly here. The default
    new_time=None branch of update_setup reads from old_setup.Time__time,
    a coord that only exists on output datasets (after xsimlab.run),
    not on bare setup datasets.
    """
    time = np.arange(0.0, 2.0, 1.0)
    model, setup = _build_setup(time=time, solver_kwargs={'method': 'RK45'})

    # Omitting new_solver_kwargs: slot is cleared.
    setup_default = xso.update_setup(
        model=model, old_setup=setup, new_solver='solve_ivp',
        new_time=time,
    )
    assert _slot(setup_default) == {}

    # Passing new_solver_kwargs explicitly: slot is set to that dict.
    setup_new = xso.update_setup(
        model=model, old_setup=setup, new_solver='solve_ivp',
        new_time=time,
        new_solver_kwargs={'method': 'LSODA', 'rtol': 1e-4},
    )
    assert _slot(setup_new) == {'method': 'LSODA', 'rtol': 1e-4}

    # Passing an empty dict is equivalent to the default.
    setup_explicit_empty = xso.update_setup(
        model=model, old_setup=setup, new_solver='solve_ivp',
        new_time=time,
        new_solver_kwargs={},
    )
    assert _slot(setup_explicit_empty) == {}


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


# -----------------------------------------------------------------------------
# Instability event — configurable thresholds + diagnostic warning
# -----------------------------------------------------------------------------

def test_instability_event_warning_names_offending_variable(monkeypatch):
    """When solve_ivp reports a terminal event (status=1), the diagnostic
    warning identifies the state variable that tripped it.

    Triggered via a monkeypatched solve_ivp that synthesizes the event
    arrays. Engineering a deterministic real-solver trip is awkward
    because the time pseudo-variable is part of the state and grows
    monotonically; this test isolates the warning logic from that
    complication.
    """
    real_solve_ivp = _solvers.solve_ivp

    def fake_solve_ivp(rhs, t_span, y0, t_eval=None, events=None, **kwargs):
        # Run the real solver to get a properly-shaped scipy result,
        # then patch in event data.
        result = real_solve_ivp(
            rhs, t_span, y0,
            t_eval=t_eval, events=events, **kwargs,
        )
        result.status = 1
        result.t_events = [np.array([0.5])]
        # y0 layout: [time, x] per Model variable ordering
        # (Time component is the first add_variable call).
        y_event = np.asarray(y0, dtype=float).copy()
        y_event[1] = -1e-4   # x has gone slightly negative
        result.y_events = [y_event.reshape(1, -1)]
        return result

    monkeypatch.setattr(_solvers, 'solve_ivp', fake_solve_ivp)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)  # silence stiffness

    model, setup = _build_setup()

    with pytest.warns(UserWarning) as record:
        with model:
            setup.xsimlab.run()

    event_warnings = [
        w for w in record
        if 'Instability event triggered' in str(w.message)
    ]
    assert event_warnings, (
        f"Expected an Instability-event UserWarning; got: "
        f"{[str(w.message) for w in record]}"
    )
    msg = str(event_warnings[0].message)
    assert 'x' in msg, f"Warning should name the offending variable 'x'; got: {msg}"
    assert 'instability_neg_threshold' in msg, (
        "Warning should instruct how to loosen the threshold; got: " + msg
    )


def test_instability_threshold_kwargs_stripped_before_scipy(monkeypatch):
    """``instability_neg_threshold`` and ``instability_pos_threshold`` are
    XSO-internal kwargs and must be popped from the merged kwargs dict
    before the remainder is forwarded to scipy.integrate.solve_ivp,
    which would otherwise raise ``TypeError: unexpected keyword``.
    """
    captured = {}
    real_solve_ivp = _solvers.solve_ivp

    def fake_solve_ivp(*args, **kwargs):
        captured.update(kwargs)
        return real_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(_solvers, 'solve_ivp', fake_solve_ivp)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)

    model, setup = _build_setup(solver_kwargs={
        'instability_neg_threshold': -1e-3,
        'instability_pos_threshold': 1e60,
        'rtol': 1e-4,
    })
    with model:
        setup.xsimlab.run()

    # XSO-internal kwargs are stripped before forwarding to scipy.
    assert 'instability_neg_threshold' not in captured
    assert 'instability_pos_threshold' not in captured
    # rtol is a scipy kwarg and DOES reach solve_ivp.
    assert captured.get('rtol') == 1e-4


# -----------------------------------------------------------------------------
# Logging — solver diagnostics no longer go to stdout
# -----------------------------------------------------------------------------

def test_fsolve_writes_no_stdout_by_default(capsys):
    """FSolver previously printed initial-state vectors, residuals, and a
    convergence summary directly to stdout. After the logging refactor,
    those diagnostics flow through the ``xso.solvers`` logger and are
    silent by default — no stdout output during the run."""
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
    )

    with model:
        setup.xsimlab.run()

    captured = capsys.readouterr()
    assert captured.out == '', (
        f"Expected no stdout from fsolve run; got: {captured.out!r}"
    )


def test_fsolve_emits_log_records_when_level_set(caplog):
    """When the caller configures the xso.solvers logger at INFO level,
    FSolver's high-level progress messages (steady state found, etc.)
    are captured as log records — proving the diagnostics survived the
    refactor and are merely silenced by default, not removed."""
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
    )

    with caplog.at_level(logging.INFO, logger='xso.solvers'):
        with model:
            setup.xsimlab.run()

    # At least one record from xso.solvers at INFO or above.
    solver_records = [r for r in caplog.records if r.name == 'xso.solvers']
    assert solver_records, (
        "Expected at least one xso.solvers log record at INFO level; "
        f"got records from: {sorted({r.name for r in caplog.records})}"
    )
    # The "Steady state found" message is the canonical success log line.
    assert any('Steady state found' in r.getMessage() for r in solver_records), (
        "Expected 'Steady state found' info record; got messages: "
        f"{[r.getMessage() for r in solver_records]}"
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


# -----------------------------------------------------------------------------
# Stage (a) — post-run stiffness warning text now suggests 'BDF' rather
# than 'LSODA'. Empirically (np_setups.py matched-TypeII), LSODA's
# auto-stiff heuristic stays in non-stiff (Adams) mode on moderately
# stiff systems and takes hundreds of thousands of small steps where
# BDF needs hundreds — so the warning's recommendation needs to match
# what actually helps.
# -----------------------------------------------------------------------------

def test_stiffness_warning_message_content(monkeypatch):
    """The post-run stiffness warning text recommends
    solver='stiff_ivp' (mentioning BDF in the description) and a
    loose-tolerances kwargs alternative with rtol=1e-4."""
    monkeypatch.setattr(_solvers.IVPSolver, 'NFEV_WARN_RATIO', 0.0)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', False)

    model, setup = _build_setup()
    with pytest.warns(UserWarning) as record:
        with model:
            setup.xsimlab.run()

    stiffness = [w for w in record if 'step controller' in str(w.message)]
    assert stiffness, (
        f"Expected stiffness warning; got: {[str(w.message) for w in record]}"
    )
    msg = str(stiffness[0].message)
    assert "BDF" in msg, (
        f"Stiffness warning should mention BDF in the stiff_ivp "
        f"description; got: {msg}"
    )
    assert "'rtol': 1e-4" in msg, (
        f"Stiffness warning should include rtol=1e-4 in the "
        f"loose-tolerances suggestion; got: {msg}"
    )


# -----------------------------------------------------------------------------
# Stage (a) — ``trace_steps`` opt-in: pop semantics + summary log line
# -----------------------------------------------------------------------------

def test_trace_steps_stripped_before_scipy(monkeypatch):
    """``trace_steps`` is an XSO-internal kwarg, popped before
    forwarding the remainder to scipy.integrate.solve_ivp."""
    captured = {}
    real_solve_ivp = _solvers.solve_ivp

    def fake_solve_ivp(*args, **kwargs):
        captured.update(kwargs)
        return real_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(_solvers, 'solve_ivp', fake_solve_ivp)
    # Silence the unrelated post-run stiffness warning if it would
    # otherwise fire on this micro-model under default tolerances.
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)

    model, setup = _build_setup(solver_kwargs={'trace_steps': True})
    with model:
        setup.xsimlab.run()

    assert 'trace_steps' not in captured


def test_trace_steps_logs_summary(caplog):
    """``trace_steps=True`` emits one INFO record on ``xso.solvers``
    containing the per-quartile RHS-call breakdown."""
    model, setup = _build_setup(solver_kwargs={'trace_steps': True})
    with caplog.at_level(logging.INFO, logger='xso.solvers'):
        with model:
            setup.xsimlab.run()

    trace_records = [
        r for r in caplog.records
        if r.name == 'xso.solvers' and 'trace_steps' in r.getMessage()
    ]
    assert trace_records, (
        f"Expected one xso.solvers INFO record containing 'trace_steps'; "
        f"got: {[(r.name, r.getMessage()) for r in caplog.records]}"
    )
    msg = trace_records[0].getMessage()
    assert 'n_calls=' in msg
    assert 'quartile' in msg


def test_trace_steps_silent_by_default(caplog):
    """Without ``trace_steps=True``, no step-trace INFO line is emitted."""
    model, setup = _build_setup()      # no trace_steps in solver_kwargs
    with caplog.at_level(logging.INFO, logger='xso.solvers'):
        with model:
            setup.xsimlab.run()

    trace_records = [
        r for r in caplog.records
        if r.name == 'xso.solvers' and 'trace_steps' in r.getMessage()
    ]
    assert trace_records == [], (
        f"Expected no 'trace_steps' records when flag is off; "
        f"got: {[r.getMessage() for r in trace_records]}"
    )


# -----------------------------------------------------------------------------
# Stage (a) — in-run stiffness alert: pop semantics + projection logic
# -----------------------------------------------------------------------------

def test_inrun_alert_threshold_stripped_before_scipy(monkeypatch):
    """``inrun_alert_threshold`` is an XSO-internal kwarg, popped
    before forwarding the remainder to scipy."""
    captured = {}
    real_solve_ivp = _solvers.solve_ivp

    def fake_solve_ivp(*args, **kwargs):
        captured.update(kwargs)
        return real_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(_solvers, 'solve_ivp', fake_solve_ivp)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)
    monkeypatch.setattr(_solvers.IVPSolver, '_inrun_warned', True)

    model, setup = _build_setup(
        solver_kwargs={'inrun_alert_threshold': 1e9},
    )
    with model:
        setup.xsimlab.run()

    assert 'inrun_alert_threshold' not in captured


def test_inrun_alert_fires_when_projection_exceeds_threshold(monkeypatch):
    """When the projected total nfev exceeds the threshold, the in-run
    alert fires mid-integration as a UserWarning. Class-level knobs
    lowered so the alert fires deterministically on the trivial
    exponential-decay test model."""
    # Check on every RHS call, arm immediately (no warm-up), reset the
    # once-per-process guard so the test is repeatable in a session.
    monkeypatch.setattr(_solvers.IVPSolver, 'INRUN_CHECK_INTERVAL', 1)
    monkeypatch.setattr(_solvers.IVPSolver, 'INRUN_MIN_PROGRESS_FRAC', 0.0)
    monkeypatch.setattr(_solvers.IVPSolver, '_inrun_warned', False)
    # Silence the unrelated post-run stiffness warning.
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)

    model, setup = _build_setup(
        solver_kwargs={'inrun_alert_threshold': 1.0},
    )
    with pytest.warns(UserWarning, match="at this rate"):
        with model:
            setup.xsimlab.run()


def test_inrun_alert_threshold_inf_disables(monkeypatch):
    """Setting ``inrun_alert_threshold=float('inf')`` disables the
    in-run alert even when knobs are tuned so it would otherwise fire."""
    monkeypatch.setattr(_solvers.IVPSolver, 'INRUN_CHECK_INTERVAL', 1)
    monkeypatch.setattr(_solvers.IVPSolver, 'INRUN_MIN_PROGRESS_FRAC', 0.0)
    monkeypatch.setattr(_solvers.IVPSolver, '_inrun_warned', False)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)

    model, setup = _build_setup(
        solver_kwargs={'inrun_alert_threshold': float('inf')},
    )
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        with model:
            setup.xsimlab.run()

    in_run = [w for w in caught if 'at this rate' in str(w.message)]
    assert in_run == [], (
        "In-run alert should not fire when threshold is +inf; got: "
        f"{[str(w.message) for w in in_run]}"
    )


def test_inrun_alert_silent_on_well_behaved_model(monkeypatch):
    """At the default threshold (1e6), a trivial exponential-decay
    model does not trip the in-run alert. The check interval (10_000
    calls) is much larger than this model's total nfev, so the alert's
    projection logic never runs — exactly the intended behaviour for
    well-behaved runs."""
    monkeypatch.setattr(_solvers.IVPSolver, '_inrun_warned', False)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)
    # Class-level INRUN_* constants left at production defaults.

    model, setup = _build_setup()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        with model:
            setup.xsimlab.run()

    in_run = [w for w in caught if 'at this rate' in str(w.message)]
    assert in_run == [], (
        "In-run alert should not fire on a trivial decay model at "
        f"default settings; got: {[str(w.message) for w in in_run]}"
    )


# -----------------------------------------------------------------------------
# Stage (b) — ``solver='stiff_ivp'``: a BDF-default IVPSolver subclass
# for moderately stiff dissipative systems. Defaults bundle only at
# this stage; sparse Jacobian, per-class atol, and QSS termination
# event are tracked separately and will land on this class without
# changing the user-facing API.
# -----------------------------------------------------------------------------

def test_stiff_ivp_in_registry():
    """``'stiff_ivp'`` is a built-in solver name resolving to
    :class:`StiffIVPSolver`."""
    from xso.core import _built_in_solvers
    assert 'stiff_ivp' in _built_in_solvers, (
        f"'stiff_ivp' should be registered as a built-in solver; "
        f"registry keys: {sorted(_built_in_solvers)}"
    )
    assert _built_in_solvers['stiff_ivp'] is _solvers.StiffIVPSolver


def test_stiff_ivp_defaults_to_BDF(monkeypatch):
    """``solver='stiff_ivp'`` with no solver_kwargs dispatches to
    scipy with ``method='BDF'`` and rtol=1e-4 (the bundle's defaults)."""
    captured = {}
    real_solve_ivp = _solvers.solve_ivp

    def fake_solve_ivp(*args, **kwargs):
        captured.update(kwargs)
        return real_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(_solvers, 'solve_ivp', fake_solve_ivp)
    # Silence unrelated diagnostics so they don't add noise to the
    # captured kwargs assertion.
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)

    model = xso.create({'X': _Linear})
    setup = xso.setup(
        solver='stiff_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'X__var_label': 'x',
            'X__var_init': 1.0,
            'X__rate': 1.0,
        },
    )
    with model:
        setup.xsimlab.run()

    assert captured.get('method') == 'BDF'
    assert captured.get('rtol') == 1e-4
    assert captured.get('atol') == 1e-9


def test_stiff_ivp_user_method_overrides_default(monkeypatch):
    """User-supplied ``solver_kwargs={'method': 'LSODA'}`` wins over
    the stiff_ivp bundle's BDF default — the bundle is opinionated by
    its defaults, not by enforcement."""
    captured = {}
    real_solve_ivp = _solvers.solve_ivp

    def fake_solve_ivp(*args, **kwargs):
        captured.update(kwargs)
        return real_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(_solvers, 'solve_ivp', fake_solve_ivp)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)
    monkeypatch.setattr(_solvers.StiffIVPSolver,
                        '_nonstiff_method_warned', True)

    model = xso.create({'X': _Linear})
    setup = xso.setup(
        solver='stiff_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'X__var_label': 'x',
            'X__var_init': 1.0,
            'X__rate': 1.0,
        },
        solver_kwargs={'method': 'LSODA'},
    )
    with model:
        setup.xsimlab.run()

    assert captured.get('method') == 'LSODA'      # user wins
    assert captured.get('rtol') == 1e-4           # bundle default kept
    assert captured.get('atol') == 1e-9           # bundle default kept


def test_stiff_ivp_warns_on_non_stiff_method(monkeypatch):
    """Supplying a non-stiff explicit-RK method (RK45 / RK23 / DOP853)
    on ``solver='stiff_ivp'`` emits a one-shot UserWarning explaining
    why the choice is inconsistent with the bundle's intent. The
    integration still proceeds with the user-supplied method."""
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)
    monkeypatch.setattr(_solvers.StiffIVPSolver,
                        '_nonstiff_method_warned', False)

    model = xso.create({'X': _Linear})
    setup = xso.setup(
        solver='stiff_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'X__var_label': 'x',
            'X__var_init': 1.0,
            'X__rate': 1.0,
        },
        solver_kwargs={'method': 'RK45'},
    )
    with pytest.warns(UserWarning, match="non-stiff scipy method"):
        with model:
            setup.xsimlab.run()


def test_stiff_ivp_silent_on_stiff_methods(monkeypatch):
    """Radau and LSODA are implicit / auto-stiff methods and do not
    trigger the non-stiff-method warning, even though they aren't the
    bundle's default."""
    for method in ('Radau', 'LSODA'):
        monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)
        monkeypatch.setattr(_solvers.StiffIVPSolver,
                            '_nonstiff_method_warned', False)

        model = xso.create({'X': _Linear})
        setup = xso.setup(
            solver='stiff_ivp',
            model=model,
            time=np.arange(0.0, 2.0, 1.0),
            input_vars={
                'X__var_label': 'x',
                'X__var_init': 1.0,
                'X__rate': 1.0,
            },
            solver_kwargs={'method': method},
        )
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            with model:
                setup.xsimlab.run()

        nonstiff_warnings = [
            w for w in caught
            if 'non-stiff scipy method' in str(w.message)
        ]
        assert nonstiff_warnings == [], (
            f"method={method!r} should NOT trip the non-stiff-method "
            f"warning on solver='stiff_ivp'; got: "
            f"{[str(w.message) for w in nonstiff_warnings]}"
        )


def test_post_run_stiffness_warning_mentions_stiff_ivp(monkeypatch):
    """The post-run stiffness diagnostic warning recommends
    ``solver='stiff_ivp'`` as the primary fix now that the bundle
    exists, while still preserving the ``method='BDF'`` one-line
    alternative for users who prefer not to switch solver."""
    monkeypatch.setattr(_solvers.IVPSolver, 'NFEV_WARN_RATIO', 0.0)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', False)

    model, setup = _build_setup()
    with pytest.warns(UserWarning) as record:
        with model:
            setup.xsimlab.run()

    stiffness = [w for w in record if 'step controller' in str(w.message)]
    assert stiffness, (
        f"Expected stiffness warning; got: {[str(w.message) for w in record]}"
    )
    msg = str(stiffness[0].message)
    assert "'stiff_ivp'" in msg, (
        f"Post-run stiffness warning should suggest solver='stiff_ivp'; "
        f"got: {msg}"
    )
    # BDF is mentioned in the stiff_ivp description ("BDF + auto-derived
    # sparse Jacobian"), as unquoted prose rather than a kwarg value.
    assert "BDF" in msg, f"Expected BDF still mentioned; got: {msg}"


# -----------------------------------------------------------------------------
# Stage (c) — solver='stiff_ivp' auto-derived Jacobian sparsity.
# Default behaviour: derive a sparse pattern from the component graph and hand
# it to scipy, which uses colored FD instead of dense FD when computing the
# Jacobian. Three plumbing tests + one correctness test on the trivial linear
# decay model.
# -----------------------------------------------------------------------------

def test_stiff_ivp_auto_sparsity_forwards_csr_to_scipy(monkeypatch):
    """With ``solver='stiff_ivp'`` and no user-supplied jac_sparsity,
    a scipy.sparse.csr_matrix derived from the component graph reaches
    scipy as the ``jac_sparsity`` kwarg."""
    import scipy.sparse

    captured = {}
    real_solve_ivp = _solvers.solve_ivp

    def fake_solve_ivp(*args, **kwargs):
        captured.update(kwargs)
        return real_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(_solvers, 'solve_ivp', fake_solve_ivp)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)

    model = xso.create({'X': _Linear})
    setup = xso.setup(
        solver='stiff_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'X__var_label': 'x',
            'X__var_init': 1.0,
            'X__rate': 1.0,
        },
    )
    with model:
        setup.xsimlab.run()

    sparsity = captured.get('jac_sparsity')
    assert scipy.sparse.issparse(sparsity), (
        f"Expected a scipy sparse matrix as jac_sparsity; "
        f"got: {type(sparsity).__name__}"
    )
    # The _Linear fixture produces a flat state of size 4:
    # [Time__time (var), X__var (var), Time__time_flux (flux), X__decay (flux)]
    assert sparsity.shape == (4, 4), (
        f"Expected (4, 4) sparsity for _Linear fixture; got {sparsity.shape}"
    )


def test_stiff_ivp_jac_sparsity_false_skips_kwarg(monkeypatch):
    """``solver_kwargs={'jac_sparsity': False}`` suppresses auto-derivation
    and does NOT forward jac_sparsity to scipy (which then uses dense FD)."""
    captured = {}
    real_solve_ivp = _solvers.solve_ivp

    def fake_solve_ivp(*args, **kwargs):
        captured.update(kwargs)
        return real_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(_solvers, 'solve_ivp', fake_solve_ivp)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)

    model = xso.create({'X': _Linear})
    setup = xso.setup(
        solver='stiff_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'X__var_label': 'x',
            'X__var_init': 1.0,
            'X__rate': 1.0,
        },
        solver_kwargs={'jac_sparsity': False},
    )
    with model:
        setup.xsimlab.run()

    assert 'jac_sparsity' not in captured, (
        "jac_sparsity=False should drop the kwarg before forwarding; "
        f"captured kwargs: {sorted(captured)}"
    )


def test_derive_jac_sparsity_linear_decay(monkeypatch):
    """For the trivial Linear decay fixture (dX/dt = -rate*X), the
    derived sparsity pattern over the (4, 4) flat state vector should
    have exactly two nonzero entries: the x state-var row and the
    X__decay flux row, both at the x column. The Time pseudo-variable's
    row and the Time__time_flux row are correctly all-zero (constant
    derivative)."""
    captured = {}
    real_solve_ivp = _solvers.solve_ivp

    def fake_solve_ivp(*args, **kwargs):
        captured.update(kwargs)
        return real_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(_solvers, 'solve_ivp', fake_solve_ivp)
    monkeypatch.setattr(_solvers.IVPSolver, '_nfev_warned', True)

    model = xso.create({'X': _Linear})
    setup = xso.setup(
        solver='stiff_ivp',
        model=model,
        time=np.arange(0.0, 2.0, 1.0),
        input_vars={
            'X__var_label': 'x',
            'X__var_init': 1.0,
            'X__rate': 1.0,
        },
    )
    with model:
        setup.xsimlab.run()

    sparsity = captured['jac_sparsity'].toarray()
    # Layout (full_model_dims order):
    #   index 0: Time__time          (state-var, scalar)
    #   index 1: X__var              (state-var, scalar) — user-labelled 'x'
    #   index 2: Time__time_flux     (flux, scalar)
    #   index 3: X__decay            (flux, scalar)
    #
    # Expected nonzeros (only entries where dy_i/dy_j != 0):
    #   - Row 0 (Time): time pseudo-var has constant d/dt = 1 → row all-zero.
    #   - Row 1 (x): dx/dt = -rate * x → nonzero at column 1 only.
    #   - Row 2 (Time flux): returns constant 1 → row all-zero.
    #   - Row 3 (X__decay): reads x → nonzero at column 1 only.
    expected_nonzero_pairs = {(1, 1), (3, 1)}
    actual_nonzero_pairs = {
        (int(i), int(j))
        for i, j in zip(*np.nonzero(sparsity))
    }
    assert actual_nonzero_pairs == expected_nonzero_pairs, (
        f"Sparsity pattern mismatch for linear decay model.\n"
        f"  expected nonzero (row, col) pairs: {sorted(expected_nonzero_pairs)}\n"
        f"  actual nonzero pairs:              {sorted(actual_nonzero_pairs)}\n"
        f"  full pattern:\n{sparsity.astype(int)}"
    )
