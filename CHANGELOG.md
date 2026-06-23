# Changelog

<!--next-version-placeholder-->

## Unreleased

### Added — Solver configuration

- `solver_kwargs` parameter on `xso.setup` and `update_setup` —
  forwards arbitrary keyword arguments to the underlying solver call.
  Enables stiff-system handling via `solver_kwargs={'method': 'LSODA'}`
  (also `'BDF'`, `'Radau'`) and per-run tolerance / step-size control
  via `solver_kwargs={'rtol': 1e-3, 'atol': 1e-12, 'max_step': 0.1}`
  on the IVP solver. The same plumbing forwards `xtol`, `maxfev`, and
  similar options to `scipy.optimize.fsolve` for `'fsolve'`,
  `'stability'`, and `'hybrid_stability'`. Each solver class defines
  its own `DEFAULT_SOLVER_KWARGS`; user-supplied values are merged on
  top key-by-key (user wins on collision). Solvers without an
  underlying scipy call (`'stepwise'`, `'deriv'`) accept but silently
  ignore the argument. When a pre-built `SolverABC` instance is passed
  alongside setup-time `solver_kwargs`, setup-time kwargs win on key
  collisions.
- Diagnostic `UserWarning` when scipy's step controller does
  disproportionately many RHS evaluations per output point on the
  IVP solver (default threshold: 100 evaluations per output point,
  configurable via `IVPSolver.NFEV_WARN_RATIO`). Recommends
  switching to `solver='stiff_ivp'` (a BDF-default bundle for
  moderately stiff systems — see entry below), or, as a one-line
  tweak on the existing solver, `solver_kwargs={'method': 'BDF',
  'rtol': 1e-4}`; alternatively suggests loosening tolerances.
  Fires at most once per Python process (so each forked parscan
  worker emits at most one warning, not one per scan cell). Suppress
  entirely with `IVPSolver._nfev_warned = True` before model setup,
  or via `warnings.filterwarnings('ignore', category=UserWarning,
  module='xso.solvers')`.
- `xso.update_setup` is now re-exported at the package top level
  (previously only accessible via `xso.xsimlabwrappers.update_setup`).
  Behaviour change: when `new_solver_kwargs` is omitted, the
  `Core__solver_kwargs` slot is now **cleared** so the new solver
  falls back to its class-level `DEFAULT_SOLVER_KWARGS`. Previously
  the slot was preserved from the old setup, which silently leaked
  solver-specific kwargs (e.g. `method='LSODA'` from `solve_ivp`)
  into incompatible solver families (e.g. `fsolve`). Pass
  `new_solver_kwargs={...}` to set kwargs explicitly across the
  update.
- Configurable IVP instability-event thresholds + diagnostic warning.
  The hardcoded safety thresholds (`< -1e-6` lower bound, `> 1e50`
  upper bound) on the terminal instability event are now exposed via
  `solver_kwargs`: `instability_neg_threshold` and
  `instability_pos_threshold` (defaults
  `IVPSolver.DEFAULT_NEG_THRESHOLD` and `IVPSolver.DEFAULT_POS_THRESHOLD`).
  These XSO-internal kwargs are popped from the merged kwargs dict
  before forwarding the remainder to `scipy.integrate.solve_ivp`.
  Pass `float('-inf')` / `float('inf')` to disable either side.
  When the event fires, a `UserWarning` now names the state variable
  that tripped it, e.g. *"Instability event triggered at t=172.4 on
  P[0] = -2.3e-6. Run terminated, remaining time points NaN-padded.
  To loosen the safety threshold pass solver_kwargs={
  'instability_neg_threshold': -1e-3} to xso.setup; to disable,
  pass float('-inf')."* Per-fire (each event firing is independently
  informative — a parscan with many cells that trip will emit one
  warning per cell). Suppress via
  `warnings.filterwarnings('ignore', category=UserWarning, module='xso.solvers')`.
- `solver='stiff_ivp'` — new built-in solver name registered in
  `xso.core._built_in_solvers`, resolving to a `StiffIVPSolver`
  subclass of `IVPSolver` with stiff-system defaults
  `{'method': 'BDF', 'rtol': 1e-4, 'atol': 1e-9}`. Inherits the
  full `IVPSolver` behaviour (instability event, post-run and
  in-run stiffness diagnostics, `trace_steps` opt-in, JSON-string
  plumbing for `Core__solver_kwargs`); overrides only the defaults
  plus a one-shot `UserWarning` when handed a non-stiff scipy
  method (`'RK45'` / `'RK23'` / `'DOP853'`). The integration still
  proceeds with the user-supplied method — the warning surfaces
  the inconsistency rather than hiding it. Empirically verified
  that LSODA's auto-stiff heuristic mis-classifies moderately
  stiff size-spectrum systems and stays in non-stiff (Adams)
  mode, where BDF reduces `nfev` by ~300× on the same regime;
  the bundle exists so the right default is one solver name
  away, not five `solver_kwargs` keys away.
- In-run stiffness alert on `IVPSolver` (inherited by
  `StiffIVPSolver`). The RHS function is wrapped with a step
  counter; every `IVPSolver.INRUN_CHECK_INTERVAL` (default
  `10_000`) calls, the projected total `nfev` is computed from
  current progress through `model.time`, and a one-shot
  `UserWarning` fires mid-integration when the projection exceeds
  `IVPSolver.INRUN_PREDICTED_NFEV_WARN` (default `1_000_000`).
  Unit-independent by construction (the time span normalises
  away). Override per-run via
  `solver_kwargs={'inrun_alert_threshold': N}`; set to
  `float('inf')` to disable. Once-per-process semantics parallel
  the existing post-run `_nfev_warned`. The pre-existing post-run
  diagnostic is preserved as a backstop — together the two cover
  *"this run will be slow"* (in-run) and *"this run was slow"*
  (post-run).
- `trace_steps` opt-in solver_kwarg on `IVPSolver`. Setting
  `solver_kwargs={'trace_steps': True}` emits one INFO record
  through `logging.getLogger('xso.solvers')` at end of run with
  the per-quartile RHS-call breakdown across the time span, e.g.
  *"trace_steps: n_calls=12345, RHS calls per quartile of
  t_span=[0, 5000]: [950, 2200, 3800, 5395] (early-to-late; cost
  concentrates early if [high, ..., low], late if [low, ...,
  high])."* Diagnoses whether the step controller's work
  concentrates in an early transient, spreads evenly, or piles up
  late (the atol-driven noise-floor pattern documented for
  matched-Type-II grazing setups). Silent by default; XSO-internal
  kwarg, popped before the remainder is forwarded to
  `scipy.integrate.solve_ivp`.
- Auto-derived Jacobian sparsity on `StiffIVPSolver`. The bundle
  now derives the Jacobian sparsity pattern from the model's
  component graph at solve time (via closure introspection of
  each `@xso.flux`-decorated callable's `flux_input_args` plus
  `model.fluxes_per_var`) and forwards it to scipy as
  `jac_sparsity`. Scipy's BDF / Radau / LSODA then use colored
  finite differences for Jacobian assembly — a 40×-ish per-Jacobian
  speedup on the matched-Type-II grazing regime over the dense FD
  scipy falls back to without a sparsity hint. No edits to
  `component.py` or `model.py`; lives entirely in `xso.solvers`,
  using a no-op `_resolve_solver_kwargs` hook added to `IVPSolver`
  that subclasses override to post-process the forwarded kwargs.
  Supported `jac_sparsity` values: `'auto'` / `True` (the default
  when the key is omitted, auto-derives from the component graph),
  `False` / `None` (drop the kwarg, scipy uses dense FD). Other
  values raise `ValueError` — precomputed sparse patterns are
  intentionally not supported on this path because `solver_kwargs`
  is JSON-encoded for zarr persistability and `scipy.sparse`
  matrices aren't JSON-serializable.

### Changed — Solver diagnostics

- Solver-internal diagnostics (initial state dumps, residual norms,
  Jacobian status, convergence outcomes, eigenvalue summaries) now
  flow through the `xso.solvers` logger instead of `print()`. Silent
  by default; enable with standard logging configuration, e.g.
  `logging.getLogger('xso.solvers').setLevel(logging.INFO)`. Single
  runs no longer require `redirect_stdout` to quiet `FSolver` /
  `NumericalStabilitySolver` / `HydridStabilitySolver` chatter, and
  the corresponding band-aid in
  `xso.parscans._run_model_scan_stability` has been removed.

### Added — Parameter scans (`xso.parscans` module)

- `run_xso_parscan` — parallel 1D and 2D parameter scans over the
  `solve_ivp` solver, driven by `multiprocessing`. Per-cell solver
  failures are NaN-filled rather than aborting the scan, with the
  output dataset preserving the scan's full shape.
- `run_xso_stabilityscan` — same shape as `run_xso_parscan` but runs
  the `NumericalStabilitySolver` per cell and attaches eigenvalue /
  stability classification metadata (`stability`,
  `max_eigenvalue`, `n_positive_eigenvalues`, `n_complex_pairs`) to
  each cell's output.
- `run_parallel_tasks` — model-agnostic streaming task pool, the sibling
  of `run_xso_parscan` for heterogeneous, self-contained tasks (each
  builds its own `xso.setup`) rather than one model swept via
  `update_vars` — for scans whose cells differ by whole arrays (e.g.
  per-era forcings) and so share no base setup. Dispatches an importable
  worker over `imap_unordered` so results stream back as they finish:
  live plain-print progress + ETA, a parent-side `on_result` hook for
  incremental save (single writer), per-task try/except → error sentinels
  (one failure never aborts the pool), `tally_flags` for a live blow-up
  count, opt-in `abort_after_errors` systemic-failure guard, and
  `pin_threads` BLAS pinning. Returns the raw per-task items
  (index-aligned); combination / scoring is left to the caller.
- `avg_tail` postprocess hook — reduces every time-dimensioned
  variable to its mean over the last N steps, for steady-state
  extraction prior to the stability scan.
- `fixed_overrides` argument — held-constant parameter overrides
  applied to every scan cell, validated against the scan axes.
- `initial_values_ds` + `iv_mapping` arguments — warm-start each
  scan cell from a prior scan output (typical pattern: IVP tail-mean
  → stability solver seeds).
- `StabilityAnalysisHook` xsimlab runtime hook for capturing
  stability metadata into the output dataset.
- Pre-flight validation: the parent process verifies that the worker
  can import the named model / setup / postprocess attributes before
  spawning workers, with suggestions on close-match names.

### Added — Additional solvers

- `FSolver` — `scipy.optimize.fsolve`-based steady-state solver.
- `DerivativeCalculator` — single `dy/dt` evaluation at the initial
  state, useful for sanity-checking flux signs and magnitudes.
- `NumericalStabilitySolver` — finds steady state via `fsolve`,
  computes Jacobian numerically, analyses eigenvalue spectrum.
- `HydridStabilitySolver` — work-in-progress SymPy + numerical
  hybrid; reverts to numerical for systems with ≥5 state variables.

### Added — Variable type features

- `xso.parameter(broadcast=True)` — exposes a parameter under a
  user-supplied label string so other components can foreign-reference
  it. Foreign references receive the value, not the label.
- `xso.parameter(foreign=True)` — reference a broadcast parameter on
  another component by its user-chosen label string. Raises a clear
  `TypeError` if a value is supplied instead of a label string.
- `xso.parameter(setup_func='<method>')` — compute the parameter
  value once at model init from other declared variables on the same
  component. Authoritative — the user cannot override the value via
  `input_vars`.
- `xso.index(as_parameter=True)` — additionally exposes the index
  values as a broadcast parameter, so other components can
  foreign-reference the size grid (or other coordinate array) as
  numeric data. Allows the Python attribute name to differ from the
  dim name.
- Flux signature filtering — fluxes need only declare the inputs they
  actually use. `**kwargs` opts back into the "receive everything"
  behaviour. Original flux functions are exposed at
  `Component.fluxes.<name>` for inspection.

### Changed

- `IVPSolver.solve()` now terminates the integration via an event
  function if the state goes non-finite or extreme (NaN, ±inf,
  `< -1e-6`, `> 1e50`). Remaining time points are NaN-padded so the
  output dataset shape is preserved.

## v0.1.0 (23/07/2023)

- First release of `xarray_simlab_ode`!
