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
  configurable via `IVPSolver.NFEV_WARN_RATIO`). Suggests tolerance
  loosening or switching to `method='LSODA'`. Fires at most once per
  Python process (so each forked parscan worker emits at most one
  warning, not one per scan cell). Suppress entirely with
  `IVPSolver._nfev_warned = True` before model setup, or via
  `warnings.filterwarnings('ignore', category=UserWarning, module='xso.solvers')`.
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
