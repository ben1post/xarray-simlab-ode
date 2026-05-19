import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import math

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from .model import return_dim_ndarray


# Module-level logger. Solvers emit diagnostic output through this
# logger instead of print(), so callers can control verbosity via
# Python's standard logging configuration without ad-hoc stdout
# redirection. Conventions:
#   - logger.debug:   chatty state-vector dumps, per-step diagnostics
#   - logger.info:    high-level progress (steady state found, etc.)
#   - logger.warning: convergence failures, NaN/Inf in Jacobians
#   - logger.error:   unrecoverable evaluation failures
# No handler is attached by default, so XSO is silent unless the caller
# configures logging:
#
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     logging.getLogger('xso.solvers').setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def to_ndarray(value):
    """Helper function to always have at least 1d numpy array returned."""
    if isinstance(value, list):
        return np.array(value)
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.array([value])


class SolverABC(ABC):
    """Abstract base class of backend solver class,
    use subclass to solve model within the XSO framework.

    Parameters
    ----------
    solver_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the underlying solver call
        (e.g. ``scipy.integrate.solve_ivp`` for :class:`IVPSolver`,
        ``scipy.optimize.fsolve`` for :class:`FSolver` and the stability
        solvers). Solvers that do not have an underlying scipy call
        (currently :class:`StepwiseSolver` and :class:`DerivativeCalculator`)
        accept but silently ignore the argument. Defaults to an empty
        dict.

        Each solver defines its own default kwargs (e.g. ``rtol=1e-6,
        atol=1e-9`` for :class:`IVPSolver`); user-supplied values
        override the defaults on a key-by-key basis.
    """

    def __init__(self, solver_kwargs=None):
        # Stored as a fresh dict so that mutations on the solver instance
        # do not leak back into the user's dict.
        self.solver_kwargs = dict(solver_kwargs) if solver_kwargs else {}

    @abstractmethod
    def add_variable(self, label, initial_value, model):
        """Method to reformat a variable object for use with solver,
        should return storage object of value."""
        pass

    @abstractmethod
    def add_parameter(self, label, value):
        """Method to reformat a parameter object for use with solver."""
        pass

    @abstractmethod
    def register_flux(self, label, flux, model, dims):
        """Method to reformat a flux function for use with solver,
        should return storage object of value."""
        pass

    @abstractmethod
    def add_forcing(self, label, flux, time):
        """Method to reformat a forcing function for use with solver,
        should return storage object of value."""
        pass

    @abstractmethod
    def assemble(self, model):
        """Method to initialize model."""
        pass

    @abstractmethod
    def solve(self, model, time_step):
        """Method to solve model, specific to solver chosen."""
        pass

    @abstractmethod
    def cleanup(self):
        """Method to clean up temporary storage or perform other actions
        after the model has been run."""
        pass

    class MathFunctionWrappers:
        """Default inner class providing mathematical function wrappers
        using numpy and math.

        This nested class can be modified in any implemented solver,
        if it requires specific math functions.

        Accessible within XSO components using self.m as defined in backendcomps.py class Backend.
         """

        # Constants:
        pi = np.pi  # pi constant
        e = np.e  # e constant

        @staticmethod
        def exp(x):
            """Exponential function"""
            return np.exp(x)


        @staticmethod
        def sqrt(x):
            """Square root function"""
            # add np.errstate to ignore superfluous warnings, caused by solve_ivp solver
            with np.errstate(all='ignore'):
                return np.sqrt(x)

        @staticmethod
        def log(x):
            """Logarithmic function """
            return np.log(x)

        @staticmethod
        def product(x):  # no axis?
            """Product function"""
            return math.prod(x)

        @staticmethod
        def sum(x, axis=None):
            """ Sum function"""
            return np.sum(x, axis=axis)

        @staticmethod
        def min(x1, x2):
            """ Minimum function """
            return np.minimum(x1, x2)

        @staticmethod
        def max(x1, x2):
            """ Maximum function """
            return np.maximum(x1, x2)

        @staticmethod
        def abs(x):
            """ Absolute value function """
            return np.abs(x)

        @staticmethod
        def sin(x):
            """ Sine function """
            return np.sin(x)

        @staticmethod
        def concatenate(arrays, axis=0):
            """Concatenate arrays along specified axis"""
            return np.concatenate(arrays, axis=axis)


class IVPSolver(SolverABC):
    """Solver backend using scipy.integrate.solve_ivp to solve model.

    SOLVE_IVP is a variable step-size solver for ordinary differential equations,
    included in the SciPy Python package.

    By default, it utilizes an explicit Runge-Kutta method of order 5(4).

    For stiff systems, pass ``solver_kwargs={'method': 'LSODA'}`` (or
    ``'BDF'`` / ``'Radau'``) to :func:`xso.setup`. Tolerances can be
    adjusted with e.g. ``solver_kwargs={'rtol': 1e-3, 'atol': 1e-12}``;
    any other keyword argument accepted by
    :func:`scipy.integrate.solve_ivp` may also be supplied this way.
    User-supplied kwargs are merged on top of the class-level defaults
    (:attr:`DEFAULT_SOLVER_KWARGS`).
    """

    #: Default keyword arguments forwarded to ``scipy.integrate.solve_ivp``.
    #: User-supplied ``solver_kwargs`` are merged on top of this dict
    #: (user wins on key collisions).
    DEFAULT_SOLVER_KWARGS = {'rtol': 1e-6, 'atol': 1e-9}

    #: Threshold for the stiffness-diagnostic warning: if the step
    #: controller does more than this many RHS evaluations per output
    #: point, a one-shot UserWarning suggests loosening tolerances or
    #: switching to a stiff method. Set to ``float('inf')`` to disable.
    NFEV_WARN_RATIO = 100

    #: Class-level guard ensuring the stiffness warning fires at most
    #: once per Python process. Set to ``True`` before model setup to
    #: suppress the warning entirely; set to ``False`` to re-arm.
    #: Forked parscan workers inherit ``False`` and emit their own
    #: warning (so a 30-cell scan run with 8 workers emits at most 8
    #: warnings, not 30).
    _nfev_warned = False

    #: Default lower bound on state values for the instability event.
    #: If any element of the integrated state drops below this value,
    #: the run is terminated and a UserWarning is emitted naming the
    #: offending variable. Override per-run via ``solver_kwargs[
    #: 'instability_neg_threshold']``; set to ``float('-inf')`` to
    #: disable the negative-side check entirely.
    DEFAULT_NEG_THRESHOLD = -1e-6

    #: Default upper bound on state values for the instability event.
    #: If any element exceeds this value, the run is terminated.
    #: Override per-run via ``solver_kwargs['instability_pos_threshold']``;
    #: set to ``float('inf')`` to disable the positive-side check.
    DEFAULT_POS_THRESHOLD = 1e50

    def __init__(self, solver_kwargs=None):
        super().__init__(solver_kwargs)
        self.var_init = defaultdict()
        self.flux_init = defaultdict()

    @staticmethod
    def return_dims_and_array(value, model_time):
        """Helper function to expand numpy array to appropriate size
        for odeint solver based on value and model time.
        """
        if np.size(value) == 1:
            _dims = None
            full_dims = (np.size(model_time),)
        elif len(np.shape(value)) == 1:
            _dims = np.size(value)
            full_dims = (_dims, np.size(model_time))
        else:
            _dims = np.shape(value)
            full_dims = (*_dims, np.size(model_time))

        array_out = np.zeros(full_dims)
        return array_out, _dims

    def add_variable(self, label, initial_value, model):
        """Reformats variable to comply with solver and return storage array."""

        if model.time is None:
            raise Exception("To use IVPSolver, model time needs to be supplied before adding variables")

        # store initial values of variables to pass to odeint function
        self.var_init[label] = to_ndarray(initial_value)

        array_out, dims = self.return_dims_and_array(initial_value, model.time)

        model.var_dims[label] = dims

        return array_out

    def add_parameter(self, label, value):
        """Returns parameter as numpy array."""
        return to_ndarray(value)

    def register_flux(self, label, flux, model, dims):
        """Method to reformat flux function with appropriate inputs and to proper size."""

        if model.time is None:
            raise Exception("To use ODEINT solver, model time needs to be supplied before adding fluxes")

        var_in_dict = defaultdict()
        for var, value in model.variables.items():
            var_in_dict[var] = self.var_init[var]
        for var, value in self.flux_init.items():
            var_in_dict[var] = value

        forcing_init = defaultdict()
        for key, func in model.forcing_func.items():
            forcing_init[key] = func(0)

        _flux_value = to_ndarray(flux(state=var_in_dict,
                                      parameters=model.parameters,
                                      forcings=forcing_init))
        self.flux_init[label] = _flux_value

        array_out, dims = self.return_dims_and_array(_flux_value, model.time)

        model.flux_dims[label] = dims

        return array_out

    def add_forcing(self, label, forcing_func, model):
        """Compute forcing for model time."""
        return forcing_func(model.time)

    def assemble(self, model):
        """Define full model dimensions after initialization."""

        for var_key, dim in model.var_dims.items():
            model.full_model_dims[var_key] = dim

        for flx_key, dim in model.flux_dims.items():
            model.full_model_dims[flx_key] = dim

        # TODO: conditional diagnostic print here
        # print model repr for diagnostic purposes:
        # print("Model is assembled!")
        # print(model)

    def solve(self, model, time_step):
        """Solve model using scipy.integrate.solve_ivp, passing model_function, initial values and model.time.
        The model output is then assigned to the previously initialized storage arrays within xsimlab backend.
        """
        # convert all initial values to a 1D array:
        full_init = np.concatenate([[v for val in self.var_init.values() for v in val.ravel()],
                                    [v for val in self.flux_init.values() for v in val.ravel()]], axis=None)

        # Merge class-level defaults with user-supplied solver_kwargs.
        # User-supplied keys win on collision (see SolverABC docstring).
        # Pop XSO-internal kwargs (instability event thresholds) BEFORE
        # forwarding the remainder to scipy.integrate.solve_ivp — scipy
        # would otherwise raise on the unknown kwarg names.
        kwargs = {**self.DEFAULT_SOLVER_KWARGS, **self.solver_kwargs}
        neg_threshold = kwargs.pop('instability_neg_threshold',
                                   self.DEFAULT_NEG_THRESHOLD)
        pos_threshold = kwargs.pop('instability_pos_threshold',
                                   self.DEFAULT_POS_THRESHOLD)

        def instability_event(t, y):
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                return 0  # Trigger event
            if np.any(y < neg_threshold) or np.any(y > pos_threshold):
                return 0
            return 1  # No event

        instability_event.terminal = True
        instability_event.direction = 0

        # solving model here:
        full_model_out = solve_ivp(model.model_function,
                                   t_span=[model.time[0], model.time[-1]],
                                   y0=full_init,
                                   t_eval=model.time,
                                   events=instability_event,
                                   **kwargs)

        # Stiffness-diagnostic: if the step controller did
        # disproportionately many RHS evaluations per output point, hint
        # at common causes. ``len(result.t)`` is used over
        # ``len(t_eval)`` so the ratio reflects work done in the
        # event-truncated case (where result.t is shorter than t_eval).
        # Fires at most once per Python process; each forked parscan
        # worker emits its own warning at most once.
        n_out = max(1, len(full_model_out.t))
        ratio = full_model_out.nfev / n_out
        if (ratio > IVPSolver.NFEV_WARN_RATIO
                and not IVPSolver._nfev_warned):
            IVPSolver._nfev_warned = True
            warnings.warn(
                f"solve_ivp used {full_model_out.nfev} RHS evaluations "
                f"for {n_out} output points ({ratio:.0f}/output). The "
                f"step controller is working hard — common causes are "
                f"stiff dynamics or noise-floor tracking under tight "
                f"atol. Try loosening tolerances, e.g. "
                f"solver_kwargs={{'atol': 1e-6, 'rtol': 1e-4}}, or "
                f"switch to a stiff method with "
                f"solver_kwargs={{'method': 'LSODA'}}. "
                f"(This warning fires once per process.)",
                UserWarning,
                stacklevel=2,
            )

        # Event-fire diagnostic: when scipy reports status==1 the
        # terminal instability event fired. Identify the state variable
        # that tripped it and warn (per fire, not once-per-process —
        # each parscan cell that trips it is independently informative).
        if full_model_out.status == 1:
            self._warn_on_instability_event(
                model, full_model_out, neg_threshold, pos_threshold,
            )

        # expected number of time steps:
        n_expected_timesteps = len(model.time)
        n_actual_timesteps = full_model_out.y.shape[1]

        # pad solution with NaNs if model stopped early
        if n_actual_timesteps < n_expected_timesteps:
            n_missing = n_expected_timesteps - n_actual_timesteps
            # create a nan array of same row shape, but fewer columns
            nan_padding = np.full((full_model_out.y.shape[0], n_missing), np.nan)
            padded_y = np.concatenate([full_model_out.y, nan_padding], axis=1)
            # make sure time is monotonically increasing to have coherent xarray output
            padded_y[0, :] = model.time
        else:
            padded_y = full_model_out.y  # no padding needed

        # round off to remove floating-point noise
        state_rows = [row for row in np.around(padded_y, decimals=150)]

        # round off 1e150-th decimal to remove floating point numerical errors
        # state_rows = [row for row in np.around(full_model_out.y, decimals=150)]

        # unpack and reshape state array to appropriate dimensions:
        state_dict = defaultdict()
        index = 0
        for key, dims in model.full_model_dims.items():
            if dims is None:
                state_dict[key] = state_rows[index]
                index += 1
            elif isinstance(dims, int):
                val_list = []
                for i in range(dims):
                    val_list.append(state_rows[index])
                    index += 1
                state_dict[key] = np.array(val_list)
            else:
                full_dims = (*dims, np.size(model.time))
                _length = int(np.prod(dims))
                val_list = []
                for i in range(_length):
                    val_list.append(state_rows[index])
                    index += 1

                state_dict[key] = np.array(val_list).reshape(full_dims)

        # assign solved model state to value storage in xsimlab framework:
        for var_key, val in model.variables.items():
            val[...] = state_dict[var_key]

        for flux_key, val in model.flux_values.items():
            state = state_dict[flux_key]
            dims = model.full_model_dims[flux_key]

            difference = np.diff(state) / time_step

            if dims:
                if isinstance(dims, tuple):
                    val[...] = np.concatenate((difference[..., 0][..., np.newaxis], difference), axis=len(dims))
                else:
                    val[...] = np.hstack((difference[..., 0][:, None], difference))
            else:
                val[...] = np.hstack((difference[0], difference))

    def cleanup(self):
        """Empty cleanup method, not necessary for this solver."""
        pass

    def _warn_on_instability_event(self, model, result,
                                   neg_threshold, pos_threshold):
        """Emit a UserWarning naming the state variable that tripped the
        instability event.

        Called from :meth:`solve` when ``scipy.integrate.solve_ivp``
        reports ``status==1`` (terminal event fired). Uses
        :meth:`Model.unpack_flat_state` to map the post-event flat state
        vector back to component-named pieces, then picks the most
        informative violator (NaN/inf trumps numeric; else most
        negative).

        Per-fire — does not deduplicate across runs. Each parscan cell
        that trips the event is independently informative, and users
        who find this noisy can ``warnings.filterwarnings('ignore',
        category=UserWarning, module='xso.solvers')``.
        """
        # Defensive guards: with terminal=True and status==1 these
        # should always be populated, but bail clean if scipy behaves
        # unexpectedly rather than raising on top of the original
        # diagnostic situation.
        if not result.t_events or len(result.t_events[0]) == 0:
            return
        if not result.y_events or len(result.y_events[0]) == 0:
            return

        t_event = float(result.t_events[0][0])
        y_event = np.asarray(result.y_events[0][0])

        # Demux flat state -> {component_label: scalar_or_array}.
        state = model.unpack_flat_state(y_event)

        # Collect every entry that violates the thresholds or is
        # non-finite. Each entry: (label, sub_index_or_None, value).
        violations = []
        for label, value in state.items():
            arr = np.asarray(value)
            violating = (
                ~np.isfinite(arr)
                | (arr < neg_threshold)
                | (arr > pos_threshold)
            )
            if not np.any(violating):
                continue
            if arr.ndim == 0:
                violations.append((label, None, float(arr)))
            else:
                for idx in np.argwhere(violating):
                    sub = tuple(int(i) for i in idx)
                    violations.append((label, sub, float(arr[sub])))

        if not violations:
            # Edge case: event fired but the state recorded at event
            # time shows no current violation (can happen when the
            # violating value crossed back before scipy logged the
            # event state). Generic message rather than silence.
            warnings.warn(
                f"Instability event triggered at t={t_event:.4g} "
                f"(no specific violator identified at event time). "
                f"Run terminated, remaining time points NaN-padded.",
                UserWarning,
                stacklevel=3,
            )
            return

        # Pick the most informative violator: NaN/inf trumps numeric,
        # otherwise the most-negative value (signed comparison, so
        # large negative beats large positive).
        def _severity(v):
            _, _, val = v
            if not np.isfinite(val):
                return (0, 0.0)   # NaN / inf -> highest severity
            return (1, val)       # else: sort by signed value

        worst_label, worst_sub, worst_val = min(violations, key=_severity)
        sub_str = (
            f"[{','.join(str(i) for i in worst_sub)}]"
            if worst_sub is not None else ""
        )
        others = (
            f" (plus {len(violations) - 1} other violating entries)"
            if len(violations) > 1 else ""
        )

        warnings.warn(
            f"Instability event triggered at t={t_event:.4g} on "
            f"{worst_label}{sub_str} = {worst_val:.3g}{others}. "
            f"Run terminated, remaining time points NaN-padded. "
            f"To loosen the safety threshold pass "
            f"solver_kwargs={{'instability_neg_threshold': -1e-3}} "
            f"to xso.setup; to disable, pass float('-inf').",
            UserWarning,
            stacklevel=3,
        )


class FSolver(IVPSolver):
    """Solver backend using scipy.optimize.fsolve to solve for steady states.

    Calls ``scipy.optimize.fsolve`` on the model's RHS and stores the
    converged steady-state in the final time index of every state
    variable's output array.

    Forward kwargs to ``fsolve`` via ``solver_kwargs`` on
    :func:`xso.setup`, e.g. ``solver_kwargs={'xtol': 1e-10,
    'maxfev': 5000}``. Defaults to scipy's own fsolve defaults.
    """

    #: Default keyword arguments forwarded to ``scipy.optimize.fsolve``.
    #: Empty by default; scipy's built-in fsolve defaults are used unless
    #: overridden via ``solver_kwargs`` on :func:`xso.setup`.
    DEFAULT_SOLVER_KWARGS = {}


    def solve(self, model, time_step):
        """Solve for steady state (dy/dt = 0) using scipy.optimize.fsolve."""

        # Flatten initial values into a 1D vector
        state_init = np.concatenate([[v for val in self.var_init.values() for v in val.ravel()]], axis=None)
        logger.debug("Initial state vector: %r", state_init)

        # fsolve expects a function of y -> dy/dt, so we wrap model_function properly
        def rhs_steady(y):
            full_init = np.concatenate([y, [v for val in self.flux_init.values() for v in val.ravel()]], axis=None)
            return model.model_function(time=0, current_state=full_init, only_return_state=True)

        # Merge class-level defaults with user-supplied solver_kwargs.
        # User-supplied keys win on collision (see SolverABC docstring).
        kwargs = {**self.DEFAULT_SOLVER_KWARGS, **self.solver_kwargs}

        # Solve for steady state
        y_steady, info, ier, msg = fsolve(rhs_steady, state_init,
                                          full_output=True, **kwargs)

        if ier != 1:
            logger.warning("fsolve did not converge: %s "
                           "(last iterate: %r)", msg, y_steady)
            y_steady[:] = np.nan
        else:
            logger.info("Steady state found, residual norm = %.3e",
                        np.linalg.norm(rhs_steady(y_steady)))
            logger.debug("Residuals: %r", info['fvec'])
            logger.debug("Steady state vector: %r", y_steady)

        # Broadcast steady-state solution across time steps
        n_time = len(model.time)

        state_rows = [np.array([init, final]) for init, final in zip(state_init, y_steady)]

        # Unpack only state variables
        state_dict = defaultdict()
        index = 0
        for key in model.variables.keys():
            dims = model.full_model_dims[key]
            if dims is None:
                state_dict[key] = state_rows[index]
                index += 1
            elif isinstance(dims, int):
                val_list = []
                for _ in range(dims):
                    val_list.append(state_rows[index])
                    index += 1
                state_dict[key] = np.array(val_list)
            else:
                full_dims = (*dims, n_time)
                n_flat = int(np.prod(dims))
                val_list = []
                for _ in range(n_flat):
                    val_list.append(state_rows[index])
                    index += 1
                state_dict[key] = np.array(val_list).reshape(full_dims)

        # Assign variables to model storage
        for var_key, val in model.variables.items():
            val[...] = state_dict[var_key]

        # Optionally set fluxes to zeros
        for flux_key, val in model.flux_values.items():
            dims = model.full_model_dims[flux_key]
            if dims:
                shape = (dims, n_time) if isinstance(dims, int) else (*dims, n_time)
                val[...] = np.zeros(shape)
            else:
                val[...] = np.zeros(n_time)



class DerivativeCalculator(IVPSolver):
    """
    A custom 'solver' for the XSO framework that calculates the derivatives
    of the state variables at their initial conditions.

    It mimics the FSolver structure, populating a 2-step time array where
    time 0 is the initial state and time 1 is the calculated derivative.
    """

    def solve(self, model, time_step):
        """
        Calculates the derivative of the system at the initial state.
        This method replaces the fsolve call but keeps the same
        data assignment structure.
        """

        # 1. Flatten initial values into a 1D vector
        state_init = np.concatenate([[v for val in self.var_init.values() for v in val.ravel()]], axis=None)

        # 2. Define the wrapper for the model's derivative function
        def get_derivatives(y):
            # This logic is identical to FSolver's `rhs_steady`
            full_state = np.concatenate([y, [v for val in self.flux_init.values() for v in val.ravel()]], axis=None)
            return model.model_function(time=0, current_state=full_state, only_return_state=True)

        # 3. Call the function ONCE to get the derivatives at the initial state
        derivatives = get_derivatives(state_init)

        # --- All logic below is identical to FSolver ---
        # --- (except 'y_steady' is replaced by 'derivatives') ---

        # 4. Get the number of time steps (should be 2)
        n_time = len(model.time)

        # 5. Create the state_rows [init, final] list
        # 'final' is now the derivative value
        state_rows = [np.array([init, final]) for init, final in zip(state_init, derivatives)]

        # 6. Unpack only state variables
        state_dict = defaultdict()
        index = 0
        for key in model.variables.keys():
            dims = model.full_model_dims[key]
            if dims is None:
                state_dict[key] = state_rows[index]
                index += 1
            elif isinstance(dims, int):
                val_list = []
                for _ in range(dims):
                    val_list.append(state_rows[index])
                    index += 1
                state_dict[key] = np.array(val_list)
            else:
                full_dims = (*dims, n_time)
                n_flat = int(np.prod(dims))
                val_list = []
                for _ in range(n_flat):
                    val_list.append(state_rows[index])
                    index += 1
                state_dict[key] = np.array(val_list).reshape(full_dims)

        # 7. Assign variables to model storage
        for var_key, val in model.variables.items():
            val[...] = state_dict[var_key]

        # 8. Set fluxes (identical logic to FSolver, setting all to zero)
        for flux_key, val in model.flux_values.items():
            dims = model.full_model_dims[flux_key]
            if dims:
                shape = (dims, n_time) if isinstance(dims, int) else (*dims, n_time)
                val[...] = np.zeros(shape)
            else:
                val[...] = np.zeros(n_time)



from scipy.linalg import eigvals


class OLDNumericalStabilitySolver(IVPSolver):
    """
    Solver backend for numerical steady-state and stability analysis.

    Inherits from IVPSolver to reuse initialization and setup methods,
    but implements its own solve method that:
    1. Finds steady state using fsolve (like FSolver)
    2. Numerically computes Jacobian
    3. Computes eigenvalues for stability analysis
    4. Stores results like FSolver

    .. note::

        Legacy class kept for reference; not in the public solver
        registry. Prefer :class:`NumericalStabilitySolver`.
    """

    #: Defaults forwarded to ``scipy.optimize.fsolve``.
    DEFAULT_SOLVER_KWARGS = {}

    def solve(self, model, time_step):
        """
        Solve for steady state and perform stability analysis using purely numerical methods.

        This implementation:
        1. Uses fsolve to find steady state (adapted from FSolver)
        2. Computes Jacobian numerically (adapted from FunctionalBifurcationSolver)
        3. Analyzes eigenvalues for stability
        4. Stores results in model arrays (like FSolver)
        """

        # ========================================
        # Step 1: Find steady state using fsolve (from FSolver)
        # ========================================

        # Flatten initial values into a 1D vector (EXCLUDING TIME)
        # Time is always first in var_init, so skip it
        state_init = np.concatenate([[v for key, val in self.var_init.items()
                                      if key != 'time'
                                      for v in val.ravel()]], axis=None)
        logger.info("Initial state dimension: %d", len(state_init))
        logger.debug("Initial state vector: %r", state_init)

        # fsolve expects a function of y -> dy/dt, so we wrap model_function properly
        def rhs_steady(y):
            # Prepend time=0 to state vector, then add fluxes
            y_with_time = np.concatenate([[0.0], y])
            full_init = np.concatenate([y_with_time, [v for val in self.flux_init.values() for v in val.ravel()]],
                                       axis=None)
            derivs = model.model_function(time=0, current_state=full_init, only_return_state=True)
            # Return only non-time derivatives (skip first element which is dtime/dt = 1)
            return derivs[1:]

        # Merge class-level defaults with user-supplied solver_kwargs.
        # User-supplied keys win on collision (see SolverABC docstring).
        kwargs = {**self.DEFAULT_SOLVER_KWARGS, **self.solver_kwargs}

        # Solve for steady state (from FSolver)
        y_steady, info, ier, msg = fsolve(rhs_steady, state_init,
                                          full_output=True, **kwargs)

        if ier != 1:
            residual_norm = np.linalg.norm(rhs_steady(y_steady))
            logger.warning("fsolve did not converge: %s "
                           "(residual norm = %.2e, last iterate = %r)",
                           msg, residual_norm, y_steady)
            # Set to NaN if not converged
            if residual_norm > 1e-3:  # Tolerance for accepting non-converged solution
                y_steady[:] = np.nan
                converged = False
            else:
                logger.info("Accepting non-converged solution "
                            "(residual norm = %.2e)", residual_norm)
                converged = True
        else:
            converged = True
            residual_norm = np.linalg.norm(rhs_steady(y_steady))
            logger.info("Steady state found, residual norm = %.2e",
                        residual_norm)
            logger.debug("Steady state vector: %r", y_steady)

        # ========================================
        # Step 2: Compute Jacobian numerically (from FunctionalBifurcationSolver)
        # ========================================

        if converged and not np.any(np.isnan(y_steady)):
            logger.info("Computing Jacobian numerically...")

            # Numerical Jacobian computation using finite differences
            J = self._numerical_jacobian(rhs_steady, y_steady)

            # Check for NaN/Inf in Jacobian
            if np.any(np.isnan(J)) or np.any(np.isinf(J)):
                logger.warning("Jacobian contains non-finite values: "
                               "%d NaN, %d Inf",
                               int(np.sum(np.isnan(J))),
                               int(np.sum(np.isinf(J))))
                eigvals_computed = np.full(len(y_steady), np.nan)
            else:
                # Compute eigenvalues
                try:
                    eigvals_computed = eigvals(J)
                    logger.info("Computed %d eigenvalues",
                                len(eigvals_computed))
                except Exception as e:
                    logger.warning("Eigenvalue computation failed: %s", e)
                    eigvals_computed = np.full(len(y_steady), np.nan)
        else:
            logger.warning("Skipping Jacobian computation due to "
                           "convergence failure")
            eigvals_computed = np.full(len(state_init), np.nan)

        # ========================================
        # Step 3: Store and print stability analysis
        # ========================================

        # attempt accessing variable metadata to store stability analysis results:
        # After computing eigenvalues and stability
        if converged:
            real_parts = np.real(eigvals_computed)
            imag_parts = np.imag(eigvals_computed)

            max_real = np.max(real_parts)
            min_real = np.min(real_parts)
            # Add metadata to specific variables
            # Store stability results as instance attributes
            self.stability_results = {
                'max_eigenvalue_real': float(np.max(np.real(eigvals_computed))),
                'stability': 'stable' if max_real < -1e-9 else 'unstable',
                'eigenvalues': eigvals_computed.tolist(),
                'steady_state': y_steady.tolist(),
                'converged': converged
            }

            model.Core.core.stability_metadata = {
                'max_eigenvalue_real': float(np.max(np.real(eigvals_computed))),
                'eigenvalues': eigvals_computed.tolist(),
                'stability': 'stable' if max_real < -1e-9 else 'unstable',
                'converged': converged,
                'steady_state': y_steady.tolist()
            }

        self._print_stability_analysis(y_steady, eigvals_computed, converged)

        # ========================================
        # Step 4: Store results (from FSolver)
        # ========================================

        # Get number of time steps
        n_time = len(model.time)

        # Build full state vectors with time prepended
        full_init = np.concatenate(([model.time[0]], state_init))
        full_steady = np.concatenate(([model.time[-1]], y_steady))

        # Create state rows [initial, final] for each variable
        state_rows_init = [init for init in full_init]
        state_rows_final = [final for final in full_steady]

        # Unpack state variables (including time)
        state_dict = defaultdict()
        index = 0

        for key in model.variables.keys():
            dims = model.full_model_dims[key]
            if dims is None:
                # Scalar variable
                state_dict[key] = np.array([state_rows_init[index], state_rows_final[index]])
                index += 1
            elif isinstance(dims, int):
                val_list = []
                for i in range(dims):
                    val_list.append(np.array([state_rows_init[index], state_rows_final[index]]))
                    index += 1
                state_dict[key] = np.array(val_list)
            else:
                full_dims = (*dims, n_time)
                n_flat = int(np.prod(dims))
                val_list = []
                for i in range(n_flat):
                    val_list.append(np.array([state_rows_init[index], state_rows_final[index]]))
                    index += 1
                state_dict[key] = np.array(val_list).reshape(full_dims)

        # Assign variables to model storage (from FSolver)
        for var_key, val in model.variables.items():
            val[...] = state_dict[var_key]

        # Set fluxes to zeros (from FSolver)
        for flux_key, val in model.flux_values.items():
            dims = model.full_model_dims[flux_key]
            if dims:
                shape = (dims, n_time) if isinstance(dims, int) else (*dims, n_time)
                val[...] = np.zeros(shape)
            else:
                val[...] = np.zeros(n_time)




        logger.debug("Model variables at solve exit: %r",
                     list(model.variables.items()))

    def _numerical_jacobian(self, f, y_steady, eps=1e-8):
        """
        Compute Jacobian numerically using finite differences.
        Adapted from FunctionalBifurcationSolver's _numerical_jacobian method.

        Parameters
        ----------
        f : callable
            Function that returns dy/dt given state y
        y_steady : array
            State at which to compute Jacobian
        eps : float
            Step size for finite differences

        Returns
        -------
        J : ndarray
            Jacobian matrix
        """
        n = len(y_steady)
        J = np.zeros((n, n))

        # Evaluate function at steady state
        f0 = f(y_steady)

        for j in range(n):
            y_plus = y_steady.copy()
            y_minus = y_steady.copy()

            # Adaptive step size based on magnitude (from FunctionalBifurcationSolver)
            h = eps * max(abs(y_steady[j]), 1.0)

            y_plus[j] += h
            y_minus[j] -= h

            f_plus = f(y_plus)
            f_minus = f(y_minus)

            # Central difference approximation
            J[:, j] = (f_plus - f_minus) / (2 * h)

        return J

    def _print_stability_analysis(self, y_steady, eigvals, converged):
        """
        Print steady state and stability analysis results.
        Adapted from FunctionalBifurcationSolver's _print_stability_analysis.

        Parameters
        ----------
        y_steady : array
            Steady state solution
        eigvals : array
            Eigenvalues of Jacobian at steady state
        converged : bool
            Whether steady state search converged
        """
        lines = [
            "Numerical stability analysis results",
            f"  Steady state: {y_steady}",
            f"  Convergence:  {'Success' if converged else 'Failed'}",
        ]

        if converged and len(eigvals) > 0 and np.all(np.isfinite(eigvals)):
            real_parts = np.real(eigvals)
            imag_parts = np.imag(eigvals)

            max_real = np.max(real_parts)
            min_real = np.min(real_parts)

            # Determine stability (from FunctionalBifurcationSolver)
            if max_real < -1e-9:
                stability = "STABLE"
            elif max_real > 1e-9:
                stability = "UNSTABLE"
            else:
                stability = "MARGINALLY STABLE"

            n_positive = int(np.sum(real_parts > 1e-9))
            n_negative = int(np.sum(real_parts < -1e-9))
            n_zero = int(np.sum(np.abs(real_parts) <= 1e-9))
            n_complex = int(np.sum(np.abs(imag_parts) > 1e-10))

            lines.extend([
                f"  Eigenvalues:  {len(eigvals)} computed; "
                f"max Re = {max_real:.4e}, min Re = {min_real:.4e}",
                f"  Stability:    {stability} (max real part = {max_real:.4e})",
                f"  Counts:       {n_positive} positive, "
                f"{n_negative} negative, {n_zero} near-zero real parts",
            ])
            if n_complex > 0:
                lines.append(
                    f"  Oscillation:  {n_complex // 2} complex pairs, "
                    f"max |Im| = {np.max(np.abs(imag_parts)):.4f}"
                )

            # First few eigenvalues for inspection.
            n_show = min(5, len(eigvals))
            eig_lines = []
            for i in range(n_show):
                if abs(imag_parts[i]) > 1e-10:
                    eig_lines.append(
                        f"    λ_{i + 1} = {real_parts[i]:.4e} "
                        f"± {abs(imag_parts[i]):.4e}i"
                    )
                else:
                    eig_lines.append(
                        f"    λ_{i + 1} = {real_parts[i]:.4e}"
                    )
            lines.append(f"  First {n_show} eigenvalues:\n"
                         + "\n".join(eig_lines))
        else:
            lines.append(
                f"  Eigenvalues:  could not compute (converged={converged})"
            )

        logger.info("\n".join(lines))


class NumericalStabilitySolver(IVPSolver):
    """
    Solver backend for numerical steady-state and stability analysis.

    Inherits from IVPSolver to reuse initialization and setup methods,
    but implements its own solve method that:
    1. Finds steady state using fsolve (like FSolver)
    2. Numerically computes Jacobian
    3. Computes eigenvalues for stability analysis
    4. Stores results like FSolver

    Forward kwargs to the inner ``fsolve`` call via ``solver_kwargs`` on
    :func:`xso.setup`.
    """

    #: Default keyword arguments forwarded to ``scipy.optimize.fsolve``.
    #: Empty by default; scipy's built-in fsolve defaults are used unless
    #: overridden via ``solver_kwargs`` on :func:`xso.setup`.
    DEFAULT_SOLVER_KWARGS = {}

    def solve(self, model, time_step):
        """
        Solve for steady state and perform stability analysis using purely numerical methods.

        This implementation:
        1. Uses fsolve to find steady state (adapted from FSolver)
        2. Computes Jacobian numerically (adapted from FunctionalBifurcationSolver)
        3. Analyzes eigenvalues for stability
        4. Stores results in model arrays (like FSolver)
        """

        # ========================================
        # Step 1: Find steady state using fsolve (from FSolver)
        # ========================================

        # Flatten initial values into a 1D vector (EXCLUDING TIME)
        # Time is always first in var_init, so skip it
        state_init = np.concatenate([[v for key, val in self.var_init.items()
                                      if key != 'time'
                                      for v in val.ravel()]], axis=None)
        logger.info("Initial state dimension: %d", len(state_init))
        logger.debug("Initial state vector: %r", state_init)

        # fsolve expects a function of y -> dy/dt, so we wrap model_function properly
        def rhs_steady(y):
            # Prepend time=0 to state vector, then add fluxes
            y_with_time = np.concatenate([[0.0], y])
            full_init = np.concatenate([y_with_time, [v for val in self.flux_init.values() for v in val.ravel()]],
                                       axis=None)
            derivs = model.model_function(time=0, current_state=full_init, only_return_state=True)
            # Return only non-time derivatives (skip first element which is dtime/dt = 1)
            return derivs[1:]

        # Merge class-level defaults with user-supplied solver_kwargs.
        # User-supplied keys win on collision (see SolverABC docstring).
        kwargs = {**self.DEFAULT_SOLVER_KWARGS, **self.solver_kwargs}

        # Solve for steady state (from FSolver)
        y_steady, info, ier, msg = fsolve(rhs_steady, state_init,
                                          full_output=True, **kwargs)

        if ier != 1:
            residual_norm = np.linalg.norm(rhs_steady(y_steady))
            logger.warning("fsolve did not converge: %s "
                           "(residual norm = %.2e, last iterate = %r)",
                           msg, residual_norm, y_steady)
            # Set to NaN if not converged
            if residual_norm > 1e-3:  # Tolerance for accepting non-converged solution
                y_steady[:] = np.nan
                converged = False
            else:
                logger.info("Accepting non-converged solution "
                            "(residual norm = %.2e)", residual_norm)
                converged = True
        else:
            converged = True
            residual_norm = np.linalg.norm(rhs_steady(y_steady))
            logger.info("Steady state found, residual norm = %.2e",
                        residual_norm)
            logger.debug("Steady state vector: %r", y_steady)

        # ========================================
        # Step 2: Compute Jacobian numerically (from FunctionalBifurcationSolver)
        # ========================================

        if converged and not np.any(np.isnan(y_steady)):
            logger.info("Computing Jacobian numerically...")

            # Numerical Jacobian computation using finite differences
            J = self._numerical_jacobian(rhs_steady, y_steady)

            # Check for NaN/Inf in Jacobian
            if np.any(np.isnan(J)) or np.any(np.isinf(J)):
                logger.warning("Jacobian contains non-finite values: "
                               "%d NaN, %d Inf",
                               int(np.sum(np.isnan(J))),
                               int(np.sum(np.isinf(J))))
                eigvals_computed = np.full(len(y_steady), np.nan)
            else:
                # Compute eigenvalues
                try:
                    eigvals_computed = eigvals(J)
                    logger.info("Computed %d eigenvalues",
                                len(eigvals_computed))
                except Exception as e:
                    logger.warning("Eigenvalue computation failed: %s", e)
                    eigvals_computed = np.full(len(y_steady), np.nan)
        else:
            logger.warning("Skipping Jacobian computation due to "
                           "convergence failure")
            eigvals_computed = np.full(len(state_init), np.nan)

        # ========================================
        # Step 3: Print stability analysis (from FunctionalBifurcationSolver)
        # ========================================

        # Store stability results for runtime hook access
        self.stability_results = self._compute_stability_results(y_steady, eigvals_computed, converged)

        #self._print_stability_analysis(y_steady, eigvals_computed, converged)

        # ========================================
        # Step 4: Store results (from FSolver)
        # ========================================

        # Get number of time steps
        n_time = len(model.time)

        # Build full state vectors with time prepended
        full_init = np.concatenate(([model.time[0]], state_init))
        full_steady = np.concatenate(([model.time[-1]], y_steady))

        # Create state rows [initial, final] for each variable
        state_rows_init = [init for init in full_init]
        state_rows_final = [final for final in full_steady]

        # Unpack state variables (including time)
        state_dict = defaultdict()
        index = 0

        for key in model.variables.keys():
            dims = model.full_model_dims[key]
            if dims is None:
                # Scalar variable
                state_dict[key] = np.array([state_rows_init[index], state_rows_final[index]])
                index += 1
            elif isinstance(dims, int):
                val_list = []
                for i in range(dims):
                    val_list.append(np.array([state_rows_init[index], state_rows_final[index]]))
                    index += 1
                state_dict[key] = np.array(val_list)
            else:
                full_dims = (*dims, n_time)
                n_flat = int(np.prod(dims))
                val_list = []
                for i in range(n_flat):
                    val_list.append(np.array([state_rows_init[index], state_rows_final[index]]))
                    index += 1
                state_dict[key] = np.array(val_list).reshape(full_dims)

        # Assign variables to model storage (from FSolver)
        for var_key, val in model.variables.items():
            val[...] = state_dict[var_key]

        # Build state dict at steady state (mirrors Model.unpack_flat_state)
        full_ss = np.concatenate([
            [model.time[-1]],
            y_steady,
            [v for val in self.flux_init.values() for v in val.ravel()]
        ])
        state_ss = model.unpack_flat_state(full_ss)
        forcing_ss = {k: f(model.time[-1]) for k, f in model.forcing_func.items()}

        # Evaluate each flux at steady state, same order/update logic as model_function
        flux_ss = {}
        for flx_label, flux in model.fluxes.items():
            val = return_dim_ndarray(flux(state=state_ss,
                                          parameters=model.parameters,
                                          forcings=forcing_ss))
            flux_ss[flx_label] = val
            if flx_label in state_ss:
                state_ss[flx_label] = val

        # Write into flux storage, broadcast over time
        for flux_key, storage in model.flux_values.items():
            ss_val = flux_ss[flux_key]
            dims = model.full_model_dims[flux_key]
            if dims:
                shape = (dims, n_time) if isinstance(dims, int) else (*dims, n_time)
                storage[...] = np.broadcast_to(ss_val[..., None], shape)
            else:
                storage[...] = np.broadcast_to(np.asarray(ss_val).ravel(), (n_time,))

    def _numerical_jacobian(self, f, y_steady, eps=1e-8):
        """
        Compute Jacobian numerically using finite differences.
        Adapted from FunctionalBifurcationSolver's _numerical_jacobian method.

        Parameters
        ----------
        f : callable
            Function that returns dy/dt given state y
        y_steady : array
            State at which to compute Jacobian
        eps : float
            Step size for finite differences

        Returns
        -------
        J : ndarray
            Jacobian matrix
        """
        n = len(y_steady)
        J = np.zeros((n, n))

        # Evaluate function at steady state
        f0 = f(y_steady)

        for j in range(n):
            y_plus = y_steady.copy()
            y_minus = y_steady.copy()

            # Adaptive step size based on magnitude (from FunctionalBifurcationSolver)
            h = eps * max(abs(y_steady[j]), 1.0)

            y_plus[j] += h
            y_minus[j] -= h

            f_plus = f(y_plus)
            f_minus = f(y_minus)

            # Central difference approximation
            J[:, j] = (f_plus - f_minus) / (2 * h)

        return J

    def _print_stability_analysis(self, y_steady, eigvals, converged):
        """
        Print steady state and stability analysis results.
        Adapted from FunctionalBifurcationSolver's _print_stability_analysis.

        Parameters
        ----------
        y_steady : array
            Steady state solution
        eigvals : array
            Eigenvalues of Jacobian at steady state
        converged : bool
            Whether steady state search converged
        """
        lines = [
            "Numerical stability analysis results",
            f"  Steady state: {y_steady}",
            f"  Convergence:  {'Success' if converged else 'Failed'}",
        ]

        if converged and len(eigvals) > 0 and np.all(np.isfinite(eigvals)):
            real_parts = np.real(eigvals)
            imag_parts = np.imag(eigvals)

            max_real = np.max(real_parts)
            min_real = np.min(real_parts)

            # Determine stability (from FunctionalBifurcationSolver)
            if max_real < -1e-9:
                stability = "STABLE"
            elif max_real > 1e-9:
                stability = "UNSTABLE"
            else:
                stability = "MARGINALLY STABLE"

            n_positive = int(np.sum(real_parts > 1e-9))
            n_negative = int(np.sum(real_parts < -1e-9))
            n_zero = int(np.sum(np.abs(real_parts) <= 1e-9))
            n_complex = int(np.sum(np.abs(imag_parts) > 1e-10))

            lines.extend([
                f"  Eigenvalues:  {len(eigvals)} computed; "
                f"max Re = {max_real:.4e}, min Re = {min_real:.4e}",
                f"  Stability:    {stability} (max real part = {max_real:.4e})",
                f"  Counts:       {n_positive} positive, "
                f"{n_negative} negative, {n_zero} near-zero real parts",
            ])
            if n_complex > 0:
                lines.append(
                    f"  Oscillation:  {n_complex // 2} complex pairs, "
                    f"max |Im| = {np.max(np.abs(imag_parts)):.4f}"
                )

            # First few eigenvalues for inspection.
            n_show = min(5, len(eigvals))
            eig_lines = []
            for i in range(n_show):
                if abs(imag_parts[i]) > 1e-10:
                    eig_lines.append(
                        f"    λ_{i + 1} = {real_parts[i]:.4e} "
                        f"± {abs(imag_parts[i]):.4e}i"
                    )
                else:
                    eig_lines.append(
                        f"    λ_{i + 1} = {real_parts[i]:.4e}"
                    )
            lines.append(f"  First {n_show} eigenvalues:\n"
                         + "\n".join(eig_lines))
        else:
            lines.append(
                f"  Eigenvalues:  could not compute (converged={converged})"
            )

        logger.info("\n".join(lines))

    def _compute_stability_results(self, y_steady, eigvals, converged):
        """
        Compute stability results dictionary for storage and retrieval.

        Parameters
        ----------
        y_steady : array
            Steady state solution
        eigvals : array
            Eigenvalues of Jacobian at steady state
        converged : bool
            Whether steady state search converged

        Returns
        -------
        dict
            Dictionary containing stability analysis results
        """
        results = {
            'converged': converged,
            'steady_state': y_steady.tolist() if not np.any(np.isnan(y_steady)) else None,
        }

        if converged and len(eigvals) > 0 and np.all(np.isfinite(eigvals)):
            real_parts = np.real(eigvals)
            imag_parts = np.imag(eigvals)
            max_real = np.max(real_parts)

            # Determine stability
            if max_real < -1e-6:
                stability = 'stable'
            elif max_real > 1e-6:
                stability = 'unstable'
            else:
                stability = 'marginal'

            results.update({
                'stability': stability,
                'max_eigenvalue_real': float(max_real),
                'min_eigenvalue_real': float(np.min(real_parts)),
                'eigenvalues_real': real_parts.tolist(),
                'eigenvalues_imag': imag_parts.tolist(),
                'n_positive_eigenvalues': int(np.sum(real_parts > 1e-9)),
                'n_negative_eigenvalues': int(np.sum(real_parts < -1e-9)),
                'n_complex_pairs': int(np.sum(np.abs(imag_parts) > 1e-10) // 2),
            })
        else:
            results['stability'] = 'unknown'

        return results


class StepwiseSolver(SolverABC):
    """Solver that can handle stepwise calculation built into xsimlab framework.

    Model output is computed step by step and assigned to the appropriate
    storage arrays in xsimlab backend.

    This solver has no underlying scipy call, so ``solver_kwargs`` on
    :func:`xso.setup` is accepted but **silently ignored**. The argument
    is recorded on the instance for introspection.
    """

    #: No underlying scipy call exists for this solver; kwargs are
    #: stored on the instance but not consumed.
    DEFAULT_SOLVER_KWARGS = {}

    def __init__(self, solver_kwargs=None):
        super().__init__(solver_kwargs)
        self.model_time = 0
        self.time_index = 0

        self.full_model_values = defaultdict()

    @staticmethod
    def return_dims_and_array(value, model_time):
        """Helper function to create arrays of appropriate size,
        and assign initial value(s) to first index
        """

        if np.size(value) == 1:
            _dims = None
            full_dims = (np.size(model_time),)
            array_out = np.zeros(full_dims)
            array_out[0] = np.asarray(value).item()
        elif len(np.shape(value)) == 1:
            _dims = np.size(value)
            full_dims = (_dims, np.size(model_time))
            array_out = np.zeros(full_dims)
            array_out[:, 0] = value
        else:
            _dims = np.shape(value)
            full_dims = (*_dims, np.size(model_time))
            array_out = np.zeros(full_dims)
            array_out[..., 0] = value

        return array_out, _dims

    def add_variable(self, label, initial_value, model):
        """Method to reformat variable and return storage array."""
        array_out, _dims = self.return_dims_and_array(initial_value, model.time)
        model.var_dims[label] = _dims
        return array_out

    def add_parameter(self, label, value):
        """Method to reformat parameter and return array."""
        return to_ndarray(value)

    def register_flux(self, label, flux, model, dims):
        """Method to reformat flux function with appropriate inputs and to proper size."""

        var_in_dict = defaultdict()
        for var_key, value in model.variables.items():
            _dims = model.var_dims[var_key]
            if _dims is None:
                var_in_dict[var_key] = value[0]
            elif isinstance(_dims, int):
                var_in_dict[var_key] = value[:, 0]
            else:
                var_in_dict[var_key] = value[..., 0]

        for flx_key, value in model.flux_values.items():
            _dims = model.flux_dims[flx_key]
            if _dims is None:
                var_in_dict[flx_key] = value[0]
            elif isinstance(_dims, int):
                var_in_dict[flx_key] = value[:, 0]
            else:
                var_in_dict[flx_key] = value[..., 0]

        forcing_now = defaultdict()
        for key, func in model.forcing_func.items():
            forcing_now[key] = func(0)

        flux_init = to_ndarray(flux(state=var_in_dict,
                                    parameters=model.parameters,
                                    forcings=forcing_now))

        array_out, _dims = self.return_dims_and_array(flux_init, model.time)

        model.flux_dims[label] = _dims

        return array_out

    def add_forcing(self, label, forcing_func, model):
        """Compute forcing over model time and provide as array."""
        return forcing_func(model.time)

    def assemble(self, model):
        """Assemble full model dimension and order fluxes for proper unpacking."""
        for var_key, value in model.variables.items():
            _dims = model.var_dims[var_key]
            model.full_model_dims[var_key] = _dims
            self.full_model_values[var_key] = value

        for flx_key, value in model.flux_values.items():
            _dims = model.flux_dims[flx_key]
            model.full_model_dims[flx_key] = _dims
            self.full_model_values[flx_key] = value

        # TODO: diagnostic print here
        # finally print model repr for diagnostic purposes:
        # print("Model is assembled:")
        # print(model)

    def solve(self, model, time_step):
        """Solve model in a stepwise fashion, calling this function at each time step."""

        self.model_time += time_step
        self.time_index += 1

        model_forcing = defaultdict()
        for key, func in model.forcing_func.items():
            # retrieve pre-computed forcing:
            model_forcing[key] = model.forcings[key][self.time_index]

        model_state = []
        for key, val in self.full_model_values.items():
            if model.full_model_dims[key]:
                model_state.append(val[..., self.time_index - 1])
            else:
                model_state.append(val[self.time_index - 1])

        # flatten list for model function:
        flat_model_state = np.concatenate(model_state, axis=None)

        state_out = model.model_function(current_state=flat_model_state, forcing=model_forcing)

        # unpack flat state:
        state_dict = model.unpack_flat_state(state_out)

        for key, val in model.variables.items():
            if model.full_model_dims[key]:
                val[..., self.time_index] = val[..., self.time_index - 1] + state_dict[key] * time_step
            else:
                val[self.time_index] = np.asarray(val[self.time_index - 1] + state_dict[key] * time_step).item()

        for key, val in model.flux_values.items():
            if model.full_model_dims[key]:
                val[..., self.time_index] = state_dict[key]
            else:
                val[self.time_index] = state_dict[key]

    def cleanup(self):
        """Empty cleanup method, not necessary for this solver."""
        pass




import sympy
import numpy.linalg as LA # For eigenvalues


class HydridStabilitySolver(SolverABC):
    """
    WORK IN PROGRESS: sympy not fully integrated, so analytical treatment of complex models is faulty,
    but it reverts to numerical solution for more than 4 state variables.
    (Then NumericalStabilitySolver should be more efficient)

    Solver backend using SymPy for steady-state and stability analysis.

    Finds equilibrium points, calculates Jacobian matrix, and computes eigenvalues
    for stability analysis. Prints analysis to console and saves steady states.

    Forward kwargs to the inner ``fsolve`` calls via ``solver_kwargs`` on
    :func:`xso.setup`.
    """

    #: Default keyword arguments forwarded to ``scipy.optimize.fsolve``.
    #: ``xtol`` and ``maxfev`` are set tighter than scipy's own defaults
    #: because the SymPy-derived RHS can be sensitive; override either
    #: via ``solver_kwargs`` on :func:`xso.setup` as needed.
    DEFAULT_SOLVER_KWARGS = {'xtol': 1e-8, 'maxfev': 10000}

    def __init__(self, solver_kwargs=None):
        super().__init__(solver_kwargs)

        # Symbolic representations
        self.symbolic_vars = {}
        self.symbolic_params = {}
        self.symbolic_fluxes = {}
        self.symbolic_forcings = {}

        # Ordering and values
        self.var_names_ordered = []
        self.sym_params_ordered = []
        self.var_init_flat = []
        self.param_values_map = {}

        # Lambdified functions
        self.f_func = None
        self.jac_func = None


    class MathFunctionWrappers:
        """
        Math function wrappers for symbolic (SymPy) operations.

        Used during symbolic model assembly. All functions assume SymPy input.
        """

        # Constants
        pi = sympy.pi
        e = sympy.E

        @staticmethod
        def exp(x):
            """Exponential function"""
            return sympy.exp(x)

        @staticmethod
        def sqrt(x):
            """Square root function"""
            return sympy.sqrt(x)

        @staticmethod
        def log(x):
            """Natural logarithm"""
            return sympy.log(x)

        @staticmethod
        def product(x):
            """Product of array elements"""
            x_array = np.asarray(x)
            result = sympy.Integer(1)
            for elem in x_array.flat:
                result *= elem
            return result

        @staticmethod
        def sum(x, axis=None):
            """Sum of array elements"""
            x_array = np.asarray(x)
            if axis is None:
                return sympy.Add(*x_array.flat)
            else:
                # Sum along specified axis - preserve array structure
                if x_array.ndim == 2:
                    if axis == 0:
                        # Sum down columns
                        return np.array([sympy.Add(*x_array[:, j]) for j in range(x_array.shape[1])])
                    elif axis == 1:
                        # Sum across rows
                        return np.array([sympy.Add(*x_array[i, :]) for i in range(x_array.shape[0])])
                elif x_array.ndim == 1:
                    return sympy.Add(*x_array)
                else:
                    # For higher dimensions, use apply_along_axis
                    return np.apply_along_axis(lambda arr: sympy.Add(*arr), axis, x_array)

        @staticmethod
        def min(x1, x2):
            """Minimum function"""
            return sympy.Min(x1, x2)

        @staticmethod
        def max(x1, x2):
            """Maximum function"""
            return sympy.Max(x1, x2)

        @staticmethod
        def abs(x):
            """Absolute value function"""
            return sympy.Abs(x)

        @staticmethod
        def sin(x):
            """Sine function"""
            return sympy.sin(x)

        @staticmethod
        def concatenate(arrays, axis=0):
            """
            Concatenate symbolic arrays.

            Combines SymPy symbolic expressions into a single numpy array.
            """
            if not isinstance(arrays, (list, tuple)):
                arrays = (arrays,)

            result = []
            for arr in arrays:
                arr_array = np.asarray(arr)
                if arr_array.ndim == 0:
                    # Scalar
                    result.append(arr_array.item())
                else:
                    # Array - flatten
                    result.extend(arr_array.flat)

            return np.array(result)

    @staticmethod
    def return_dims_and_array(value, model_time):
        """Create output array with appropriate dimensions."""
        value_array = to_ndarray(value)

        if value_array.size == 1:
            dims = None
            full_dims = (len(model_time),)
        elif value_array.ndim == 1:
            dims = value_array.size
            full_dims = (dims, len(model_time))
        else:
            dims = value_array.shape
            full_dims = (*dims, len(model_time))

        return np.zeros(full_dims), dims

    def add_variable(self, label, initial_value, model):
        """Register a state variable with symbolic representation.

        For dimensional variables (arrays), creates indexed symbolic variables
        (e.g., var_0, var_1, var_2) to properly handle vectorization.
        """
        # Time always goes first, others append in order
        if label == 'time':
            self.var_names_ordered.insert(0, label)
            # Time is scalar
            self.symbolic_vars[label] = sympy.symbols(label)
        else:
            self.var_names_ordered.append(label)

            # Create symbolic variable(s)
            init_array = to_ndarray(initial_value)

            if init_array.size == 1:
                # Scalar variable
                self.symbolic_vars[label] = sympy.symbols(label)
                self.var_init_flat.extend(init_array.ravel())
            else:
                # Array variable: create indexed symbolic variables
                var_symbols = sympy.symbols(f'{label}_:{init_array.size}')
                # Convert tuple to numpy array and reshape to original shape
                if isinstance(var_symbols, tuple):
                    var_symbols = np.array(var_symbols).reshape(init_array.shape)
                self.symbolic_vars[label] = var_symbols
                self.var_init_flat.extend(init_array.ravel())

        # Create output array
        array_out, dims = self.return_dims_and_array(initial_value, model.time)
        model.var_dims[label] = dims

        return array_out

    def add_parameter(self, label, value):
        """Register a parameter with symbolic representation.

        For dimensional parameters (arrays), creates indexed symbolic variables
        (e.g., param_0, param_1, param_2) to properly handle vectorization.
        """
        value_array = to_ndarray(value)

        if label not in self.symbolic_params:
            if value_array.size == 1:
                # Scalar parameter: single symbolic variable
                self.symbolic_params[label] = sympy.symbols(label)
                self.param_values_map[label] = float(value_array.item())
            else:
                # Array parameter: create indexed symbolic variables
                param_symbols = sympy.symbols(f'{label}_:{value_array.size}')
                # Convert tuple to numpy array and reshape to original shape
                if isinstance(param_symbols, tuple):
                    param_symbols = np.array(param_symbols).reshape(value_array.shape)
                self.symbolic_params[label] = param_symbols

                # Store each value with its index
                for i, val in enumerate(value_array.flat):
                    self.param_values_map[f'{label}_{i}'] = float(val)

        return value_array

    def add_forcing(self, label, forcing_func, model):
        """Register a time-dependent forcing function.

        For dimensional forcings (arrays), creates indexed symbolic variables.
        """
        forcing_val = forcing_func(0)
        forcing_array = to_ndarray(forcing_val)

        if label not in self.symbolic_forcings:
            if forcing_array.size == 1:
                # Scalar forcing
                self.symbolic_forcings[label] = sympy.symbols(label)
                self.param_values_map[label] = float(forcing_array.item())
            else:
                # Array forcing: create indexed symbolic variables
                forcing_symbols = sympy.symbols(f'{label}_:{forcing_array.size}')
                # Convert tuple to numpy array and reshape to original shape
                if isinstance(forcing_symbols, tuple):
                    forcing_symbols = np.array(forcing_symbols).reshape(forcing_array.shape)
                self.symbolic_forcings[label] = forcing_symbols

                # Store each value with its index
                for i, val in enumerate(forcing_array.flat):
                    self.param_values_map[f'{label}_{i}'] = float(val)

        return forcing_func(model.time)

    def _get_state_at_time(self, model, t_index=0):
        """Extract state variables at specific time index."""
        state = {}

        # Variables
        for var_key, value in model.variables.items():
            if var_key not in model.var_dims:
                continue
            dims = model.var_dims[var_key]

            if dims is None:
                state[var_key] = value[t_index]
            elif isinstance(dims, int):
                state[var_key] = value[:, t_index]
            else:
                state[var_key] = value[..., t_index]

        # Fluxes
        for flux_key, value in model.flux_values.items():
            if flux_key in model.flux_dims:
                dims = model.flux_dims[flux_key]

                if dims is None:
                    state[flux_key] = value[t_index]
                elif isinstance(dims, int):
                    state[flux_key] = value[:, t_index]
                else:
                    state[flux_key] = value[..., t_index]

        return state

    def register_flux(self, label, flux, model, dims):
        """
        Register a flux function with both symbolic and numerical evaluation.

        Calls flux twice:
        1. Symbolically to build equations
        2. Numerically to determine output shape
        """
        # --- Symbolic evaluation ---
        symbolic_state = {**self.symbolic_vars, **self.symbolic_fluxes}
        symbolic_params = {**self.symbolic_params}
        symbolic_forcings = {**self.symbolic_forcings}

        try:
            symbolic_result = flux(
                state=symbolic_state,
                parameters=symbolic_params,
                forcings=symbolic_forcings
            )
            self.symbolic_fluxes[label] = symbolic_result
        except Exception as e:
            raise TypeError(
                f"Flux '{label}' failed symbolic evaluation. "
                f"Available state: {list(symbolic_state.keys())}, "
                f"params: {list(symbolic_params.keys())}, "
                f"forcings: {list(symbolic_forcings.keys())}. "
                f"Error: {e}"
            )

        # --- Numerical evaluation for shape ---
        state_at_zero = self._get_state_at_time(model, t_index=0)
        forcing_at_zero = {key: func(0) for key, func in model.forcing_func.items()}

        flux_init = to_ndarray(flux(
            state=state_at_zero,
            parameters=model.parameters,
            forcings=forcing_at_zero
        ))

        # Store dimensions
        array_out, flux_dims = self.return_dims_and_array(flux_init, model.time)
        model.flux_dims[label] = flux_dims

        return array_out

    def _extract_flux_element(self, flux_vector, index, flux_label, var_name):
        """Safely extract element from flux vector (handles various types)."""
        try:
            if isinstance(flux_vector, (np.ndarray, sympy.MatrixBase, list, tuple)):
                return flux_vector[index]
            elif index == 0:
                # Scalar flux for single variable
                return flux_vector
            else:
                raise TypeError(
                    f"Cannot index flux '{flux_label}' for variable '{var_name}': "
                    f"unexpected type {type(flux_vector)}"
                )
        except (IndexError, TypeError) as e:
            raise IndexError(
                f"Failed to extract element {index} from flux '{flux_label}' "
                f"for variable '{var_name}': {e}"
            )

    def assemble(self, model):
        """
        Assemble symbolic ODE system and calculate Jacobian.

        Builds symbolic equations from flux connections, computes Jacobian
        matrix, and lambdifies for numerical evaluation.
        """
        # Store all dimensions
        model.full_model_dims.update(model.var_dims)
        model.full_model_dims.update(model.flux_dims)

        # Build ordered symbolic variable list (flatten arrays)
        sym_vars_ordered = []
        for var_name in self.var_names_ordered:
            sym_var = self.symbolic_vars[var_name]
            if isinstance(sym_var, np.ndarray):
                sym_vars_ordered.extend(sym_var.flat)
            else:
                sym_vars_ordered.append(sym_var)

        # Build ordered symbolic parameter list (flatten arrays)
        sym_params_flat = []
        for param_sym in self.symbolic_params.values():
            if isinstance(param_sym, np.ndarray):
                sym_params_flat.extend(param_sym.flat)
            else:
                sym_params_flat.append(param_sym)

        for forcing_sym in self.symbolic_forcings.values():
            if isinstance(forcing_sym, np.ndarray):
                sym_params_flat.extend(forcing_sym.flat)
            else:
                sym_params_flat.append(forcing_sym)

        self.sym_params_ordered = sym_params_flat

        # Build ODE equations for each variable
        ode_eqs = []
        for var_name in self.var_names_ordered:
            flux_terms = []

            # Direct flux connections
            if var_name in model.fluxes_per_var:
                for flux_info in model.fluxes_per_var[var_name]:
                    if flux_info.get('list_input'):
                        continue

                    flux_label = flux_info['label']
                    negative = flux_info['negative']

                    if flux_label not in self.symbolic_fluxes:
                        raise KeyError(
                            f"Symbolic flux '{flux_label}' not found for variable '{var_name}'"
                        )

                    flux_expr = self.symbolic_fluxes[flux_label]
                    flux_terms.append(-flux_expr if negative else flux_expr)

            # List input flux connections
            if "list_input" in model.fluxes_per_var:
                for flux_info in model.fluxes_per_var["list_input"]:
                    flux_label = flux_info['label']
                    list_input_vars = flux_info['list_input']

                    if var_name not in list_input_vars:
                        continue

                    if flux_label not in self.symbolic_fluxes:
                        raise KeyError(
                            f"Symbolic list_input flux '{flux_label}' not found"
                        )

                    flux_vector = self.symbolic_fluxes[flux_label]
                    idx = list(list_input_vars).index(var_name)
                    negative = flux_info['negative']

                    flux_element = self._extract_flux_element(
                        flux_vector, idx, flux_label, var_name
                    )
                    flux_terms.append(-flux_element if negative else flux_element)

            # Determine the size of equations needed for this variable
            sym_var = self.symbolic_vars[var_name]
            if isinstance(sym_var, np.ndarray):
                var_size = sym_var.size
            else:
                var_size = 1

            if flux_terms:
                # Convert all flux terms to consistent array form
                flux_arrays = []
                for term in flux_terms:
                    term_array = np.asarray(term)
                    if term_array.ndim == 0:
                        # Scalar - broadcast to var_size
                        flux_arrays.append(np.full(var_size, term_array.item()))
                    elif term_array.size == 1:
                        # Single-element array - broadcast to var_size
                        flux_arrays.append(np.full(var_size, term_array.flat[0]))
                    elif term_array.size == var_size:
                        # Multi-element array matching var_size - use as is
                        flux_arrays.append(term_array.flatten())
                    else:
                        # Array size mismatch - need to sum or handle specially
                        if var_size == 1:
                            # Scalar variable receiving array flux - sum all elements
                            flux_sum = sympy.Add(*term_array.flat)
                            flux_arrays.append(np.array([flux_sum]))
                        else:
                            raise ValueError(
                                f"Flux size mismatch for variable '{var_name}': "
                                f"expected {var_size}, got {term_array.size}"
                            )

                # Sum element-wise across all flux terms
                for i in range(var_size):
                    terms_at_i = [arr[i] for arr in flux_arrays]
                    ode_eqs.append(sympy.Add(*terms_at_i))
            else:
                # No fluxes - all zeros
                ode_eqs.extend([sympy.Float(0.0)] * var_size)

        # Calculate Jacobian (exclude time variable)
        # Flatten state variables
        state_vars = []
        state_eqs = []
        idx = 0
        for var_name in self.var_names_ordered:
            if var_name == 'time':
                idx += 1
                continue

            sym_var = self.symbolic_vars[var_name]
            if isinstance(sym_var, np.ndarray):
                n_elem = sym_var.size
                state_vars.extend(sym_var.flat)
                state_eqs.extend(ode_eqs[idx:idx + n_elem])
                idx += n_elem
            else:
                state_vars.append(sym_var)
                state_eqs.append(ode_eqs[idx])
                idx += 1

        if not state_vars:
            raise RuntimeError("No state variables found for Jacobian calculation")

        logger.debug("Computing Jacobian: %d equations, %d variables",
                     len(state_eqs), len(state_vars))

        # Check for problematic expressions before computing Jacobian
        for i, eq in enumerate(state_eqs[:2]):  # Just check first 2
            logger.debug("Sample equation %d: %s", i, str(eq)[:200])

        jacobian_matrix = sympy.Matrix(state_eqs).jacobian(state_vars)

        logger.debug("Jacobian computed, shape: %s", jacobian_matrix.shape)

        try:
            # For smaller models, try full simplification
            if jacobian_matrix.shape[0] <= 4:
                logger.debug("Attempting Jacobian simplification...")
                jacobian_matrix = jacobian_matrix.applyfunc(
                    lambda x: sympy.cancel(x) if x != 0 else x
                )
                simplification_success = True
                logger.debug("Jacobian simplified successfully")
            else:
                logger.debug("Model too large, skipping Jacobian "
                             "simplification")
                simplification_success = False
        except KeyboardInterrupt:
            logger.warning("Simplification interrupted by user; "
                           "proceeding with unsimplified Jacobian")
        except Exception as e:
            logger.warning("Could not simplify Jacobian (%s); "
                           "proceeding with unsimplified Jacobian", e)

        if not simplification_success:
            logger.info("Will use numerical Jacobian for eigenvalues "
                        "if needed")

        # Lambdify for numerical evaluation
        try:
            self.f_func = sympy.lambdify(
                [sym_vars_ordered, self.sym_params_ordered],
                ode_eqs,
                modules='numpy'
            )
            self.jac_func = sympy.lambdify(
                [sym_vars_ordered, self.sym_params_ordered],
                jacobian_matrix,
                modules='numpy'
            )
        except Exception as e:
            logger.error("Lambdification failed: %s", e)
            # Only print on error - these can be huge
            # print(f"ODE equations: {ode_eqs}")
            # print(f"Jacobian: {jacobian_matrix}")
            raise RuntimeError(f"Lambdification failed: {e}")

        logger.info("HybridStabilitySolver: model assembled and "
                    "lambdified successfully")

    def _find_steady_state(self):
        """Find steady state using fsolve with Jacobian."""
        # Prepare parameter values in correct order
        # sym_params_ordered contains flattened symbolic variables with indexed names

        try:
            param_values = [
                self.param_values_map[sym.name]
                for sym in self.sym_params_ordered
            ]
        except KeyError as e:
            raise KeyError(
                f"Parameter '{e.args[0]}' not found in param_values_map. "
                f"Available: {list(self.param_values_map.keys())}"
            )

        logger.debug("Initial state size: %d", len(self.var_init_flat))
        logger.debug("Initial state: %r", self.var_init_flat)
        logger.debug("Param values size: %d", len(param_values))
        logger.debug("Param values: %r", param_values)

        def rhs_steady(y_state):
            """Right-hand side for steady state (dy/dt = 0)."""
            # Prepend time=0 to state vector
            y_full = [0.0] + list(y_state)
            try:
                derivs = self.f_func(y_full, param_values)
                result = np.array(derivs[1:], dtype=float)
                return result
            except Exception as e:
                logger.error("RHS evaluation failed: %s "
                             "(y_state=%r, y_full length=%d, "
                             "param_values length=%d)",
                             e, y_state, len(y_full), len(param_values))
                raise RuntimeError(f"RHS evaluation failed: {e}")

        def jac_steady(y_state):
            """Jacobian for steady state."""
            y_full = [0.0] + list(y_state)
            try:
                jac = self.jac_func(y_full, param_values)
                return np.array(jac, dtype=float)
            except Exception as e:
                logger.error("Jacobian evaluation failed: %s "
                             "(y_state=%r, y_full length=%d, "
                             "param_values length=%d)",
                             e, y_state, len(y_full), len(param_values))
                raise RuntimeError(f"Jacobian evaluation failed: {e}")

        # Test the functions before calling fsolve
        logger.debug("Testing RHS at initial state...")
        try:
            rhs_init = rhs_steady(self.var_init_flat)
            logger.debug("RHS at initial state: %r "
                         "(norm = %.4e)", rhs_init,
                         np.linalg.norm(rhs_init))
        except Exception as e:
            logger.error("RHS test failed: %s", e)
            raise

        logger.debug("Testing Jacobian at initial state...")
        try:
            pass
            #jac_init = jac_steady(self.var_init_flat)
            #logger.debug("Jacobian shape: %s", jac_init.shape)
            #logger.debug("Jacobian dtype: %s", jac_init.dtype)
            #logger.debug("Jacobian sample (first 3x3): %r", jac_init[:3, :3])
        except Exception as e:
            logger.error("Jacobian test failed: %s", e)
            raise

        # Solve for steady state
        logger.debug("Starting fsolve...")
        n_states = len(self.var_init_flat)
        # Set your desired threshold (e.g., <= 4 states use analytical)
        jacobian_threshold = 4

        use_analytical = (n_states <= jacobian_threshold)

        # Merge class-level defaults (xtol=1e-8, maxfev=10000) with user
        # solver_kwargs. User-supplied keys win on collision (see
        # SolverABC docstring).
        kwargs = {**self.DEFAULT_SOLVER_KWARGS, **self.solver_kwargs}

        if not use_analytical:
            # Use numerical Jacobian approximation (more robust)
            logger.debug("Using numerical Jacobian approximation...")
            y_steady, info, ier, msg = fsolve(
                rhs_steady,
                self.var_init_flat,
                full_output=True,
                **kwargs,
            )
        else:
            # Try analytical Jacobian (currently has NaN issues)
            logger.debug("Using analytical Jacobian...")
            y_steady, info, ier, msg = fsolve(
                rhs_steady,
                self.var_init_flat,
                fprime=jac_steady,
                full_output=True,
                **kwargs,
            )

        converged = (ier == 1)
        if not converged:
            try:
                residual = rhs_steady(y_steady)
                residual_str = (
                    f"residual={residual!r}, "
                    f"residual norm={np.linalg.norm(residual):.4e}"
                )
            except Exception:
                residual_str = "could not evaluate residual"
            logger.warning(
                "fsolve did not converge: %s "
                "(initial guess=%r, last iterate=%r, %s)",
                msg, self.var_init_flat, y_steady, residual_str,
            )
            y_steady = np.full_like(self.var_init_flat, np.nan)
        else:
            logger.info(
                "Steady state found, residual norm = %.4e",
                np.linalg.norm(rhs_steady(y_steady)),
            )
            logger.debug("Steady state vector: %r", y_steady)

        return y_steady, converged, jac_steady

    def _compute_eigenvalues(self, y_steady, jac_func, converged):
        """Compute eigenvalues of Jacobian at steady state."""
        n_states = len(self.var_init_flat)

        if not converged or n_states == 0:
            return np.full(n_states, np.nan)

        try:
            J_numerical = jac_func(y_steady)

            # Check for NaN/Inf in Jacobian
            if np.any(np.isnan(J_numerical)) or np.any(np.isinf(J_numerical)):
                logger.warning(
                    "Analytical Jacobian contains non-finite values "
                    "(%d NaN, %d Inf); falling back to numerical "
                    "approximation for eigenvalues",
                    int(np.sum(np.isnan(J_numerical))),
                    int(np.sum(np.isinf(J_numerical))),
                )

                # Compute Jacobian numerically using finite differences
                J_numerical = self._numerical_jacobian(y_steady)

                if np.any(np.isnan(J_numerical)) or np.any(np.isinf(J_numerical)):
                    logger.error(
                        "Numerical Jacobian also contains non-finite values"
                    )
                    return np.full(n_states, np.nan)

            eigvals = LA.eigvals(J_numerical)
            return eigvals
        except Exception as e:
            logger.warning("Eigenvalue computation failed: %s", e)
            return np.full(n_states, np.nan)

    def _numerical_jacobian(self, y_steady, eps=1e-8):
        """Compute Jacobian numerically using finite differences."""
        n = len(y_steady)

        # Get parameter values
        param_values = [
            self.param_values_map[sym.name]
            for sym in self.sym_params_ordered
        ]

        # RHS function
        def f(y):
            y_full = [0.0] + list(y)
            derivs = self.f_func(y_full, param_values)
            return np.array(derivs[1:], dtype=float)

        # Compute Jacobian using central differences
        J = np.zeros((n, n))
        f0 = f(y_steady)

        for j in range(n):
            y_plus = y_steady.copy()
            y_minus = y_steady.copy()

            # Adaptive step size based on magnitude
            h = eps * max(abs(y_steady[j]), 1.0)

            y_plus[j] += h
            y_minus[j] -= h

            f_plus = f(y_plus)
            f_minus = f(y_minus)

            J[:, j] = (f_plus - f_minus) / (2 * h)

        return J

    def _print_stability_analysis(self, y_steady, eigvals, converged):
        """Log steady state and stability analysis results."""
        lines = [
            "Bifurcation analysis results",
            f"  Steady state: {y_steady}",
            f"  Convergence:  {'Success' if converged else 'Failed'}",
        ]

        if converged and len(eigvals) > 0 and np.all(np.isfinite(eigvals)):
            max_real = np.max(np.real(eigvals))
            if max_real < -1e-9:
                stability = "STABLE"
            elif max_real > 1e-9:
                stability = "UNSTABLE"
            else:
                stability = "MARGINALLY STABLE"
            lines.append(
                f"  Stability:    {stability} "
                f"(max real part = {max_real:.4e})"
            )
        else:
            lines.append(
                f"  Eigenvalues:  could not compute (converged={converged})"
            )

        logger.info("\n".join(lines))

    def _store_results(self, model, y_steady):
        """Store steady state results in model arrays."""
        n_time = len(model.time)

        # Build result array: [initial, steady] for each variable
        full_init = np.concatenate(([model.time[0]], self.var_init_flat))
        full_steady = np.concatenate(([model.time[-1]], y_steady))

        results = np.column_stack([full_init, full_steady])

        # Distribute results to variables based on dimensions
        state_dict = {}
        idx = 0

        for var_name in self.var_names_ordered:
            if var_name not in model.full_model_dims:
                continue

            dims = model.full_model_dims[var_name]

            if dims is None:
                # Scalar variable
                state_dict[var_name] = results[idx]
                idx += 1
            elif isinstance(dims, int):
                # 1D variable
                state_dict[var_name] = results[idx:idx + dims]
                idx += dims
            else:
                # Multi-dimensional variable
                n_flat = int(np.prod(dims))
                reshaped = results[idx:idx + n_flat].reshape(*dims, 2)
                state_dict[var_name] = reshaped
                idx += n_flat

        # Assign to model arrays
        for var_key, val_array in model.variables.items():
            if var_key in state_dict:
                try:
                    val_array[...] = state_dict[var_key]
                except ValueError as e:
                    logger.error(
                        "Shape mismatch for variable '%s': "
                        "expected %s, got %s (%s)",
                        var_key, val_array.shape,
                        state_dict[var_key].shape, e,
                    )

        # Zero out flux values
        for flux_key, val_array in model.flux_values.items():
            if flux_key in model.full_model_dims:
                val_array[...] = 0.0

    def solve(self, model, time_step):
        """
        Perform steady-state and stability analysis.

        Finds equilibrium points, computes eigenvalues, prints analysis,
        and stores results in model arrays.
        """
        if self.f_func is None or self.jac_func is None:
            raise RuntimeError(
                "Model not assembled. Call assemble() before solve()."
            )

        # Find steady state
        y_steady, converged, jac_func = self._find_steady_state()

        # Compute eigenvalues
        eigvals = self._compute_eigenvalues(y_steady, jac_func, converged)

        # Print analysis
        self._print_stability_analysis(y_steady, eigvals, converged)

        # Store results
        self._store_results(model, y_steady)

    def cleanup(self):
        """Clear all solver state."""
        self.symbolic_vars.clear()
        self.symbolic_params.clear()
        self.symbolic_fluxes.clear()
        self.symbolic_forcings.clear()
        self.var_names_ordered.clear()
        self.sym_params_ordered.clear()
        self.var_init_flat.clear()
        self.param_values_map.clear()
        self.f_func = None
        self.jac_func = None

