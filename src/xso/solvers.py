from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import math

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from .model import return_dim_ndarray


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
    """

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
    """

    def __init__(self):
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

        def instability_event(t, y):
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                return 0  # Trigger event
            if np.any(y < -1e-6) or np.any(y > 1e50):  # Arbitrary upper bound
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
                                   rtol=1e-6,
                                   atol=1e-9)

        # if full_model_out.t_events[0].size > 0:
        #     print("Event triggered at t =", full_model_out.t_events[0])
        # else:
        #     print("No event detected")

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


class FSolver(IVPSolver):
    """Solver backend using scipy.integrate.fsolve to solve model steady states.

    SOLVE_IVP is a variable step-size solver for ordinary differential equations,
    included in the SciPy Python package.

    By default, it utilizes an explicit Runge-Kutta method of order 5(4).
    """


    def solve(self, model, time_step):
        """Solve for steady state (dy/dt = 0) using scipy.optimize.fsolve."""

        # Flatten initial values into a 1D vector
        state_init = np.concatenate([[v for val in self.var_init.values() for v in val.ravel()]], axis=None)
        print("InitState", state_init)

        # fsolve expects a function of y -> dy/dt, so we wrap model_function properly
        def rhs_steady(y):
            full_init = np.concatenate([y, [v for val in self.flux_init.values() for v in val.ravel()]], axis=None)
            return model.model_function(time=0, current_state=full_init, only_return_state=True)

        # Solve for steady state
        y_steady, info, ier, msg = fsolve(rhs_steady, state_init, full_output=True)

        if ier != 1:
            print(f"[WARNING] fsolve did not converge: {msg}")
            print("YSTEADY", y_steady)
            y_steady[:] = np.nan
        else:
            print("[INFO] Steady state found with residual norm:",
                  np.linalg.norm(rhs_steady(y_steady)))
            print("Residuals:", info['fvec'])  # Should be close to 0
            print("YSTEADY", y_steady)

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
    """

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
        print(f"[INFO] Initial state dimension: {len(state_init)}")
        print(f"[INFO] Initial state: {state_init}")

        # fsolve expects a function of y -> dy/dt, so we wrap model_function properly
        def rhs_steady(y):
            # Prepend time=0 to state vector, then add fluxes
            y_with_time = np.concatenate([[0.0], y])
            full_init = np.concatenate([y_with_time, [v for val in self.flux_init.values() for v in val.ravel()]],
                                       axis=None)
            derivs = model.model_function(time=0, current_state=full_init, only_return_state=True)
            # Return only non-time derivatives (skip first element which is dtime/dt = 1)
            return derivs[1:]

        # Solve for steady state (from FSolver)
        y_steady, info, ier, msg = fsolve(rhs_steady, state_init, full_output=True)

        if ier != 1:
            print(f"[WARNING] fsolve did not converge: {msg}")
            print(f"  Final state: {y_steady}")
            residual_norm = np.linalg.norm(rhs_steady(y_steady))
            print(f"  Residual norm: {residual_norm:.2e}")
            # Set to NaN if not converged
            if residual_norm > 1e-3:  # Tolerance for accepting non-converged solution
                y_steady[:] = np.nan
                converged = False
            else:
                print(f"[INFO] Accepting solution with residual norm {residual_norm:.2e}")
                converged = True
        else:
            converged = True
            residual_norm = np.linalg.norm(rhs_steady(y_steady))
            print(f"[INFO] Steady state found with residual norm: {residual_norm:.2e}")
            print(f"  Steady state: {y_steady}")

        # ========================================
        # Step 2: Compute Jacobian numerically (from FunctionalBifurcationSolver)
        # ========================================

        if converged and not np.any(np.isnan(y_steady)):
            print("[INFO] Computing Jacobian numerically...")

            # Numerical Jacobian computation using finite differences
            J = self._numerical_jacobian(rhs_steady, y_steady)

            # Check for NaN/Inf in Jacobian
            if np.any(np.isnan(J)) or np.any(np.isinf(J)):
                print(f"[WARNING] Jacobian contains NaN or Inf values")
                print(f"  NaN count: {np.sum(np.isnan(J))}")
                print(f"  Inf count: {np.sum(np.isinf(J))}")
                eigvals_computed = np.full(len(y_steady), np.nan)
            else:
                # Compute eigenvalues
                try:
                    eigvals_computed = eigvals(J)
                    print(f"[INFO] Successfully computed {len(eigvals_computed)} eigenvalues")
                except Exception as e:
                    print(f"[WARNING] Eigenvalue computation failed: {e}")
                    eigvals_computed = np.full(len(y_steady), np.nan)
        else:
            print("[WARNING] Skipping Jacobian computation due to convergence failure")
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




        print(model.variables.items())

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
        print("-" * 60)
        print("NUMERICAL STABILITY ANALYSIS RESULTS")
        print("-" * 60)
        print(f"Steady State: {y_steady}")
        print(f"Convergence: {'Success' if converged else 'Failed'}")

        if converged and len(eigvals) > 0 and np.all(np.isfinite(eigvals)):
            # Analyze eigenvalues
            real_parts = np.real(eigvals)
            imag_parts = np.imag(eigvals)

            max_real = np.max(real_parts)
            min_real = np.min(real_parts)

            print(f"\nEigenvalue Analysis:")
            print(f"  Number of eigenvalues: {len(eigvals)}")
            print(f"  Max real part: {max_real:.4e}")
            print(f"  Min real part: {min_real:.4e}")

            # Determine stability (from FunctionalBifurcationSolver)
            if max_real < -1e-9:
                stability = "STABLE"
            elif max_real > 1e-9:
                stability = "UNSTABLE"
            else:
                stability = "MARGINALLY STABLE"

            print(f"\nStability: {stability} (max real part: {max_real:.4e})")

            # Count eigenvalue types
            n_positive = np.sum(real_parts > 1e-9)
            n_negative = np.sum(real_parts < -1e-9)
            n_zero = np.sum(np.abs(real_parts) <= 1e-9)

            print(f"  Positive real parts: {n_positive}")
            print(f"  Negative real parts: {n_negative}")
            print(f"  Near-zero real parts: {n_zero}")

            # Check for oscillatory behavior
            n_complex = np.sum(np.abs(imag_parts) > 1e-10)
            if n_complex > 0:
                print(f"  Complex eigenvalue pairs: {n_complex // 2}")
                max_frequency = np.max(np.abs(imag_parts))
                print(f"  Max oscillation frequency: {max_frequency:.4f}")

            # Print first few eigenvalues for inspection
            n_show = min(5, len(eigvals))
            print(f"\n  First {n_show} eigenvalues:")
            for i in range(n_show):
                if abs(imag_parts[i]) > 1e-10:
                    print(f"    λ_{i + 1} = {real_parts[i]:.4e} ± {abs(imag_parts[i]):.4e}i")
                else:
                    print(f"    λ_{i + 1} = {real_parts[i]:.4e}")
        else:
            print(f"\nEigenvalues: Could not compute (converged={converged})")

        print("-" * 60)


class NumericalStabilitySolver(IVPSolver):
    """
    Solver backend for numerical steady-state and stability analysis.

    Inherits from IVPSolver to reuse initialization and setup methods,
    but implements its own solve method that:
    1. Finds steady state using fsolve (like FSolver)
    2. Numerically computes Jacobian
    3. Computes eigenvalues for stability analysis
    4. Stores results like FSolver
    """

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
        print(f"[INFO] Initial state dimension: {len(state_init)}")
        print(f"[INFO] Initial state: {state_init}")

        # fsolve expects a function of y -> dy/dt, so we wrap model_function properly
        def rhs_steady(y):
            # Prepend time=0 to state vector, then add fluxes
            y_with_time = np.concatenate([[0.0], y])
            full_init = np.concatenate([y_with_time, [v for val in self.flux_init.values() for v in val.ravel()]],
                                       axis=None)
            derivs = model.model_function(time=0, current_state=full_init, only_return_state=True)
            # Return only non-time derivatives (skip first element which is dtime/dt = 1)
            return derivs[1:]

        # Solve for steady state (from FSolver)
        y_steady, info, ier, msg = fsolve(rhs_steady, state_init, full_output=True)

        if ier != 1:
            print(f"[WARNING] fsolve did not converge: {msg}")
            print(f"  Final state: {y_steady}")
            residual_norm = np.linalg.norm(rhs_steady(y_steady))
            print(f"  Residual norm: {residual_norm:.2e}")
            # Set to NaN if not converged
            if residual_norm > 1e-3:  # Tolerance for accepting non-converged solution
                y_steady[:] = np.nan
                converged = False
            else:
                print(f"[INFO] Accepting solution with residual norm {residual_norm:.2e}")
                converged = True
        else:
            converged = True
            residual_norm = np.linalg.norm(rhs_steady(y_steady))
            print(f"[INFO] Steady state found with residual norm: {residual_norm:.2e}")
            print(f"  Steady state: {y_steady}")

        # ========================================
        # Step 2: Compute Jacobian numerically (from FunctionalBifurcationSolver)
        # ========================================

        if converged and not np.any(np.isnan(y_steady)):
            print("[INFO] Computing Jacobian numerically...")

            # Numerical Jacobian computation using finite differences
            J = self._numerical_jacobian(rhs_steady, y_steady)

            # Check for NaN/Inf in Jacobian
            if np.any(np.isnan(J)) or np.any(np.isinf(J)):
                print(f"[WARNING] Jacobian contains NaN or Inf values")
                print(f"  NaN count: {np.sum(np.isnan(J))}")
                print(f"  Inf count: {np.sum(np.isinf(J))}")
                eigvals_computed = np.full(len(y_steady), np.nan)
            else:
                # Compute eigenvalues
                try:
                    eigvals_computed = eigvals(J)
                    print(f"[INFO] Successfully computed {len(eigvals_computed)} eigenvalues")
                except Exception as e:
                    print(f"[WARNING] Eigenvalue computation failed: {e}")
                    eigvals_computed = np.full(len(y_steady), np.nan)
        else:
            print("[WARNING] Skipping Jacobian computation due to convergence failure")
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
        print("-" * 60)
        print("NUMERICAL STABILITY ANALYSIS RESULTS")
        print("-" * 60)
        print(f"Steady State: {y_steady}")
        print(f"Convergence: {'Success' if converged else 'Failed'}")

        if converged and len(eigvals) > 0 and np.all(np.isfinite(eigvals)):
            # Analyze eigenvalues
            real_parts = np.real(eigvals)
            imag_parts = np.imag(eigvals)

            max_real = np.max(real_parts)
            min_real = np.min(real_parts)

            print(f"\nEigenvalue Analysis:")
            print(f"  Number of eigenvalues: {len(eigvals)}")
            print(f"  Max real part: {max_real:.4e}")
            print(f"  Min real part: {min_real:.4e}")

            # Determine stability (from FunctionalBifurcationSolver)
            if max_real < -1e-9:
                stability = "STABLE"
            elif max_real > 1e-9:
                stability = "UNSTABLE"
            else:
                stability = "MARGINALLY STABLE"

            print(f"\nStability: {stability} (max real part: {max_real:.4e})")

            # Count eigenvalue types
            n_positive = np.sum(real_parts > 1e-9)
            n_negative = np.sum(real_parts < -1e-9)
            n_zero = np.sum(np.abs(real_parts) <= 1e-9)

            print(f"  Positive real parts: {n_positive}")
            print(f"  Negative real parts: {n_negative}")
            print(f"  Near-zero real parts: {n_zero}")

            # Check for oscillatory behavior
            n_complex = np.sum(np.abs(imag_parts) > 1e-10)
            if n_complex > 0:
                print(f"  Complex eigenvalue pairs: {n_complex // 2}")
                max_frequency = np.max(np.abs(imag_parts))
                print(f"  Max oscillation frequency: {max_frequency:.4f}")

            # Print first few eigenvalues for inspection
            n_show = min(5, len(eigvals))
            print(f"\n  First {n_show} eigenvalues:")
            for i in range(n_show):
                if abs(imag_parts[i]) > 1e-10:
                    print(f"    λ_{i + 1} = {real_parts[i]:.4e} ± {abs(imag_parts[i]):.4e}i")
                else:
                    print(f"    λ_{i + 1} = {real_parts[i]:.4e}")
        else:
            print(f"\nEigenvalues: Could not compute (converged={converged})")

        print("-" * 60)

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
    storage arrays in xsimlab backend."""

    def __init__(self):
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
            array_out[0] = value
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
                val[self.time_index] = val[self.time_index - 1] + state_dict[key] * time_step

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
    """

    def __init__(self):
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

        print(f"[DEBUG] Computing Jacobian: {len(state_eqs)} equations, {len(state_vars)} variables")

        # Check for problematic expressions before computing Jacobian
        for i, eq in enumerate(state_eqs[:2]):  # Just check first 2
            print(f"[DEBUG] Sample equation {i}: {str(eq)[:200]}")

        jacobian_matrix = sympy.Matrix(state_eqs).jacobian(state_vars)

        print(f"[DEBUG] Jacobian computed, shape: {jacobian_matrix.shape}")



        try:
            # For smaller models, try full simplification
            if jacobian_matrix.shape[0] <= 4:
                print(f"[DEBUG] Attempting Jacobian simplification...")
                jacobian_matrix = jacobian_matrix.applyfunc(
                    lambda x: sympy.cancel(x) if x != 0 else x
                )
                simplification_success = True
                print(f"[DEBUG] Jacobian simplified successfully")
            else:
                print(f"[DEBUG] Model too large, no attempt at Jacobian simplification...")
                simplification_success = False
        except KeyboardInterrupt:
            print(f"[WARNING] Simplification interrupted by user")
            print(f"[INFO] Proceeding with unsimplified Jacobian")
        except Exception as e:
            print(f"[WARNING] Could not simplify Jacobian: {e}")
            print(f"[INFO] Proceeding with unsimplified Jacobian")

        if not simplification_success:
            print(f"[INFO] Will automatically use numerical Jacobian for eigenvalues if needed")

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
            print("ERROR during lambdification")
            # Only print on error - these can be huge
            # print(f"ODE equations: {ode_eqs}")
            # print(f"Jacobian: {jacobian_matrix}")
            raise RuntimeError(f"Lambdification failed: {e}")

        print("[BifurcationSolver] Model assembled and lambdified successfully")

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

        print(f"[DEBUG] Initial state size: {len(self.var_init_flat)}")
        print(f"[DEBUG] Initial state: {self.var_init_flat}")
        print(f"[DEBUG] Param values size: {len(param_values)}")
        print(f"[DEBUG] Param values: {param_values}")

        def rhs_steady(y_state):
            """Right-hand side for steady state (dy/dt = 0)."""
            # Prepend time=0 to state vector
            y_full = [0.0] + list(y_state)
            try:
                derivs = self.f_func(y_full, param_values)
                result = np.array(derivs[1:], dtype=float)
                return result
            except Exception as e:
                print(f"[ERROR] RHS evaluation failed")
                print(f"  y_state: {y_state}")
                print(f"  y_full length: {len(y_full)}")
                print(f"  param_values length: {len(param_values)}")
                raise RuntimeError(f"RHS evaluation failed: {e}")

        def jac_steady(y_state):
            """Jacobian for steady state."""
            y_full = [0.0] + list(y_state)
            try:
                jac = self.jac_func(y_full, param_values)
                return np.array(jac, dtype=float)
            except Exception as e:
                print(f"[ERROR] Jacobian evaluation failed")
                print(f"  y_state: {y_state}")
                print(f"  y_full length: {len(y_full)}")
                print(f"  param_values length: {len(param_values)}")
                raise RuntimeError(f"Jacobian evaluation failed: {e}")

        # Test the functions before calling fsolve
        print("[DEBUG] Testing RHS at initial state...")
        try:
            rhs_init = rhs_steady(self.var_init_flat)
            print(f"[DEBUG] RHS at initial state: {rhs_init}")
            print(f"[DEBUG] RHS norm: {np.linalg.norm(rhs_init)}")
        except Exception as e:
            print(f"[ERROR] RHS test failed: {e}")
            raise

        print("[DEBUG] Testing Jacobian at initial state...")
        try:
            pass
            #jac_init = jac_steady(self.var_init_flat)
            #print(f"[DEBUG] Jacobian shape: {jac_init.shape}")
            #print(f"[DEBUG] Jacobian dtype: {jac_init.dtype}")
            #print(f"[DEBUG] Jacobian contains NaN: {np.any(np.isnan(jac_init))}")
            #print(f"[DEBUG] Jacobian contains Inf: {np.any(np.isinf(jac_init))}")
            #print(f"[DEBUG] Jacobian min/max: {np.min(jac_init)}, {np.max(jac_init)}")
            #print(f"[DEBUG] Jacobian sample (first 3x3):\n{jac_init[:3, :3]}")
        except Exception as e:
            print(f"[ERROR] Jacobian test failed: {e}")
            raise

        # Solve for steady state
        print("[DEBUG] Starting fsolve...")
        n_states = len(self.var_init_flat)
        # Set your desired threshold (e.g., <= 4 states use analytical)
        jacobian_threshold = 4

        use_analytical = (n_states <= jacobian_threshold)

        if not use_analytical:
            # Use numerical Jacobian approximation (more robust)
            print("[DEBUG] Using numerical Jacobian approximation...")
            y_steady, info, ier, msg = fsolve(
                rhs_steady,
                self.var_init_flat,
                full_output=True,
                xtol=1e-8,
                maxfev=10000
            )
        else:
            # Try analytical Jacobian (currently has NaN issues)
            print("[DEBUG] Using analytical Jacobian...")
            y_steady, info, ier, msg = fsolve(
                rhs_steady,
                self.var_init_flat,
                fprime=jac_steady,
                full_output=True,
                xtol=1e-8,
                maxfev=10000
            )

        converged = (ier == 1)
        if not converged:
            print(f"[WARNING] fsolve did not converge: {msg}")
            print(f"  Initial guess: {self.var_init_flat}")
            print(f"  Final attempt: {y_steady}")
            try:
                residual = rhs_steady(y_steady)
                print(f"  Residual: {residual}")
                print(f"  Residual norm: {np.linalg.norm(residual)}")
            except:
                print(f"  Could not evaluate residual")
            y_steady = np.full_like(self.var_init_flat, np.nan)
        else:
            print(f"[INFO] Steady state found!")
            print(f"  Steady state: {y_steady}")
            print(f"  Residual norm: {np.linalg.norm(rhs_steady(y_steady))}")

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
                print(f"[WARNING] Analytical Jacobian contains NaN or Inf values")
                print(f"  NaN count: {np.sum(np.isnan(J_numerical))}")
                print(f"  Inf count: {np.sum(np.isinf(J_numerical))}")
                print(f"[INFO] Falling back to numerical Jacobian approximation for eigenvalues...")

                # Compute Jacobian numerically using finite differences
                J_numerical = self._numerical_jacobian(y_steady)

                if np.any(np.isnan(J_numerical)) or np.any(np.isinf(J_numerical)):
                    print(f"[ERROR] Numerical Jacobian also has NaN/Inf!")
                    return np.full(n_states, np.nan)

            eigvals = LA.eigvals(J_numerical)
            return eigvals
        except Exception as e:
            print(f"[WARNING] Eigenvalue computation failed: {e}")
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
        """Print steady state and stability analysis results."""
        print("-" * 50)
        print("BIFURCATION ANALYSIS RESULTS")
        print("-" * 50)
        print(f"Steady State: {y_steady}")
        print(f"Convergence: {'Success' if converged else 'Failed'}")

        if converged and len(eigvals) > 0 and np.all(np.isfinite(eigvals)):
            #print(f"\nEigenvalues: {eigvals}")
            #print(f"  Real parts: {np.real(eigvals)}")
            #print(f"  Imaginary parts: {np.imag(eigvals)}")

            max_real = np.max(np.real(eigvals))

            if max_real < -1e-9:
                stability = "STABLE"
            elif max_real > 1e-9:
                stability = "UNSTABLE"
            else:
                stability = "MARGINALLY STABLE"

            print(f"\nStability: {stability} (max real part: {max_real:.4e})")
        else:
            print(f"\nEigenvalues: Could not compute (converged={converged})")

        print("-" * 50)

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
                    print(
                        f"[ERROR] Shape mismatch for variable '{var_key}': "
                        f"expected {val_array.shape}, got {state_dict[var_key].shape}. {e}"
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

