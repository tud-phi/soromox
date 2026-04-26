__all__ = ["DynamicalSystem"]
import math
import warnings
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax

# Diffrax (for time integration helpers)
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    ConstantStepSize,
    ODETerm,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from jax import Array, lax, vmap
from jax import numpy as jnp

from soromox.systems.system_state import SystemState


class DynamicalSystem(eqx.Module):
    num_dofs: int = eqx.field(static=True)  # Number of degrees of freedom
    num_actuators: int = eqx.field(static=True)  # Number of actuators

    @abstractmethod
    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: tuple | None = None
    ) -> Array:
        """
        Compute the forward dynamics of the system.

        This method computes the state derivative yd = [qd, qdd] given the
        current time, state, and actuation inputs.

        Args:
            t (Array): Current time.
            y (Array): State vector containing configuration and velocity.
            actuation_args (Optional[Tuple]): Tuple of actuation inputs, typically (u, tau_ext) where
                u is the control input and tau_ext is the external force/torque.

        Returns:
            yd (Array): State derivative yd with the same shape as y.
        """
        ...

    @staticmethod
    def _compute_save_times(
        t0: float | Array,
        t1: float | Array,
        solver_dt: float | Array,
        save_dt: float | Array | None,
        save_ts: Array | None = None,
    ) -> Array:
        """Build the array of time stamps to save during integration."""
        if save_ts is not None:
            return save_ts

        assert save_dt is not None, "Either save_ts or save_dt must be provided."
        assert save_dt > 0.0, "save_dt must be positive."
        assert save_dt >= solver_dt, (
            "save_dt must be greater than or equal to the solver step size."
        )
        return jnp.arange(t0, t1 + save_dt, save_dt)

    @staticmethod
    def _zero_like_control_state(control_state: Any | None) -> Any | None:
        """Create a zero-structured control_state derivative matching the provided PyTree."""
        if control_state is None:
            return None
        return jax.tree_util.tree_map(jnp.zeros_like, control_state)

    def _open_loop_forward_dynamics(
        self, t: Array, ode_state: Array, actuation_args: tuple[Array, Array]
    ) -> Array:
        """Wrapper for forward_dynamics used in open-loop rollouts.

        This is defined as a class method rather than an inline closure to avoid
        JAX/XLA compilation overhead associated with closures.
        """
        base_u_val, base_tau_ext = actuation_args
        return self.forward_dynamics(t, ode_state, (base_u_val, base_tau_ext))

    def _closed_loop_forward_dynamics(
        self,
        t: Array,
        ode_state,
        actuation_args: tuple[Array, Array, Callable, bool, Any | None],
    ):
        """Wrapper for forward_dynamics used in closed-loop rollouts.

        This is defined as a class method rather than an inline closure to avoid
        JAX/XLA compilation overhead associated with closures.

        Args:
            t: Current time.
            ode_state: Current ODE state (y or (y, control_state) depending on track_control_state).
            actuation_args: Tuple containing:
                - base_u_val: Base actuation vector.
                - base_tau_ext: External torques/forces.
                - controller: Controller callable.
                - track_control_state: Whether to track control state.
                - zero_control_state_dot: Zero-structured control state derivative.
        """
        (
            base_u_val,
            base_tau_ext,
            controller,
            track_control_state,
            zero_control_state_dot,
        ) = actuation_args

        if track_control_state:
            y, ctrl_state = ode_state
        else:
            y = ode_state
            ctrl_state = None

        system_state = SystemState(t=t, y=y, u=base_u_val, control_state=ctrl_state)
        u_control, control_state_dot = controller(system_state)
        if control_state_dot is not None and not track_control_state:
            raise ValueError(
                "Controller returned a control_state derivative but no initial control_state was provided."
            )
        u_total = base_u_val + u_control
        yd = self.forward_dynamics(t, y, (u_total, base_tau_ext))
        if track_control_state:
            control_state_dot = (
                control_state_dot
                if control_state_dot is not None
                else zero_control_state_dot
            )
            return yd, control_state_dot
        return yd

    def rollout_to(
        self,
        initial_state: SystemState,
        u: Array | None = None,
        tau_ext: Array | None = None,
        t1: float | Array = 10.0,
        solver_dt: float | Array = 1e-4,
        save_dt: float | Array | None = 0.01,
        save_ts: Array | None = None,
        solver: AbstractSolver | None = None,
        stepsize_controller: AbstractStepSizeController | None = ConstantStepSize(),
        max_steps: int | None = None,
    ) -> SystemState:
        """
        Roll out the system dynamics in open loop using Diffrax.

        Args:
            initial_state: Dataclass holding the initial time, system state vector (y),
                optional actuation (u), and optional control state.
            u: Constant actuation to apply throughout the rollout. If not provided,
                falls back to `initial_state.u`, then zeros.
            tau_ext: External forces/torques applied to the system (broadcast as constant).
            t1: Final time of the simulation, included in the saved trajectory.
            solver_dt: Time step for the solver.
            save_dt: Time interval at which to save the solution when `save_ts` is
                not provided.
            save_ts: Explicit time points to be saved in the output. Must be within
                [initial_state.t, t1]. Falls back to `save_dt` if None.
            solver: Solver to use for the ODE integration.
            stepsize_controller: Stepsize controller for the solver.
            max_steps: Maximum number of steps for the solver.

        Returns:
            SystemState PyTree containing time samples, system state trajectory,
            actuation at each saved step, and (optionally) the control state
            trajectory. Each leaf has a leading time dimension aligned with `save_ts`.
        """
        if initial_state.control_state is not None:
            warnings.warn(
                "control_state provided to rollout_to is ignored for open-loop rollouts.",
                UserWarning,
                stacklevel=2,
            )
        base_u = u if u is not None else initial_state.u
        if base_u is None:
            base_u = jnp.zeros((self.num_actuators,))

        y0 = initial_state.y  # initial state vector
        t0 = initial_state.t
        save_ts = self._compute_save_times(t0, t1, solver_dt, save_dt, save_ts)

        term = ODETerm(self._open_loop_forward_dynamics)
        saveat = SaveAt(ts=save_ts)  # Save at specified time points

        if solver is None:
            solver = Tsit5()

        sol = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1 + solver_dt,
            dt0=solver_dt,
            y0=y0,
            args=(base_u, tau_ext),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )

        ts = sol.ts
        y_ts = sol.ys

        us = jnp.broadcast_to(base_u, (ts.shape[0], base_u.shape[0]))

        return SystemState(t=ts, y=y_ts, u=us, control_state=None)

    def rollout_closed_loop_to(
        self,
        initial_state: SystemState,
        controller: Callable[[SystemState], tuple[Array, Any | None]],
        tau_ext: Array | None = None,
        t1: float | Array = 10.0,
        solver_dt: float | Array = 1e-4,
        save_dt: float | Array = 0.01,
        save_ts: Array | None = None,
        solver: AbstractSolver | None = None,
        stepsize_controller: AbstractStepSizeController | None = ConstantStepSize(),
        max_steps: int | None = None,
    ) -> SystemState:
        """
        Roll out the system dynamics in closed loop using Diffrax.

        The provided controller is queried at every integration step with the current
        system state (and optional control_state). Its actuation output is added to any
        feed-forward actuation contained in `initial_state.u`.

        Args:
            initial_state: Dataclass holding the initial time, system state vector (y),
                optional actuation (u), and optional control state.
            controller: Callable that accepts a `SystemState` and returns a tuple
                `(u_control, control_state_dot)`. `u_control` is added to the base
                actuation from the initial state.
            tau_ext: External forces/torques applied to the system (broadcast as constant).
            t1: Final time of the simulation, included in the saved trajectory.
            solver_dt: Time step for the solver.
            save_dt: Time interval at which to save the solution when `save_ts` is
                not provided.
            save_ts: Explicit time points to be saved in the output. Must be within
                [initial_state.t, t1]. Falls back to `save_dt` if None.
            solver: Solver to use for the ODE integration.
            stepsize_controller: Stepsize controller for the solver.
            max_steps: Maximum number of steps for the solver.

        Returns:
            SystemState PyTree containing time samples, system state trajectory,
            actuation at each saved step, and controller state trajectory. Each
            leaf has a leading time dimension aligned with `save_ts`.
        """
        if controller is None:
            raise ValueError("A controller must be provided for closed-loop rollouts.")

        base_u = initial_state.u
        if base_u is None:
            base_u = jnp.zeros((self.num_actuators,))

        y0 = initial_state.y

        controller_state0 = initial_state.control_state
        track_control_state = controller_state0 is not None
        zero_control_state_dot = self._zero_like_control_state(controller_state0)

        t0 = initial_state.t
        save_ts = self._compute_save_times(t0, t1, solver_dt, save_dt, save_ts)

        # Use class method to avoid closure overhead
        actuation_args = (
            base_u,
            tau_ext,
            controller,
            track_control_state,
            zero_control_state_dot,
        )
        term = ODETerm(self._closed_loop_forward_dynamics)
        saveat = SaveAt(ts=save_ts)  # Save at specified time points

        y0_for_solver = (y0, controller_state0) if track_control_state else y0

        if solver is None:
            solver = Tsit5()

        sol = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1 + solver_dt,
            dt0=solver_dt,
            y0=y0_for_solver,
            args=actuation_args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )

        ts = sol.ts
        if track_control_state:
            y_ts, controller_state_out = sol.ys
        else:
            y_ts = sol.ys
            controller_state_out = None

        if track_control_state:
            u_controls = vmap(
                lambda t_i, y_i, c_i: controller(
                    SystemState(t=t_i, y=y_i, u=base_u, control_state=c_i)
                )[0]
            )(ts, y_ts, controller_state_out)
        else:
            u_controls = vmap(
                lambda t_i, y_i: controller(
                    SystemState(t=t_i, y=y_i, u=base_u, control_state=None)
                )[0]
            )(ts, y_ts)
        us = u_controls + base_u

        return SystemState(t=ts, y=y_ts, u=us, control_state=controller_state_out)

    def rollout_discrete_closed_loop_to(
        self,
        initial_state: SystemState,
        controller: Callable[[SystemState], tuple[Array, Any | None]],
        tau_ext: Array | None = None,
        duration: float = 10.0,
        solver_dt: float | Array = 1e-4,
        control_dt: float = 0.01,
        save_dt: float = 0.01,
        solver: AbstractSolver | None = None,
        stepsize_controller: AbstractStepSizeController | None = ConstantStepSize(),
        max_steps: int | None = None,
    ) -> SystemState:
        """
        Roll out the system in discrete-time closed loop.

        The controller is evaluated every ``control_dt`` seconds. Between control
        evaluations the actuation is held constant and the system is integrated
        with Diffrax (equivalent to repeatedly calling ``rollout_to`` over each
        control interval).

        The controller signature mirrors ``rollout_closed_loop_to``. If it returns a
        ``control_state_dot`` and ``initial_state.control_state`` is provided, the
        control state is advanced with a forward-Euler step over the control period;
        otherwise the control state is held constant between controller calls.

        The rollout uses a regular time grid. For static tracing/compilation the
        following constraints are enforced:
        - ``duration`` is an integer multiple of ``control_dt``.
        - ``control_dt`` is an integer multiple of ``save_dt``.

        Args:
            initial_state: initial time/state (and optional feedforward actuation/control state).
            controller: callable returning ``(u_control, control_state_dot)``.
            tau_ext: constant external wrench/force applied during the rollout.
            duration: simulation duration in seconds.
            solver_dt: initial step size for the solver.
            control_dt: sampling period for the controller. Must be positive.
            save_dt: save interval. Must be positive and evenly divide ``control_dt``.
            solver: Diffrax solver.
            stepsize_controller: Diffrax stepsize controller.
            max_steps: maximum solver steps.
        """
        if controller is None:
            raise ValueError("A controller must be provided for closed-loop rollouts.")

        if not isinstance(duration, (float, int)):
            raise TypeError("duration must be a float.")
        if not isinstance(control_dt, (float, int)):
            raise TypeError("control_dt must be a float.")
        if not isinstance(save_dt, (float, int)):
            raise TypeError("save_dt must be a float.")

        duration = float(duration)
        control_dt = float(control_dt)
        save_dt = float(save_dt)

        if duration <= 0.0:
            raise ValueError("duration must be positive.")
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive.")
        if save_dt <= 0.0:
            raise ValueError("save_dt must be positive.")
        if save_dt > control_dt:
            raise ValueError("save_dt must be less than or equal to control_dt.")

        # Require commensurate, regular control/save grids so the (control, save) sequence
        # is static and can be traced/compiled without runtime masking.
        saves_per_control = int(round(control_dt / save_dt))
        if saves_per_control < 1 or not math.isclose(
            saves_per_control * save_dt,
            control_dt,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("control_dt must be an integer multiple of save_dt.")

        num_control_steps = int(round(duration / control_dt))
        if num_control_steps < 1 or not math.isclose(
            num_control_steps * control_dt,
            duration,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("duration must be an integer multiple of control_dt.")

        if solver is None:
            solver = Tsit5()

        term = ODETerm(self._open_loop_forward_dynamics)

        base_u = initial_state.u
        if base_u is None:
            base_u = jnp.zeros((self.num_actuators,))

        y_curr = initial_state.y
        control_state = initial_state.control_state
        track_control_state = control_state is not None
        zero_control_state_dot = self._zero_like_control_state(control_state)

        t0 = initial_state.t
        t0_dtype = jnp.asarray(t0).dtype
        save_offsets = save_dt * jnp.arange(saves_per_control + 1, dtype=t0_dtype)
        control_offsets = control_dt * jnp.arange(num_control_steps, dtype=t0_dtype)
        t_starts = t0 + control_offsets
        t_ends = t_starts + control_dt

        def body(
            carry: tuple[Array, Any | None],
            t_bounds: tuple[Array, Array],
        ) -> tuple[tuple[Array, Any | None], tuple[Array, Array, Array, Any | None]]:
            """
            Advance one control interval: evaluate controller at ``t_start``, hold actuation
            for the interval, integrate dynamics, and return saved trajectories.

            Args:
                carry: tuple of current state ``y_in`` and current control state ``ctrl_state_in``.
                t_bounds: tuple of start and end times for this control interval.
            Returns:
                next_carry: tuple of next state and next control state.
                outputs: tuple of saved times, states, actuations, and control states.
            """
            y_in, ctrl_state_in = carry
            t_start, t_end = t_bounds

            system_state = SystemState(
                t=t_start, y=y_in, u=base_u, control_state=ctrl_state_in
            )
            u_control, control_state_dot = controller(system_state)
            if control_state_dot is not None and not track_control_state:
                raise ValueError(
                    "Controller returned a control_state derivative but no initial control_state was provided."
                )
            control_state_dot = (
                control_state_dot
                if control_state_dot is not None
                else zero_control_state_dot
            )

            u_total = base_u + u_control

            ts_control_interval = t_start + save_offsets
            # Protect against tiny floating-point drift (especially visible during the
            # backward pass) that can push the final save time slightly outside
            # [t_start, t_end].
            ts_control_interval = jnp.clip(ts_control_interval, t_start, t_end)
            saveat = SaveAt(ts=ts_control_interval, t0=False, t1=False)
            sol = diffeqsolve(
                terms=term,
                solver=solver,
                t0=t_start,
                t1=t_end,
                dt0=solver_dt,
                y0=y_in,
                args=(u_total, tau_ext),
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                max_steps=max_steps,
            )

            ctrl_state_next = (
                jax.tree_util.tree_map(
                    lambda c, cdot: c + cdot * (t_end - t_start),
                    ctrl_state_in,
                    control_state_dot,
                )
                if track_control_state
                else None
            )

            us_out = jnp.broadcast_to(u_total, (sol.ts.shape[0], u_total.shape[0]))

            if track_control_state:
                # Controller state is held constant over the interval, then advanced at the end.
                cs_out = jax.tree_util.tree_map(
                    lambda c, c_next: jnp.concatenate(
                        [
                            jnp.broadcast_to(
                                c, (sol.ts.shape[0] - 1,) + c.shape  # type: ignore[misc]
                            ),
                            c_next[None, ...],
                        ],
                        axis=0,
                    ),
                    ctrl_state_in,
                    ctrl_state_next,
                )
            else:
                cs_out = None

            y_next = sol.ys[-1]

            outputs = (sol.ts, sol.ys, us_out, cs_out)
            next_carry = (y_next, ctrl_state_next)
            return next_carry, outputs

        _, scan_outputs = lax.scan(body, (y_curr, control_state), (t_starts, t_ends))
        ts_segments, ys_segments, us_segments, cs_segments = scan_outputs

        # Drop the first (duplicate) sample of every interval except the first.
        if num_control_steps == 1:
            ts_all = ts_segments[0]
            ys_all = ys_segments[0]
            us_all = us_segments[0]
            control_state_out = cs_segments[0] if track_control_state else None
        else:
            ts_tail = ts_segments[1:, 1:].reshape(-1)
            ys_tail = ys_segments[1:, 1:].reshape((-1,) + ys_segments.shape[2:])
            us_tail = us_segments[1:, 1:].reshape((-1, self.num_actuators))
            ts_all = jnp.concatenate([ts_segments[0], ts_tail], axis=0)
            ys_all = jnp.concatenate([ys_segments[0], ys_tail], axis=0)
            us_all = jnp.concatenate([us_segments[0], us_tail], axis=0)
            if track_control_state and cs_segments is not None:
                control_state_out = jax.tree_util.tree_map(
                    lambda arr: jnp.concatenate(
                        [arr[0], arr[1:, 1:].reshape((-1,) + arr.shape[2:])], axis=0
                    ),
                    cs_segments,
                )
            else:
                control_state_out = None

        return SystemState(
            t=ts_all, y=ys_all, u=us_all, control_state=control_state_out
        )
