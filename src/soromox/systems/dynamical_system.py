__all__ = ["DynamicalSystem"]
import equinox as eqx
from jax import Array, jit, lax
from jax import numpy as jnp
import numpy as onp
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type, Union, Optional

# Diffrax (for time integration helpers)
from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    Tsit5,
    AbstractStepSizeController,
    ConstantStepSize,
    AbstractSolver,
)


class DynamicalSystem(eqx.Module):
    def resolve_upon_time(
        self,
        q0: Array,
        qd0: Array,
        u: Optional[Array] = None,
        tau_ext: Optional[Array] = None,
        t0: Optional[float] = 0.0,
        t1: Optional[float] = 10.0,
        dt: Optional[float] = 1e-4,
        save_every_n_steps: int = 1,
        solver: Optional[AbstractSolver] = Tsit5(),
        stepsize_controller: Optional[AbstractStepSizeController] = ConstantStepSize(),
        max_steps: Optional[int] = None,
    ) -> Tuple[Array, Array, Array]:
        """
        Resolve the system dynamics over time using Diffrax.

        Args:
            q0 (Array): Initial configuration (strains).
            qd0 (Array): Initial velocity (strains).
            u (Array, optional): Actuation/control input.
                Default is None (no actuation).
            tau_ext (Array, optional): External forces/torques applied to the system.
            t0 (float, optional): Initial time.
                Default is 0.0.
            t1 (float, optional): Final time.
                Default is 10.0.
            dt (float, optional): Time step for the solver.
                Default is 1e-4.
            save_every_n_steps (int, optional): Determines how many time steps to skip
                when saving the output. For example, if set to 1, every time step is saved;
                if set to 10, every 10th time step is saved.
                Default is 1 (save every step).
            solver (AbstractSolver, optional): Solver to use for the ODE integration.
                Default is Tsit5() (Runge-Kutta 5(4) method).
            stepsize_controller (PIDController, optional): Stepsize controller for the solver.
                Default is ConstantStepSize().
            max_steps (int, optional): Maximum number of steps for the solver.
                Default is None (no limit).

        Returns:
            ts (Array): Time points at which the solution is saved.
            qs (Array): Configuration (strains) at the saved time points.
            qds (Array): Velocity (strains) at the saved time points.
        """
        y0 = jnp.concatenate([q0, qd0])  # Initial state vector
        if u is None:
            u = jnp.zeros((self.num_actuators,))
        if tau_ext is None:
            tau_ext = jnp.zeros((q0.shape[-1],))

        term = ODETerm(self.forward_dynamics)

        t = jnp.arange(t0, t1, dt)  # Time points for the solution
        
        assert save_every_n_steps > 0, "save_every_n_steps must be a positive integer."
        assert isinstance(save_every_n_steps, int), "save_every_n_steps must be an integer."
        saveat = SaveAt(ts=t[::save_every_n_steps])  # Save at specified time points

        sol = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t[0],
            t1=t[-1],
            dt0=dt,
            y0=y0,
            args=(u, tau_ext),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )

        ts = sol.ts
        # Extract the configuration and velocity from the solution
        y_out = sol.ys
        qs, qds = jnp.split(y_out, 2, axis=1)

        return ts, qs, qds
