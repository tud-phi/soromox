"""Unit tests for backend selection and public dynamics dispatch."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array
from numpy.testing import assert_allclose

from soromox.systems.execution import (
    GVS_DYNAMICS,
    PCS_DYNAMICS,
    PCS_KINEMATICS,
    ExecutionBackend,
    dispatch_dynamics_terms,
    dispatch_kinematics,
    dispatch_kinematics_abscissa_batched,
)

jax.config.update("jax_enable_x64", True)


class _DispatchProbe(eqx.Module):
    """Minimal public dynamics model used to observe dispatch behavior."""

    scale: Array
    backend: ExecutionBackend = eqx.field(static=True)
    num_coordinates: int = eqx.field(static=True)
    num_velocities: int = eqx.field(static=True)
    num_gauss_points: int = eqx.field(static=True, default=5)

    def _assemble_dynamics_terms(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array, Array]:
        """Return differentiable terms with the production output shapes."""

        inertia = self.scale * jnp.eye(self.num_velocities, dtype=q.dtype) + jnp.outer(
            q, q
        )
        return inertia, self.scale * qd, q + self.scale

    def dynamics_terms(
        self,
        q: Array,
        qd: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> tuple[Array, Array, Array]:
        """Dispatch through the same public boundary as supported systems."""

        return dispatch_dynamics_terms(
            self,
            q,
            qd,
            backend=backend,
            capabilities=GVS_DYNAMICS,
        )


def _probe(
    *, backend: ExecutionBackend = "jax", gauss_points: int = 5
) -> _DispatchProbe:
    """Construct a three-DOF dispatch probe."""

    return _DispatchProbe(
        scale=jnp.asarray(2.0, dtype=jnp.float64),
        backend=backend,
        num_coordinates=3,
        num_velocities=3,
        num_gauss_points=gauss_points,
    )


class _KinematicsProbe(eqx.Module):
    """Minimal planar model exposing scalar and specialized spatial paths."""

    backend: ExecutionBackend = eqx.field(static=True, default="jax")
    num_coordinates: int = eqx.field(static=True, default=3)
    num_velocities: int = eqx.field(static=True, default=3)
    floating_base: bool = eqx.field(static=True, default=False)
    is_planar: bool = eqx.field(static=True, default=True)

    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        """Return a scalar-path marker pose."""

        return jnp.stack((s + q[0], q[1], q[2]))

    def _absolute_forward_kinematics(self, q: Array, s: Array) -> Array:
        """Return the probe's differentiable JAX pose."""

        return self._forward_kinematics(q, s)

    def _forward_kinematics_abscissa_batched(self, q: Array, s: Array) -> Array:
        """Return the same poses through the specialized spatial traversal."""

        return jnp.stack(
            (s + q[0], jnp.full_like(s, q[1]), jnp.full_like(s, q[2])), axis=-1
        )

    def _absolute_forward_kinematics_abscissa_batched(
        self, q: Array, s: Array
    ) -> Array:
        """Return the probe's differentiable JAX pose batch."""

        return self._forward_kinematics_abscissa_batched(q, s)

    def _forward_kinematics_jvp(
        self,
        q: Array,
        s: Array,
        qd: Array | None,
        sd: Array | None,
    ) -> tuple[Array, Array]:
        """Differentiate the probe's scalar pose implementation."""

        qd = jnp.zeros_like(q) if qd is None else qd
        sd = jnp.zeros_like(s) if sd is None else sd
        return jax.jvp(self._forward_kinematics, (q, s), (qd, sd))

    def _jacobian_inertialframe(self, q: Array, s: Array) -> Array:
        """Return a scalar-path inertial Jacobian."""

        del s
        return jnp.eye(3, self.num_velocities, dtype=q.dtype)

    def _absolute_inertial_jacobian(self, q: Array, s: Array) -> Array:
        """Return the probe's differentiable JAX Jacobian."""

        return self._jacobian_inertialframe(q, s)

    def _jacobian_inertialframe_abscissa_batched(self, q: Array, s: Array) -> Array:
        """Return inertial Jacobians through the specialized spatial path."""

        return jnp.broadcast_to(
            self._jacobian_inertialframe(q, s[0]),
            (s.shape[0], 3, self.num_velocities),
        )

    def _absolute_inertial_jacobian_abscissa_batched(self, q: Array, s: Array) -> Array:
        """Return the probe's differentiable JAX Jacobian batch."""

        return self._jacobian_inertialframe_abscissa_batched(q, s)

    def forward_kinematics(
        self,
        q: Array,
        s: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Dispatch probe poses through the production neutral boundary."""

        return dispatch_kinematics(
            self,
            q,
            s,
            operation="pose",
            backend=backend,
            capabilities=PCS_KINEMATICS,
        )

    def jacobian_inertialframe(
        self,
        q: Array,
        s: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Dispatch probe Jacobians through the neutral boundary."""

        return dispatch_kinematics(
            self,
            q,
            s,
            operation="jacobian",
            backend=backend,
            capabilities=PCS_KINEMATICS,
        )

    def forward_kinematics_abscissa_batched(
        self,
        q: Array,
        s: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Dispatch a probe pose batch through the neutral boundary."""

        return dispatch_kinematics_abscissa_batched(
            self,
            q,
            s,
            operation="pose",
            backend=backend,
            capabilities=PCS_KINEMATICS,
        )

    def jacobian_inertialframe_abscissa_batched(
        self,
        q: Array,
        s: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Dispatch a probe Jacobian batch through the neutral boundary."""

        return dispatch_kinematics_abscissa_batched(
            self,
            q,
            s,
            operation="jacobian",
            backend=backend,
            capabilities=PCS_KINEMATICS,
        )


@pytest.mark.parametrize(
    ("q_shape", "qd_shape", "message"),
    [
        ((2,), (2,), "q must have shape"),
        ((1, 2, 3), (1, 2, 3), "q must have shape"),
        ((3,), (1, 3), "qd must have shape"),
    ],
)
def test_dispatch_validates_state_shapes(
    q_shape: tuple[int, ...],
    qd_shape: tuple[int, ...],
    message: str,
) -> None:
    """Reject states that do not follow the scalar-or-one-batch contract."""

    model = _probe()
    with pytest.raises(ValueError, match=message):
        model.dynamics_terms(jnp.zeros(q_shape), jnp.zeros(qd_shape))


def test_dispatch_rejects_unknown_backend() -> None:
    """Validate runtime strings even though the public type is a Literal."""

    model = _probe()
    state = jnp.zeros((model.num_velocities,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="backend must be one of"):
        model.dynamics_terms(state, state, backend="unknown")  # type: ignore[arg-type]


def test_direct_vmap_reaches_one_batched_warp_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map the public scalar API without launching one Warp batch per item."""

    from soromox.systems.execution.warp import loader

    model = _probe(backend="warp")
    q = jnp.arange(12, dtype=jnp.float64).reshape(4, 3) / 10.0
    qd = -0.5 * q
    observed_batches: list[int] = []

    def fake_batch(
        model: _DispatchProbe, q: Array, qd: Array
    ) -> tuple[Array, Array, Array]:
        del qd
        batch_size = q.shape[0]
        jax.debug.callback(
            lambda value: observed_batches.append(int(value)),
            jnp.asarray(batch_size),
        )
        marker = jnp.asarray(batch_size, dtype=q.dtype)
        return (
            jnp.full((batch_size, model.num_velocities, model.num_velocities), marker),
            jnp.full((batch_size, model.num_velocities), marker + 1.0),
            jnp.full((batch_size, model.num_velocities), marker + 2.0),
        )

    monkeypatch.setattr(loader, "_execute_gvs_batch", fake_batch)
    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )

    mapped = jax.vmap(model.dynamics_terms)(q, qd)

    assert observed_batches == [4]
    assert_allclose(mapped[0], jnp.full((4, 3, 3), 4.0), rtol=0.0, atol=0.0)


def test_explicit_batch_reaches_one_batched_warp_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve one executor call for the explicit leading-batch API."""

    from soromox.systems.execution.warp import loader

    model = _probe(backend="warp")
    q = jnp.ones((5, model.num_velocities), dtype=jnp.float64)
    observed_batches: list[int] = []

    def fake_batch(
        model: _DispatchProbe, q: Array, qd: Array
    ) -> tuple[Array, Array, Array]:
        del qd
        batch_size = q.shape[0]
        jax.debug.callback(
            lambda value: observed_batches.append(int(value)),
            jnp.asarray(batch_size),
        )
        return (
            jnp.zeros((batch_size, model.num_velocities, model.num_velocities)),
            jnp.zeros((batch_size, model.num_velocities)),
            jnp.zeros((batch_size, model.num_velocities)),
        )

    monkeypatch.setattr(loader, "_execute_gvs_batch", fake_batch)
    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )

    model.dynamics_terms(q, q)

    assert observed_batches == [5]


def test_explicit_warp_reports_unsupported_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinguish an explicit unsupported request from an automatic fallback."""

    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )
    model = _probe(backend="warp")
    state = jnp.zeros((model.num_velocities,), dtype=jnp.float64)

    with pytest.raises(NotImplementedError, match="not enabled"):
        dispatch_dynamics_terms(
            model,
            state,
            state,
            backend=None,
            capabilities=GVS_DYNAMICS,
            warp_supported=False,
        )


@pytest.mark.parametrize("device", ["cpu", "tpu"])
def test_explicit_warp_reports_unsupported_device(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never silently replace an explicit CUDA-only Warp request with JAX."""

    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: device,
    )
    model = _probe(backend="warp")
    state = jnp.zeros((model.num_velocities,), dtype=jnp.float64)

    with pytest.raises(NotImplementedError, match=f"active {device.upper()} device"):
        dispatch_dynamics_terms(
            model,
            state,
            state,
            backend=None,
            capabilities=PCS_DYNAMICS,
        )


def test_auto_falls_back_for_unsupported_cpu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep CPU fallback exclusive to automatic backend selection."""

    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "cpu",
    )
    model = _probe(backend="auto")
    state = jnp.linspace(-0.1, 0.1, model.num_velocities, dtype=jnp.float64)
    expected = model._assemble_dynamics_terms(state, state)
    actual = dispatch_dynamics_terms(
        model,
        state,
        state,
        backend=None,
        capabilities=PCS_DYNAMICS,
    )

    for actual_term, expected_term in zip(actual, expected, strict=True):
        assert_allclose(actual_term, expected_term)


def test_auto_falls_back_for_unsupported_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep ``auto`` usable when a model instance cannot use Warp."""

    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )
    model = _probe(backend="auto")
    state = jnp.linspace(-0.1, 0.1, model.num_velocities, dtype=jnp.float64)
    expected = model._assemble_dynamics_terms(state, state)
    actual = dispatch_dynamics_terms(
        model,
        state,
        state,
        backend=None,
        capabilities=GVS_DYNAMICS,
        warp_supported=False,
    )

    for actual_term, expected_term in zip(actual, expected, strict=True):
        assert_allclose(actual_term, expected_term)


@pytest.mark.parametrize("gauss_points", [1, 3, 5, 7, 9])
@pytest.mark.parametrize("configured_backend", ["auto", "warp"])
def test_pcs_dispatch_accepts_runtime_quadrature_counts(
    configured_backend: ExecutionBackend,
    gauss_points: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route every positive PCS quadrature count through the Warp executor."""

    from soromox.systems.execution.warp import loader

    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )
    model = _probe(backend=configured_backend, gauss_points=gauss_points)
    state = jnp.zeros((model.num_velocities,), dtype=jnp.float64)

    def fake_batch(model_, q, qd):
        del qd
        batch_size = q.shape[0]
        marker = jnp.asarray(model_.num_gauss_points, dtype=q.dtype)
        return (
            jnp.full(
                (batch_size, model_.num_velocities, model_.num_velocities), marker
            ),
            jnp.full((batch_size, model_.num_velocities), marker),
            jnp.full((batch_size, model_.num_velocities), marker),
        )

    monkeypatch.setattr(loader, "_execute_pcs_batch", fake_batch)
    actual = dispatch_dynamics_terms(
        model,
        state,
        state,
        backend=None,
        capabilities=PCS_DYNAMICS,
    )

    for term in actual:
        assert_allclose(term, gauss_points)


def test_kinematics_jax_spatial_vmap_uses_specialized_model_path() -> None:
    """Route direct and transformed spatial batches to protected batch hooks."""

    class SpecializedProbe(_KinematicsProbe):
        def _forward_kinematics(self, q: Array, s: Array) -> Array:
            del q, s
            return -jnp.ones((3,), dtype=jnp.float64)

        def _forward_kinematics_abscissa_batched(self, q: Array, s: Array) -> Array:
            del q
            return jnp.full((s.shape[0], 3), 7.0, dtype=jnp.float64)

        def _jacobian_inertialframe(self, q: Array, s: Array) -> Array:
            del q, s
            return -jnp.ones((3, 3), dtype=jnp.float64)

        def _jacobian_inertialframe_abscissa_batched(self, q: Array, s: Array) -> Array:
            del q
            return jnp.full((s.shape[0], 3, 3), 11.0, dtype=jnp.float64)

    model = SpecializedProbe()
    q = jnp.zeros((model.num_velocities,), dtype=jnp.float64)
    q_batch = jnp.zeros((4, model.num_velocities), dtype=jnp.float64)
    s = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float64)

    direct_pose = model.forward_kinematics_abscissa_batched(q, s)
    mapped_pose = jax.vmap(lambda value: model.forward_kinematics(q, value))(s)
    direct_jacobian = model.jacobian_inertialframe_abscissa_batched(q, s)
    mapped_jacobian = jax.vmap(lambda value: model.jacobian_inertialframe(q, value))(s)
    cartesian_pose = jax.vmap(
        lambda q_value: jax.vmap(
            lambda s_value: model.forward_kinematics(q_value, s_value)
        )(s)
    )(q_batch)
    cartesian_jacobian = jax.vmap(
        lambda q_value: jax.vmap(
            lambda s_value: model.jacobian_inertialframe(q_value, s_value)
        )(s)
    )(q_batch)

    assert_allclose(direct_pose, jnp.full((5, 3), 7.0))
    assert_allclose(mapped_pose, direct_pose)
    assert_allclose(direct_jacobian, jnp.full((5, 3, 3), 11.0))
    assert_allclose(mapped_jacobian, direct_jacobian)
    assert_allclose(cartesian_pose, jnp.full((4, 5, 3), 7.0))
    assert_allclose(cartesian_jacobian, jnp.full((4, 5, 3, 3), 11.0))


@pytest.mark.parametrize(
    ("q_shape", "s_shape", "message"),
    [
        ((2,), (), "q must have shape"),
        ((1, 1, 3), (), "q must have shape"),
        ((2, 3), (), "q must have shape"),
        ((3,), (3,), "s must be scalar"),
    ],
)
def test_kinematics_scalar_dispatch_validates_shapes(
    q_shape: tuple[int, ...],
    s_shape: tuple[int, ...],
    message: str,
) -> None:
    """Reject batched inputs at the single-point public-method boundary."""

    model = _KinematicsProbe()
    with pytest.raises(ValueError, match=message):
        model.forward_kinematics(jnp.zeros(q_shape), jnp.zeros(s_shape))


@pytest.mark.parametrize(
    ("q_shape", "s_shape", "message"),
    [
        ((2, 3), (4,), "q must have shape"),
        ((3,), (), "s must have shape"),
        ((3,), (2, 4), "s must have shape"),
    ],
)
def test_kinematics_abscissa_dispatch_validates_shapes(
    q_shape: tuple[int, ...],
    s_shape: tuple[int, ...],
    message: str,
) -> None:
    """Keep environment batching outside the explicit abscissa-batch API."""

    model = _KinematicsProbe()
    with pytest.raises(ValueError, match=message):
        model.forward_kinematics_abscissa_batched(
            jnp.zeros(q_shape), jnp.zeros(s_shape)
        )


def test_kinematics_explicit_cartesian_batch_reaches_one_warp_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collapse environment-by-spatial inputs into one canonical Warp call."""

    from soromox.systems.execution.warp import loader

    model = _KinematicsProbe(backend="warp")
    q = jnp.arange(12, dtype=jnp.float64).reshape(4, 3) / 10.0
    s = jnp.linspace(0.0, 1.0, 6, dtype=jnp.float64)

    def fake_batch(
        model: _KinematicsProbe,
        q: Array,
        s: Array,
        operation: str,
    ) -> Array:
        del operation
        marker = 100 * q.shape[0] + s.shape[1]
        return jnp.full((q.shape[0], s.shape[1], 3), marker, dtype=q.dtype)

    monkeypatch.setattr(loader, "_execute_pcs_kinematics_batch", fake_batch)
    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )

    result = jax.vmap(
        model.forward_kinematics_abscissa_batched,
        in_axes=(0, None),
    )(q, s)
    jax.block_until_ready(result)

    assert result.shape == (4, 6, 3)
    assert_allclose(result, 406.0)


@pytest.mark.parametrize(
    ("point_method_name", "batch_method_name", "trailing_shape"),
    [
        ("forward_kinematics", "forward_kinematics_abscissa_batched", (3,)),
        (
            "jacobian_inertialframe",
            "jacobian_inertialframe_abscissa_batched",
            (3, 3),
        ),
    ],
)
def test_public_kinematics_vmaps_reach_batch_shaped_warp_executors(
    monkeypatch: pytest.MonkeyPatch,
    point_method_name: str,
    batch_method_name: str,
    trailing_shape: tuple[int, ...],
) -> None:
    """Fuse pose and Jacobian public-method vmaps across every batch layout."""

    from soromox.systems.execution.warp import loader

    model = _KinematicsProbe(backend="warp")
    q = jnp.arange(12, dtype=jnp.float64).reshape(4, 3) / 10.0
    s = jnp.linspace(0.0, 1.0, 6, dtype=jnp.float64)
    point_method = getattr(model, point_method_name)
    batch_method = getattr(model, batch_method_name)

    def fake_batch(
        model: _KinematicsProbe,
        q: Array,
        s: Array,
        operation: str,
    ) -> Array:
        del model
        marker = 100 * q.shape[0] + s.shape[1]
        result_shape = (q.shape[0], s.shape[1])
        if operation == "pose":
            result_shape = (*result_shape, 3)
        elif operation == "jacobian":
            result_shape = (*result_shape, 3, q.shape[1])
        else:
            raise AssertionError(f"Unexpected operation {operation!r}.")
        return jnp.full(result_shape, marker, dtype=q.dtype)

    monkeypatch.setattr(loader, "_execute_pcs_kinematics_batch", fake_batch)
    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: "gpu",
    )

    spatial = jax.vmap(lambda value: point_method(q[0], value))(s)
    jax.block_until_ready(spatial)
    assert_allclose(spatial, 106.0)

    environments = jax.vmap(lambda value: point_method(value, s[0]))(q)
    jax.block_until_ready(environments)
    assert_allclose(environments, 401.0)

    pairwise = jax.vmap(point_method)(q, s[: q.shape[0]])
    jax.block_until_ready(pairwise)
    assert pairwise.shape == (4, *trailing_shape)
    assert_allclose(pairwise, 401.0)

    non_pairwise = jax.vmap(point_method, in_axes=(0, None))(q, s[0])
    jax.block_until_ready(non_pairwise)
    assert non_pairwise.shape == (4, *trailing_shape)
    assert_allclose(non_pairwise, 401.0)

    explicit_cartesian = jax.vmap(
        batch_method,
        in_axes=(0, None),
    )(q, s)
    jax.block_until_ready(explicit_cartesian)
    assert explicit_cartesian.shape == (4, 6, *trailing_shape)
    assert_allclose(explicit_cartesian, 406.0)

    per_environment_s = jnp.stack([s, s[::-1], s, s[::-1]])
    per_environment = jax.vmap(batch_method)(q, per_environment_s)
    jax.block_until_ready(per_environment)
    assert per_environment.shape == (4, 6, *trailing_shape)
    assert_allclose(per_environment, 406.0)

    nested_cartesian = jax.vmap(
        lambda value: jax.vmap(lambda sample: point_method(value, sample))(s)
    )(q)
    jax.block_until_ready(nested_cartesian)
    assert nested_cartesian.shape == (4, 6, *trailing_shape)
    assert_allclose(nested_cartesian, 406.0)
