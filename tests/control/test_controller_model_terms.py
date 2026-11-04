from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.actuation import ThreadlikeActuator, ThreadlikeFriction, ThreadlikeRouting
from soromox.control import (
    ActuationSpaceFeedforwardCompensationTracker,
    ActuationSpaceGravityCancellationRegulator,
    ActuationSpaceMixedStateFeedbackTracker,
    ActuationSpacePIDController,
    ActuationSpacePotentialCancellationRegulator,
    ActuationSpacePotentialCompensationRegulator,
    ConfigurationSpaceComputedTorqueTracker,
    ConfigurationSpaceFeedforwardCompensationTracker,
    ConfigurationSpaceGravityCancellationRegulator,
    ConfigurationSpaceMixedStateFeedbackTracker,
    ConfigurationSpacePIDController,
    ConfigurationSpacePotentialCancellationRegulator,
    ConfigurationSpacePotentialCompensationRegulator,
    OperationalSpaceImpedanceControlTracker,
    OperationalSpaceSynergisticController,
    PIDControl,
    PIDControllerState,
    ReferenceTrajectory,
)
from soromox.coordinate_transformations import (
    ActuationSpaceDynamics,
    OperationalSpaceDynamics,
)
from soromox.systems import LinkSpec, ModelTerms, PlanarPCS, SystemState
from soromox.utils.geometry import poses


def _robot(tendons: bool, friction: bool = False) -> PlanarPCS:
    link = LinkSpec.circular(
        length=0.1,
        radius=0.01,
        density=1000.0,
        young_modulus=1.0e5,
        shear_modulus=4.0e4,
        material_damping_coefficient=1.0e2,
        reference_strain=[0.0, 1.0, 0.0],
    )
    kwargs = {}
    if tendons:
        routing = ThreadlikeRouting.linear(
            intercept=jnp.array([[0.0, 0.005, 0.0], [0.0, -0.005, 0.0]]),
            start_segment_index=0,
            end_segment_index=(1, 1),
        )
        kwargs["actuators"] = ThreadlikeActuator.tendons(
            routing,
            friction=ThreadlikeFriction.capstan(coefficient=0.8) if friction else None,
        )
    return PlanarPCS.from_links(
        [link] * 2,
        gravity=jnp.array([0.0, -9.81]),
        base_pose=poses.planar_mounting_pose("hanging"),
        **kwargs,
    )


def _pid(num_dofs: int) -> PIDControl:
    return PIDControl(
        Kp=2.0 * jnp.ones((num_dofs,)),
        Ki=0.5 * jnp.ones((num_dofs,)),
        Kd=1.0 * jnp.ones((num_dofs,)),
    )


def _integral(num_dofs: int) -> PIDControllerState:
    return PIDControllerState(integral_error=0.1 * jnp.arange(1.0, num_dofs + 1.0))


def _configuration_reference(num_dofs: int) -> ReferenceTrajectory:
    return ReferenceTrajectory(
        ts=jnp.array([0.0, 1.0]),
        x_des_fn=lambda t: 0.2 * jnp.ones((num_dofs,)),
        xd_des_fn=lambda t: 0.1 * jnp.ones((num_dofs,)),
        xdd_des_fn=lambda t: -0.3 * jnp.ones((num_dofs,)),
    )


def _configuration_space_case(controller_class):
    robot = _robot(tendons=False)
    n = robot.num_internal_dofs
    controller = controller_class(robot, _configuration_reference(n), _pid(n))
    return robot, controller, _integral(n)


def _actuation_space_case(controller_class, friction: bool = False):
    robot = _robot(tendons=True, friction=friction)
    asd = ActuationSpaceDynamics(robot)
    controller = controller_class(
        asd, _configuration_reference(robot.num_internal_dofs), _pid(asd.n_actuated)
    )
    return robot, controller, _integral(asd.n_actuated)


def _capstan_actuation_space_case(controller_class):
    # Capstan friction separates the actuation matrix, which the terms hold, from
    # the transpose of the actuator-coordinate Jacobian, which they do not.
    return _actuation_space_case(controller_class, friction=True)


def _operational_space_case(controller_class):
    # The synergistic controller needs one task coordinate per actuator.
    synergistic = controller_class is OperationalSpaceSynergisticController
    robot = _robot(tendons=synergistic)
    osd = OperationalSpaceDynamics(
        robot=robot,
        s_ps=jnp.array([0.2]) if synergistic else jnp.array([0.15, 0.2]),
        task_selector=jnp.array([False, True, True]),
    )
    q_des = 0.2 * jnp.ones((robot.num_internal_dofs,))
    reference = ReferenceTrajectory(
        ts=jnp.array([0.0, 1.0]),
        x_des_fn=lambda t: osd.operational_space_poses((1.0 + t) * q_des),
        rotation_representation=osd.rotation_representation,
        n_points=osd.n_points,
        is_planar=osd.is_planar,
    )
    if controller_class is OperationalSpaceImpedanceControlTracker:
        return robot, controller_class(osd, reference, K_x=10.0, D_x=2.0), None
    o = osd.n_operational_space
    return robot, controller_class(osd, reference, _pid(o)), _integral(o)


ACTUATION_SPACE_CONTROLLERS = (
    ("actuation-pid", ActuationSpacePIDController),
    ("actuation-feedforward", ActuationSpaceFeedforwardCompensationTracker),
    ("actuation-gravity-cancellation", ActuationSpaceGravityCancellationRegulator),
    ("actuation-mixed-state", ActuationSpaceMixedStateFeedbackTracker),
    (
        "actuation-potential-cancellation",
        ActuationSpacePotentialCancellationRegulator,
    ),
    (
        "actuation-potential-compensation",
        ActuationSpacePotentialCompensationRegulator,
    ),
)
CASES = [
    *(
        pytest.param(_configuration_space_case, controller_class, id=name)
        for name, controller_class in (
            ("configuration-pid", ConfigurationSpacePIDController),
            ("configuration-computed-torque", ConfigurationSpaceComputedTorqueTracker),
            (
                "configuration-feedforward",
                ConfigurationSpaceFeedforwardCompensationTracker,
            ),
            (
                "configuration-gravity-cancellation",
                ConfigurationSpaceGravityCancellationRegulator,
            ),
            ("configuration-mixed-state", ConfigurationSpaceMixedStateFeedbackTracker),
            (
                "configuration-potential-cancellation",
                ConfigurationSpacePotentialCancellationRegulator,
            ),
            (
                "configuration-potential-compensation",
                ConfigurationSpacePotentialCompensationRegulator,
            ),
        )
    ),
    *(
        pytest.param(_actuation_space_case, controller_class, id=name)
        for name, controller_class in ACTUATION_SPACE_CONTROLLERS
    ),
    *(
        pytest.param(
            _capstan_actuation_space_case, controller_class, id=f"{name}-capstan"
        )
        for name, controller_class in ACTUATION_SPACE_CONTROLLERS
    ),
    *(
        pytest.param(_operational_space_case, controller_class, id=name)
        for name, controller_class in (
            ("operational-synergistic", OperationalSpaceSynergisticController),
            ("operational-impedance", OperationalSpaceImpedanceControlTracker),
        )
    ),
]


def _public_terms(robot: PlanarPCS, y: jax.Array) -> ModelTerms:
    """Model terms assembled from the robot's separate public methods, without
    node kinematics, as main's controllers evaluated them."""
    q, qd, _ = robot.split_state(y)
    return ModelTerms(
        inertia_matrix=robot.inertia_matrix(q),
        coriolis_force=robot.coriolis_matrix(q, qd) @ qd,
        gravitational_force=robot.gravitational_force(q),
        elastic_force=robot.elastic_force(q),
        damping_matrix=robot.damping_matrix(q),
        actuation_matrix=robot.actuation_matrix(q),
    )


def _state(robot: PlanarPCS, control_state) -> SystemState:
    y = 0.3 * jax.random.normal(jax.random.PRNGKey(0), (robot.state_size,))
    return SystemState(
        t=jnp.array(0.3), y=robot.project_state(y), control_state=control_state
    )


def _assert_leaves_close(actual, desired):
    for actual_leaf, desired_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(desired), strict=True
    ):
        assert_allclose(
            actual_leaf,
            desired_leaf,
            rtol=1e-9,
            atol=1e-9 * jnp.max(jnp.abs(desired_leaf)),
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(("build_case", "controller_class"), CASES)
def test_controller_matches_with_given_model_terms(build_case, controller_class):
    robot, controller, control_state = build_case(controller_class)
    state = _state(robot, control_state)

    own = controller(state)
    # The terms of the simulated system, as a rollout with share_model_terms=True
    # passes them, and the terms from the separate model methods.
    given = controller(replace(state, model_terms=robot.model_terms(state.t, state.y)))
    public = controller(replace(state, model_terms=_public_terms(robot, state.y)))

    _assert_leaves_close(given, own)
    _assert_leaves_close(public, own)


def test_synergistic_controller_matches_explicit_projection():
    robot, controller, control_state = _operational_space_case(
        OperationalSpaceSynergisticController
    )
    state = _state(robot, control_state)

    u, _ = controller(state)

    # u = (J M^-1 A)^-1 J M^-1 J^T f with explicit inverses
    q, _, _ = robot.split_state(state.y)
    J = controller.operational_space_dynamics.jacobian(q)
    f, _ = controller.error_based_feedback_term(state)
    M_inv = jnp.linalg.inv(robot.inertia_matrix(q))
    A = robot.actuation_matrix(q)
    expected = jnp.linalg.inv(J @ M_inv @ A) @ J @ M_inv @ J.T @ f
    _assert_leaves_close(u, expected)
