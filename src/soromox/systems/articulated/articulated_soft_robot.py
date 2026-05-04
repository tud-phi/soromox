__all__ = ["ArticulatedSoftRobot"]

from typing import Any

import equinox as eqx
import jax
from jax import Array, lax, vmap
from jax import numpy as jnp

import soromox.utils.lie_algebra as lie
from soromox.systems.articulated.params import ArticulatedSoftRobotParams
from soromox.systems.soft_robot import CrossSectionGeometry, SoftRobot


class ArticulatedSoftRobot(SoftRobot):
    """
    Spatial serial chain with rigid links and optional compliant joints.

    This system models an open kinematic chain whose links are rigid and whose
    one-degree-of-freedom joints are described by screw axes. It follows the
    same `SoftRobot` interface as continuum models, so points can be queried by
    arc length and the standard configuration-space dynamics methods are
    available.

    Notes
    -----
    The generalized coordinates `q` are the scalar joint coordinates. Each
    joint has a screw axis `joint_screw[i] = [omega, v]` expressed in the
    joint frame before the joint motion is applied. The post-joint link frame is
    connected to the next link through the fixed tip vector `p_tip[i]`.

    Dynamics are implemented through two complementary algorithmic paths. The
    public matrix-valued methods use a dense Jacobian-energy formulation:
    link COM Jacobians map generalized velocities to spatial twists, kinetic
    energy is assembled as `T = 0.5 * sum_i V_i.T @ I_i @ V_i`, and the dense
    inertia matrix follows from the quadratic form in `qd`. Gravity is computed
    as the gradient of gravitational potential energy, while the Coriolis matrix
    is assembled from Christoffel symbols of `M(q)`. This path is intentionally
    explicit and dense because model-based controllers, coordinate
    transformations, and diagnostics often need `M(q)`, `C(q, qd)`, `G(q)`,
    elastic forces, and damping as standalone arrays. Readers interested in the
    Lagrangian formulation and dense direct-dynamics construction should start
    with Hollerbach (1980) and Walker & Orin (1982).

    Forward dynamics uses an articulated-body algorithm (ABA) instead of solving
    the dense equations at every call. ABA performs a forward velocity pass, a
    backward articulated-inertia pass, and a final forward acceleration pass to
    compute `qdd` directly in linear time for a serial chain. The recursion uses
    spatial inertias, motion transforms, and joint motion subspaces, and includes
    gravity, actuation, external generalized forces, joint elasticity, and joint
    damping. The implementation uses `jax.lax.scan` for the serial recurrences
    so JAX does not unroll Python loops as the number of links grows. Readers
    interested in the articulated-body recursion should start with Featherstone
    (1983).

    Both paths represent the same equations of motion:
    `M(q) qdd + C(q, qd) qd + G(q) + elastic_force(q) + D(q) qd = tau`.
    Tests validate the ABA result against the dense solve. Keeping the dense
    Jacobian-energy assembly separate from ABA makes the controller-facing API
    easy to inspect while preserving a faster default `forward_dynamics` path.

    References:
        Hollerbach, J. M. (1980). A Recursive Lagrangian Formulation of
        Manipulator Dynamics and a Comparative Study of Dynamics Formulation
        Complexity. IEEE Transactions on Systems, Man, and Cybernetics,
        SMC-10(11), 730-736.

        Walker, M. W., & Orin, D. E. (1982). Efficient Dynamic Computer
        Simulation of Robotic Mechanisms. Journal of Dynamic Systems,
        Measurement, and Control, 104(3), 205-211.
        https://doi.org/10.1115/1.3139699

        Featherstone, R. (1983). The Calculation of Robot Dynamics Using
        Articulated-Body Inertias. The International Journal of Robotics
        Research, 2(1), 13-30.
        https://doi.org/10.1177/027836498300200102

    Attributes:
        num_links: Number of links and joints.
        joint_screw: Joint screw axis array, shape `(num_links, 6)`.
        g_parent_joint: Fixed transforms from previous link tip to joint frame,
            shape `(num_links, 4, 4)`.
        p_tip: Link tip vectors in post-joint link frames, shape `(num_links, 3)`.
        p_com: Link COM vectors in post-joint link frames, shape `(num_links, 3)`.
        m: Link masses, shape `(num_links,)`.
        I_com: Link inertia matrices about each COM, expressed in the post-joint
            link frame, shape `(num_links, 3, 3)`.
        g: Gravity acceleration vector in the base frame, shape `(3,)`.
        K: Joint stiffness matrix, shape `(num_links, num_links)`.
        D: Joint damping matrix, shape `(num_links, num_links)`.
        q_ref_k: Rest configuration for the joint springs, shape `(num_links,)`.
        r: Circular visualization radii, shape `(num_links,)`.
    """

    num_links: int = eqx.field(static=True)

    params: ArticulatedSoftRobotParams
    joint_screw: Array
    g_parent_joint: Array
    p_tip: Array
    p_com: Array
    m: Array
    I_com: Array
    g: Array
    K: Array
    D: Array
    q_ref_k: Array
    r: Array

    def __init__(self, params: ArticulatedSoftRobotParams, **kwargs: Any) -> None:
        """Initialize an articulated soft robot from typed dynamic parameters."""
        super().__init__(**kwargs)
        if not isinstance(params, ArticulatedSoftRobotParams):
            raise TypeError("params must be an ArticulatedSoftRobotParams instance.")
        params.validate()
        self.params = params

        joint_screw = jnp.asarray(params.joint_screw)
        p_tip = jnp.asarray(params.tip_position)
        p_com = jnp.asarray(params.center_of_mass_position)
        m = jnp.asarray(params.mass)
        I_com = jnp.asarray(params.center_of_mass_inertia)
        g = jnp.asarray(params.gravity)

        if joint_screw.ndim != 2 or joint_screw.shape[1] != 6:
            raise ValueError("joint_screw must have shape (N, 6).")
        n = joint_screw.shape[0]
        expected_vec = (n, 3)
        if p_tip.shape != expected_vec:
            raise ValueError(f"tip_position must have shape {expected_vec}.")
        if p_com.shape != expected_vec:
            raise ValueError(f"center_of_mass_position must have shape {expected_vec}.")
        if m.shape != (n,):
            raise ValueError(f"mass must have shape {(n,)}.")
        if I_com.shape != (n, 3, 3):
            raise ValueError(f"center_of_mass_inertia must have shape {(n, 3, 3)}.")
        if g.shape != (3,):
            raise ValueError("gravity must have shape (3,).")

        g_parent_joint = jnp.asarray(params.parent_to_joint_transform)
        if g_parent_joint.shape != (n, 4, 4):
            raise ValueError(f"parent_to_joint_transform must have shape {(n, 4, 4)}.")

        self.num_links = int(n)
        self.num_dofs = self.num_links
        self.num_actuators = self.num_links

        self.joint_screw = joint_screw
        self.g_parent_joint = g_parent_joint
        self.p_tip = p_tip
        self.p_com = p_com
        self.m = m
        self.I_com = I_com
        self.g = g

        self.K = jnp.asarray(params.joint_stiffness)
        self.D = jnp.asarray(params.joint_damping)
        self.q_ref_k = jnp.asarray(params.joint_rest_configuration)
        self.r = jnp.asarray(params.radius)

        if self.K.shape != (n, n):
            raise ValueError(f"joint_stiffness must have shape {(n, n)}.")
        if self.D.shape != (n, n):
            raise ValueError(f"joint_damping must have shape {(n, n)}.")
        if self.q_ref_k.shape != (n,):
            raise ValueError(f"joint_rest_configuration must have shape {(n,)}.")
        if self.r.shape != (n,):
            raise ValueError(f"radius must have shape {(n,)}.")

    @property
    def is_planar(self) -> bool:
        """Return False because this is a spatial SE(3) system."""
        return False

    @property
    def length(self) -> Array:
        """Total chain length measured along the straight link centerlines."""
        return jnp.sum(self.segment_length)

    @property
    def segment_length(self) -> Array:
        """Per-link centerline lengths."""
        return jnp.linalg.norm(self.p_tip, axis=1)

    @property
    def L_cum(self) -> Array:
        """Cumulative link lengths `[0, L_0, ..., sum_i L_i]`."""
        return jnp.concatenate(
            [jnp.zeros(1, dtype=self.p_tip.dtype), jnp.cumsum(self.segment_length)]
        )

    @property
    def total_length(self) -> Array:
        """Alias for `length` used by other articulated systems."""
        return self.length

    @staticmethod
    def _translation(p: Array) -> Array:
        """Return an SE(3) transform with identity rotation and translation `p`."""
        return jnp.eye(4, dtype=p.dtype).at[:3, 3].set(p)

    @staticmethod
    def _motion_cross(v: Array) -> Array:
        """Return the spatial motion cross-product matrix."""
        return lie.adjoint_se3(v)

    @staticmethod
    def _force_cross(v: Array) -> Array:
        """Return the spatial force cross-product matrix."""
        return -lie.adjoint_se3(v).T

    def _joint_motion(self, screw: Array, q_i: Array) -> Array:
        """Return the joint transform generated by a screw coordinate."""
        return lie.exp_gn_SE3(screw * q_i, eps=self.global_eps)

    def _spatial_inertia(self, m: Array, I_com: Array, p_com: Array) -> Array:
        """Return spatial inertia about the post-joint link-frame origin."""
        p_tilde = lie.tilde_SE3(p_com)
        return jnp.block(
            [
                [I_com + m * p_tilde.T @ p_tilde, m * p_tilde],
                [-m * p_tilde, m * jnp.eye(3, dtype=I_com.dtype)],
            ]
        )

    def _spatial_inertias(self) -> Array:
        """Return link spatial inertias in their post-joint link frames."""
        return vmap(self._spatial_inertia)(self.m, self.I_com, self.p_com)

    def _link_motion_transforms(self, q: Array) -> Array:
        """
        Return parent-to-child motion transforms for the articulated chain.

        The returned matrices map parent spatial motion vectors into the current
        link frame.
        """
        parent_offsets = jnp.concatenate(
            [jnp.zeros((1, 3), dtype=q.dtype), self.p_tip[:-1]], axis=0
        )
        fixed_transforms = vmap(
            lambda p, g_parent_joint: self._translation(p) @ g_parent_joint
        )(parent_offsets, self.g_parent_joint)
        joint_transforms = vmap(self._joint_motion)(self.joint_screw, q)
        g_parent_child = fixed_transforms @ joint_transforms
        return vmap(lie.Adjoint_g_inv_SE3)(g_parent_child)

    def _kinematic_frames(self, q: Array) -> tuple[Array, Array, Array, Array]:
        """
        Compute joint, link, COM, and tip frames for all links.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Tuple `(g_joints, g_links, g_coms, g_tips)`, each with shape
            `(num_links, 4, 4)`.
        """

        def _step(
            g_parent_tip: Array, inputs: tuple[Array, Array, Array, Array, Array]
        ):
            g_parent_joint_i, screw_i, q_i, p_com_i, p_tip_i = inputs
            g_joint = g_parent_tip @ g_parent_joint_i
            g_link = g_joint @ self._joint_motion(screw_i, q_i)
            g_com = g_link @ self._translation(p_com_i)
            g_tip = g_link @ self._translation(p_tip_i)
            return g_tip, (g_joint, g_link, g_com, g_tip)

        _, frames = lax.scan(
            _step,
            jnp.eye(4, dtype=q.dtype),
            (self.g_parent_joint, self.joint_screw, q, self.p_com, self.p_tip),
        )
        return frames

    def _world_joint_screws(self, q: Array) -> Array:
        """Return joint screw axes expressed in the base frame."""
        g_joints, _, _, _ = self._kinematic_frames(q)
        return self._world_joint_screws_from_joint_frames(g_joints)

    def _world_joint_screws_from_joint_frames(self, g_joints: Array) -> Array:
        """Return base-frame screw axes from precomputed joint frames."""
        return vmap(lambda g, s: lie.Adjoint_g_SE3(g) @ s)(g_joints, self.joint_screw)

    def _jacobian_for_point(self, q: Array, link_idx: Array, p_world: Array) -> Array:
        """Compute the spatial Jacobian for a point on `link_idx`."""
        screws = self._world_joint_screws(q)
        return self._jacobian_for_point_from_screws(screws, link_idx, p_world)

    def _jacobian_for_point_from_screws(
        self, screws: Array, link_idx: Array, p_world: Array
    ) -> Array:
        """Compute a point Jacobian using precomputed world-frame screws."""
        idxs = jnp.arange(self.num_links)
        mask = (idxs <= link_idx).astype(screws.dtype)
        omega = screws[:, :3]
        v_origin = screws[:, 3:]
        v_point = v_origin + jnp.cross(omega, p_world[None, :])
        J_cols = jnp.concatenate([omega, v_point], axis=1) * mask[:, None]
        return J_cols.T

    def _jacobian_for_points(
        self, screws: Array, link_idxs: Array, p_worlds: Array
    ) -> Array:
        """Compute point Jacobians using precomputed world-frame screws."""
        return vmap(lambda i, p: self._jacobian_for_point_from_screws(screws, i, p))(
            link_idxs, p_worlds
        )

    def _world_joint_screw_derivatives(self, screws: Array, qd: Array) -> Array:
        """Return time derivatives of world-frame joint screw axes."""

        def _step(eta_parent: Array, inputs: tuple[Array, Array]):
            screw_i, qd_i = inputs
            screw_dot_i = lie.adjoint_se3(eta_parent) @ screw_i
            eta_next = eta_parent + screw_i * qd_i
            return eta_next, screw_dot_i

        _, screw_dots = lax.scan(
            _step,
            jnp.zeros((6,), dtype=screws.dtype),
            (screws, qd),
        )
        return screw_dots

    def _jacobian_and_time_derivative_for_point_from_screws(
        self,
        screws: Array,
        screw_dots: Array,
        link_idx: Array,
        p_world: Array,
        qd: Array,
    ) -> tuple[Array, Array]:
        """Compute a point Jacobian and derivative from world-frame screw data."""
        idxs = jnp.arange(self.num_links)
        mask = (idxs <= link_idx).astype(screws.dtype)

        omega = screws[:, :3]
        v_origin = screws[:, 3:]
        v_point = v_origin + jnp.cross(omega, p_world[None, :])
        J_cols = jnp.concatenate([omega, v_point], axis=1) * mask[:, None]
        J = J_cols.T

        p_world_dot = J[3:, :] @ qd
        omega_dot = screw_dots[:, :3]
        v_origin_dot = screw_dots[:, 3:]
        v_point_dot = (
            v_origin_dot
            + jnp.cross(omega_dot, p_world[None, :])
            + jnp.cross(omega, p_world_dot[None, :])
        )
        Jd_cols = jnp.concatenate([omega_dot, v_point_dot], axis=1) * mask[:, None]
        Jd = Jd_cols.T

        return J, Jd

    def _jacobian_and_time_derivative_for_points_from_screws(
        self,
        screws: Array,
        screw_dots: Array,
        link_idxs: Array,
        p_worlds: Array,
        qd: Array,
    ) -> tuple[Array, Array]:
        """Compute point Jacobians and derivatives from world-frame screw data."""
        return vmap(
            lambda i, p: self._jacobian_and_time_derivative_for_point_from_screws(
                screws, screw_dots, i, p, qd
            )
        )(link_idxs, p_worlds)

    def classify_segment(self, s: Array) -> tuple[Array, Array]:
        """
        Determine which link contains an arc-length position.

        Args:
            s: Arc-length position along the serial chain.

        Returns:
            Tuple `(link_idx, s_local)` with the zero-based link index and local
            arc length within that link.
        """
        L_cum = self.L_cum
        link_idx = jnp.searchsorted(L_cum[1:], s, side="left")
        link_idx = jnp.clip(link_idx, 0, self.num_links - 1)
        s_local = s - L_cum[link_idx]
        return link_idx, s_local

    def cross_section_geometry(self, q: Array, s: Array) -> tuple[Array, Array]:
        """Return circular cross-section metadata for visualization."""
        link_idx, _ = self.classify_segment(s)
        tag = jnp.asarray(CrossSectionGeometry.CIRCULAR, dtype=jnp.int32)
        return tag, jnp.array([self.r[link_idx]])

    @eqx.filter_jit
    def forward_kinematics_joints(self, q: Array) -> Array:
        """
        Compute SE(3) frames at all joint origins.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Joint frames, shape `(num_links, 4, 4)`.
        """
        g_joints, _, _, _ = self._kinematic_frames(q)
        return g_joints

    @eqx.filter_jit
    def forward_kinematics_coms(self, q: Array) -> Array:
        """
        Compute SE(3) frames at all link centers of mass.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            COM frames, shape `(num_links, 4, 4)`.
        """
        _, _, g_coms, _ = self._kinematic_frames(q)
        return g_coms

    @eqx.filter_jit
    def forward_kinematics_tips(self, q: Array) -> Array:
        """
        Compute SE(3) frames at all link tips.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Tip frames, shape `(num_links, 4, 4)`.
        """
        _, _, _, g_tips = self._kinematic_frames(q)
        return g_tips

    @eqx.filter_jit
    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        """
        Compute the SE(3) frame at an arc-length position.

        Args:
            q: Joint coordinates, shape `(num_links,)`.
            s: Arc-length position along the chain.

        Returns:
            SE(3) frame at `s`, shape `(4, 4)`.
        """
        link_idx, s_local = self.classify_segment(s)
        _, g_links, _, _ = self._kinematic_frames(q)
        L_i = self.segment_length[link_idx]
        safe_L_i = jnp.where(L_i > 0, L_i, jnp.ones_like(L_i))
        direction = jnp.where(
            L_i > 0,
            self.p_tip[link_idx] / safe_L_i,
            jnp.zeros_like(self.p_tip[link_idx]),
        )
        displacement = jnp.where(
            L_i > 0,
            direction * s_local,
            jnp.zeros_like(direction),
        )
        return g_links[link_idx] @ self._translation(displacement)

    @eqx.filter_jit
    def jacobian_coms(self, q: Array) -> Array:
        """
        Compute spatial Jacobians at all link COMs.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            COM Jacobians, shape `(num_links, 6, num_links)`.
        """
        g_joints, _, g_coms, _ = self._kinematic_frames(q)
        screws = self._world_joint_screws_from_joint_frames(g_joints)
        idxs = jnp.arange(self.num_links)
        positions = g_coms[:, :3, 3]
        return self._jacobian_for_points(screws, idxs, positions)

    @eqx.filter_jit
    def jacobian_tips(self, q: Array) -> Array:
        """
        Compute spatial Jacobians at all link tips.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Tip Jacobians, shape `(num_links, 6, num_links)`.
        """
        g_joints, _, _, g_tips = self._kinematic_frames(q)
        screws = self._world_joint_screws_from_joint_frames(g_joints)
        idxs = jnp.arange(self.num_links)
        positions = g_tips[:, :3, 3]
        return self._jacobian_for_points(screws, idxs, positions)

    @eqx.filter_jit
    def jacobian_joints(self, q: Array) -> Array:
        """
        Compute spatial Jacobians at all joint origins.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Joint-origin Jacobians, shape `(num_links, 6, num_links)`.
        """
        g_joints, _, _, _ = self._kinematic_frames(q)
        screws = self._world_joint_screws_from_joint_frames(g_joints)
        idxs = jnp.arange(self.num_links)
        positions = g_joints[:, :3, 3]
        return self._jacobian_for_points(screws, idxs - 1, positions)

    @eqx.filter_jit
    def _jacobian(self, q: Array, s: Array) -> Array:
        """
        Compute the spatial Jacobian at an arc-length position.

        Args:
            q: Joint coordinates, shape `(num_links,)`.
            s: Arc-length position along the chain.

        Returns:
            Spatial Jacobian, shape `(6, num_links)`.
        """
        link_idx, s_local = self.classify_segment(s)
        g_joints, g_links, _, _ = self._kinematic_frames(q)
        L_i = self.segment_length[link_idx]
        safe_L_i = jnp.where(L_i > 0, L_i, jnp.ones_like(L_i))
        direction = jnp.where(
            L_i > 0,
            self.p_tip[link_idx] / safe_L_i,
            jnp.zeros_like(self.p_tip[link_idx]),
        )
        displacement = jnp.where(
            L_i > 0,
            direction * s_local,
            jnp.zeros_like(direction),
        )
        g_s = g_links[link_idx] @ self._translation(displacement)
        screws = self._world_joint_screws_from_joint_frames(g_joints)
        return self._jacobian_for_point_from_screws(screws, link_idx, g_s[:3, 3])

    @eqx.filter_jit
    def jacobian_and_time_derivatives_tips(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array]:
        """
        Compute tip Jacobians and their time derivatives.

        Args:
            q: Joint coordinates, shape `(num_links,)`.
            qd: Joint velocities, shape `(num_links,)`.

        Returns:
            Tuple `(J, Jd)`, both with shape `(num_links, 6, num_links)`.
        """
        g_joints, _, _, g_tips = self._kinematic_frames(q)
        screws = self._world_joint_screws_from_joint_frames(g_joints)
        screw_dots = self._world_joint_screw_derivatives(screws, qd)
        idxs = jnp.arange(self.num_links)
        positions = g_tips[:, :3, 3]
        return self._jacobian_and_time_derivative_for_points_from_screws(
            screws, screw_dots, idxs, positions, qd
        )

    @eqx.filter_jit
    def _jacobian_and_time_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and time derivative at an arc-length position.

        Args:
            q: Joint coordinates, shape `(num_links,)`.
            qd: Joint velocities, shape `(num_links,)`.
            s: Arc-length position along the chain.

        Returns:
            Tuple `(J, Jd)`, both with shape `(6, num_links)`.
        """
        link_idx, s_local = self.classify_segment(s)
        g_joints, g_links, _, _ = self._kinematic_frames(q)
        L_i = self.segment_length[link_idx]
        safe_L_i = jnp.where(L_i > 0, L_i, jnp.ones_like(L_i))
        direction = jnp.where(
            L_i > 0,
            self.p_tip[link_idx] / safe_L_i,
            jnp.zeros_like(self.p_tip[link_idx]),
        )
        displacement = jnp.where(
            L_i > 0,
            direction * s_local,
            jnp.zeros_like(direction),
        )
        g_s = g_links[link_idx] @ self._translation(displacement)
        screws = self._world_joint_screws_from_joint_frames(g_joints)
        screw_dots = self._world_joint_screw_derivatives(screws, qd)
        return self._jacobian_and_time_derivative_for_point_from_screws(
            screws, screw_dots, link_idx, g_s[:3, 3], qd
        )

    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
        """
        Compute the dense generalized inertia matrix.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Inertia matrix, shape `(num_links, num_links)`.
        """
        g_joints, g_links, g_coms, _ = self._kinematic_frames(q)
        screws = self._world_joint_screws_from_joint_frames(g_joints)
        idxs = jnp.arange(self.num_links)
        positions = g_coms[:, :3, 3]
        J_coms = self._jacobian_for_points(screws, idxs, positions)
        R_links = g_links[:, :3, :3]
        I_world = jnp.einsum("nij,njk,nlk->nil", R_links, self.I_com, R_links)
        Jw = J_coms[:, :3, :]
        Jv = J_coms[:, 3:, :]
        angular = jnp.einsum("nia,nij,njb->ab", Jw, I_world, Jw)
        linear = jnp.einsum("n,nia,nib->ab", self.m, Jv, Jv)
        return angular + linear

    @eqx.filter_jit
    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        """
        Compute a Christoffel-consistent dense Coriolis matrix.

        Args:
            q: Joint coordinates, shape `(num_links,)`.
            qd: Joint velocities, shape `(num_links,)`.

        Returns:
            Coriolis matrix, shape `(num_links, num_links)`.
        """
        dM = jax.jacfwd(self.inertia_matrix)(q)
        christoffel = dM + jnp.swapaxes(dM, 1, 2) - jnp.transpose(dM, (2, 0, 1))
        return 0.5 * jnp.einsum("ijk,k->ij", christoffel, qd)

    @eqx.filter_jit
    def _gravitational_energy(self, q: Array) -> Array:
        """
        Compute gravitational potential energy.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Gravitational potential energy as a scalar.
        """
        g_coms = self.forward_kinematics_coms(q)
        p_coms = g_coms[:, :3, 3]
        return -jnp.sum(self.m * (p_coms @ self.g))

    @eqx.filter_jit
    def _gravitational_force(self, q: Array) -> Array:
        """
        Compute generalized gravitational forces.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Gravity vector, shape `(num_links,)`.
        """
        return jax.grad(self._gravitational_energy)(q)

    @eqx.filter_jit
    def stiffness_matrix(self) -> Array:
        """
        Return the linear joint stiffness matrix.

        Returns:
            Stiffness matrix, shape `(num_links, num_links)`.
        """
        return self.K

    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """
        Compute generalized elastic joint forces.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Elastic force vector, shape `(num_links,)`.
        """
        return self.K @ (q - self.q_ref_k)

    @eqx.filter_jit
    def _elastic_energy(self, q: Array) -> Array:
        """
        Compute elastic potential energy stored in joint springs.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Elastic potential energy as a scalar.
        """
        dq = q - self.q_ref_k
        return 0.5 * dq @ self.K @ dq

    @eqx.filter_jit
    def damping_matrix(self, q: Array) -> Array:
        """
        Return the viscous joint damping matrix.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Damping matrix, shape `(num_links, num_links)`.
        """
        return self.D

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Return the direct joint actuation matrix.

        Args:
            q: Joint coordinates, shape `(num_links,)`.

        Returns:
            Identity actuation matrix, shape `(num_links, num_links)`.
        """
        return jnp.eye(self.num_links, dtype=q.dtype)

    def actuation_force(self, q: Array, u: Array) -> Array:
        """
        Map joint inputs to generalized forces.

        Args:
            q: Joint coordinates, shape `(num_links,)`.
            u: Joint actuation inputs, shape `(num_links,)`.

        Returns:
            Generalized actuation force, shape `(num_links,)`.
        """
        return self.actuation_matrix(q) @ u

    def _aba_forward_accelerations(self, q: Array, qd: Array, tau: Array) -> Array:
        """
        Compute joint accelerations with the articulated-body algorithm.

        Args:
            q: Joint coordinates, shape `(num_links,)`.
            qd: Joint velocities, shape `(num_links,)`.
            tau: Total applied generalized forces after actuation and external
                forces, shape `(num_links,)`.

        Returns:
            Joint accelerations, shape `(num_links,)`.
        """
        tau_el = self.elastic_force(q)
        D = self.damping_matrix(q)
        tau_total = tau - tau_el - D @ qd

        Xup = self._link_motion_transforms(q)
        spatial_inertias = self._spatial_inertias()
        S = self.joint_screw

        def _velocity_step(
            v_parent: Array, inputs: tuple[Array, Array, Array, Array]
        ) -> tuple[Array, tuple[Array, Array, Array]]:
            Xup_i, S_i, inertia_i, qd_i = inputs
            vJ = S_i * qd_i
            v_i = Xup_i @ v_parent + vJ
            c_i = self._motion_cross(v_i) @ vJ
            pA_i = self._force_cross(v_i) @ inertia_i @ v_i
            return v_i, (v_i, c_i, pA_i)

        _, (_, c, pA_base) = lax.scan(
            _velocity_step,
            jnp.zeros((6,), dtype=q.dtype),
            (Xup, S, spatial_inertias, qd),
        )

        def _backward_step(
            carry: tuple[Array, Array],
            inputs: tuple[Array, Array, Array, Array, Array, Array],
        ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
            IA_child, pA_child = carry
            Xup_i, S_i, inertia_i, pA_base_i, c_i, tau_i = inputs
            IA_i = inertia_i + IA_child
            pA_i = pA_base_i + pA_child
            U_i = IA_i @ S_i
            d_i = S_i @ U_i
            u_i = tau_i - S_i @ pA_i
            Ia_i = IA_i - jnp.outer(U_i, U_i) / d_i
            pa_i = pA_i + Ia_i @ c_i + U_i * (u_i / d_i)
            IA_parent = Xup_i.T @ Ia_i @ Xup_i
            pA_parent = Xup_i.T @ pa_i
            return (IA_parent, pA_parent), (U_i, d_i, u_i)

        _, (U_rev, d_rev, u_rev) = lax.scan(
            _backward_step,
            (
                jnp.zeros((6, 6), dtype=q.dtype),
                jnp.zeros((6,), dtype=q.dtype),
            ),
            (
                Xup[::-1],
                S[::-1],
                spatial_inertias[::-1],
                pA_base[::-1],
                c[::-1],
                tau_total[::-1],
            ),
        )
        U = U_rev[::-1]
        d = d_rev[::-1]
        u = u_rev[::-1]

        def _acceleration_step(
            a_parent: Array, inputs: tuple[Array, Array, Array, Array, Array, Array]
        ) -> tuple[Array, Array]:
            Xup_i, S_i, c_i, U_i, d_i, u_i = inputs
            a_i = Xup_i @ a_parent + c_i
            qdd_i = (u_i - U_i @ a_i) / d_i
            return a_i + S_i * qdd_i, qdd_i

        _, qdd = lax.scan(
            _acceleration_step,
            jnp.concatenate([jnp.zeros(3, dtype=q.dtype), -self.g]),
            (Xup, S, c, U, d, u),
        )

        return qdd

    @eqx.filter_jit
    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: tuple | None = None
    ) -> Array:
        """
        Compute state-space forward dynamics.

        Args:
            t: Current time, shape `()`. The model is autonomous, so this value is
                unused.
            y: State vector `[q, qd]`, shape `(2 * num_links,)`.
            actuation_args: Optional actuation tuple:
                - `None`: zero actuation and zero external force.
                - `(u,)`: joint inputs only.
                - `(u, tau_ext)`: joint inputs and external generalized forces.

        Returns:
            State derivative `[qd, qdd]`, shape `(2 * num_links,)`.

        Raises:
            ValueError: If `actuation_args` has an unsupported length.
        """
        del t
        q, qd = jnp.split(y, 2)

        if actuation_args is None:
            u, tau_ext = None, None
        elif len(actuation_args) == 1:
            u, tau_ext = actuation_args[0], None
        elif len(actuation_args) == 2:
            u, tau_ext = actuation_args
        else:
            raise ValueError("actuation_args must be None, (u,), or (u, tau_ext).")

        if u is None:
            u = jnp.zeros((self.num_actuators,), dtype=y.dtype)
        if tau_ext is None:
            tau_ext = jnp.zeros((self.num_dofs,), dtype=y.dtype)

        tau = self.actuation_force(q, u) + tau_ext
        qdd = self._aba_forward_accelerations(q, qd, tau)
        return jnp.concatenate([qd, qdd])

    def _validated_param_arrays(
        self, params: ArticulatedSoftRobotParams
    ) -> tuple[Array, ...]:
        """Coerce typed parameters and enforce the fixed serial-chain layout."""
        arrays = (
            jnp.asarray(params.joint_screw),
            jnp.asarray(params.parent_to_joint_transform),
            jnp.asarray(params.tip_position),
            jnp.asarray(params.center_of_mass_position),
            jnp.asarray(params.mass),
            jnp.asarray(params.center_of_mass_inertia),
            jnp.asarray(params.gravity),
            jnp.asarray(params.joint_stiffness),
            jnp.asarray(params.joint_damping),
            jnp.asarray(params.joint_rest_configuration),
            jnp.asarray(params.radius),
        )
        expected_shapes = (
            (self.num_links, 6),
            (self.num_links, 4, 4),
            (self.num_links, 3),
            (self.num_links, 3),
            (self.num_links,),
            (self.num_links, 3, 3),
            (3,),
            (self.num_dofs, self.num_dofs),
            (self.num_dofs, self.num_dofs),
            (self.num_dofs,),
            (self.num_links,),
        )
        names = (
            "joint_screw",
            "parent_to_joint_transform",
            "tip_position",
            "center_of_mass_position",
            "mass",
            "center_of_mass_inertia",
            "gravity",
            "joint_stiffness",
            "joint_damping",
            "joint_rest_configuration",
            "radius",
        )
        for name, value, expected_shape in zip(names, arrays, expected_shapes):
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )
        return arrays

    def with_params(self, params: ArticulatedSoftRobotParams) -> "ArticulatedSoftRobot":
        """Return an updated copy with a full typed parameter object."""
        if not isinstance(params, ArticulatedSoftRobotParams):
            raise TypeError("params must be an ArticulatedSoftRobotParams instance.")
        params.validate()
        if params.mass.shape != self.params.mass.shape:
            raise ValueError(
                "mass shape changes the model structure; construct a new ArticulatedSoftRobot."
            )
        (
            joint_screw,
            parent_to_joint_transform,
            tip_position,
            center_of_mass_position,
            mass,
            center_of_mass_inertia,
            gravity,
            joint_stiffness,
            joint_damping,
            joint_rest_configuration,
            radius,
        ) = self._validated_param_arrays(params)
        return eqx.tree_at(
            lambda model: (
                model.params,
                model.joint_screw,
                model.g_parent_joint,
                model.p_tip,
                model.p_com,
                model.m,
                model.I_com,
                model.g,
                model.K,
                model.D,
                model.q_ref_k,
                model.r,
            ),
            self,
            (
                params,
                joint_screw,
                parent_to_joint_transform,
                tip_position,
                center_of_mass_position,
                mass,
                center_of_mass_inertia,
                gravity,
                joint_stiffness,
                joint_damping,
                joint_rest_configuration,
                radius,
            ),
        )

    def update_params(self, **updates: Array) -> "ArticulatedSoftRobot":
        """Return an updated copy with selected typed parameter fields replaced."""
        return self.with_params(self.params.replace(**updates))
