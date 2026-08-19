import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.systems import (
    GVS,
    PCS,
    CrossSectionGeometry,
    GVSSegment,
    GVSStructure,
    IsotropicMaterialParams,
    JointSpec,
    LinearProfile,
    LinkSpec,
    PlanarPCS,
    StrainBasisSpec,
    shear_modulus_from_poisson_ratio,
)
from soromox.systems.components import section_properties

jax.config.update("jax_enable_x64", True)

REFERENCE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def _material() -> IsotropicMaterialParams:
    return IsotropicMaterialParams(
        young_modulus=jnp.array([1.0e6]),
        shear_modulus=jnp.array([3.4e5]),
        material_damping_coefficient=jnp.array([1.0e4]),
    )


def _poisson_material(*, damping: bool = True) -> IsotropicMaterialParams:
    return IsotropicMaterialParams(
        young_modulus=jnp.array([1.0e6]),
        poisson_ratio=jnp.array([1.0e6 / (2.0 * 3.4e5) - 1.0]),
        material_damping_coefficient=(jnp.array([1.0e4]) if damping else None),
    )


def _pcs() -> PCS:
    return PCS.from_links(
        [
            LinkSpec.circular(
                length=0.2,
                radius=0.012,
                density=1000.0,
                young_modulus=1.0e6,
                shear_modulus=3.4e5,
                material_damping_coefficient=1.0e4,
                reference_strain=REFERENCE,
            )
        ]
    )


def _gvs() -> GVS:
    return GVS.from_segments(
        [
            GVSSegment(
                link=LinkSpec.rectangular(
                    length=0.2,
                    height=LinearProfile(0.03, 0.02),
                    width=0.025,
                    density=1000.0,
                    young_modulus=1.0e6,
                    shear_modulus=3.4e5,
                    material_damping_coefficient=1.0e4,
                    reference_strain=REFERENCE,
                ),
                joint=JointSpec.revolute(
                    "z",
                    stiffness=jnp.array([[0.3]]),
                    damping=jnp.array([[0.02]]),
                ),
                basis=StrainBasisSpec(
                    type="legendre",
                    strain_selector=("kappa_y", "sigma_x"),
                    basis_order=1,
                ),
                num_gauss_points=7,
            )
        ]
    )


def test_shared_params_replace_and_explicit_matrix_bypass() -> None:
    stiffness = jnp.diag(jnp.arange(1.0, 7.0))
    damping = 0.1 * stiffness
    robot = PCS.from_links(
        [
            LinkSpec.circular(
                length=0.2,
                radius=0.01,
                density=1000.0,
                stiffness=stiffness,
                damping=damping,
                reference_strain=REFERENCE,
            )
        ]
    )
    assert_allclose(robot.params.link.stiffness[0], stiffness)
    assert_allclose(robot.params.link.damping[0], damping)
    updated = robot.update_link_params(density=jnp.array([1050.0]))
    assert_allclose(updated.params.link.density, jnp.array([1050.0]))
    assert_allclose(robot.params.link.density, jnp.array([1000.0]))


def test_material_params_normalize_python_sequences() -> None:
    material = IsotropicMaterialParams(
        young_modulus=[1.0e6],
        shear_modulus=[3.4e5],
        material_damping_coefficient=[1.0e4],
    )

    assert isinstance(material.young_modulus, jax.Array)
    updated = material.replace(young_modulus=1.1 * material.young_modulus)
    assert_allclose(updated.young_modulus, jnp.array([1.1e6]))


def test_material_params_preserve_static_parameterization_structure() -> None:
    shear_material = _material()
    poisson_material = _poisson_material()
    undamped_material = _poisson_material(damping=False)

    assert jax.tree.structure(shear_material) != jax.tree.structure(poisson_material)
    assert jax.tree.structure(poisson_material) != jax.tree.structure(undamped_material)
    assert poisson_material.shear_modulus is None
    assert shear_material.poisson_ratio is None
    assert undamped_material.material_damping_coefficient is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("length", 0.0, "length must be strictly positive"),
        ("length", jnp.nan, "length must contain only finite"),
        ("density", -1.0, "density must be strictly positive"),
        ("young_modulus", 0.0, "young_modulus must be strictly positive"),
        ("shear_modulus", jnp.inf, "shear_modulus must contain only finite"),
        (
            "material_damping_coefficient",
            -1.0,
            "material_damping_coefficient must be nonnegative",
        ),
    ],
)
def test_link_spec_rejects_nonphysical_values(field, value, match) -> None:
    kwargs = {
        "length": 0.2,
        "radius": 0.012,
        "density": 1000.0,
        "young_modulus": 1.0e6,
        "shear_modulus": 3.4e5,
        "material_damping_coefficient": 1.0e4,
        "reference_strain": REFERENCE,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        LinkSpec.circular(**kwargs)


@pytest.mark.parametrize("poisson_ratio", [-1.0, 0.5, jnp.nan])
def test_link_spec_rejects_invalid_poisson_ratio(poisson_ratio) -> None:
    with pytest.raises(ValueError, match="poisson_ratio"):
        LinkSpec.circular(
            length=0.2,
            radius=0.012,
            density=1000.0,
            young_modulus=1.0e6,
            poisson_ratio=poisson_ratio,
            reference_strain=REFERENCE,
        )


@pytest.mark.parametrize(
    "stiffness_kwargs",
    [
        {},
        {"young_modulus": 1.0e6},
        {
            "young_modulus": 1.0e6,
            "shear_modulus": 3.4e5,
            "poisson_ratio": 0.45,
        },
    ],
)
def test_link_spec_requires_exactly_one_isotropic_parameterization(
    stiffness_kwargs,
) -> None:
    with pytest.raises(ValueError, match="material|shear_modulus"):
        LinkSpec.circular(
            length=0.2,
            radius=0.012,
            density=1000.0,
            reference_strain=REFERENCE,
            **stiffness_kwargs,
        )


@pytest.mark.parametrize("radius", [0.0, -0.01, jnp.nan])
def test_link_spec_rejects_invalid_cross_section_dimensions(radius) -> None:
    with pytest.raises(ValueError, match="cross_section_coefficients"):
        LinkSpec.circular(
            length=0.2,
            radius=radius,
            density=1000.0,
            young_modulus=1.0e6,
            shear_modulus=3.4e5,
            material_damping_coefficient=1.0e4,
            reference_strain=REFERENCE,
        )


@pytest.mark.parametrize("endpoints", [(0.0, 0.01), (0.01, -0.01), (jnp.nan, 0.01)])
def test_linear_profile_rejects_invalid_endpoints(endpoints) -> None:
    with pytest.raises(ValueError, match="finite and strictly positive"):
        LinearProfile(*endpoints)


def test_runtime_link_params_reject_nonphysical_geometry_updates() -> None:
    link = _pcs().params.link

    with pytest.raises(ValueError, match="length must be strictly positive"):
        link.replace(length=jnp.array([0.0]))
    with pytest.raises(ValueError, match="density must contain only finite"):
        link.replace(density=jnp.array([jnp.nan]))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        link.replace(
            cross_section=link.cross_section.replace(coefficients=jnp.array([[-0.01]]))
        )
    with pytest.raises(ValueError, match="reference_strain.*finite"):
        link.replace(reference_strain=link.reference_strain.at[0, 0].set(jnp.nan))


def test_gvs_rejects_zero_active_cross_section_dimension() -> None:
    robot = _gvs()
    coefficients = robot.params.link.cross_section.coefficients.at[0, 2].set(0.0)
    invalid_params = robot.params.replace(
        link=robot.params.link.replace(
            cross_section=robot.params.link.cross_section.replace(
                coefficients=coefficients
            )
        )
    )

    with pytest.raises(ValueError, match="strictly positive.*segment 0"):
        GVS(invalid_params, robot.structure)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("young_modulus", [0.0], "strictly positive"),
        ("shear_modulus", [jnp.nan], "finite"),
        ("material_damping_coefficient", [-1.0], "nonnegative"),
    ],
)
def test_material_params_reject_nonphysical_values(field, value, match) -> None:
    kwargs = {
        "young_modulus": [1.0e6],
        "shear_modulus": [3.4e5],
        "material_damping_coefficient": [1.0e4],
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        IsotropicMaterialParams(**kwargs)


@pytest.mark.parametrize("poisson_ratio", [[-1.0], [0.5], [jnp.nan]])
def test_material_params_reject_invalid_poisson_ratio(poisson_ratio) -> None:
    with pytest.raises(ValueError, match="poisson_ratio"):
        IsotropicMaterialParams(
            young_modulus=[1.0e6],
            poisson_ratio=poisson_ratio,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"young_modulus": [1.0e6]},
        {
            "young_modulus": [1.0e6],
            "shear_modulus": [3.4e5],
            "poisson_ratio": [0.45],
        },
    ],
)
def test_material_params_require_exactly_one_stiffness_parameterization(
    kwargs,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        IsotropicMaterialParams(**kwargs)


def test_gvs_joint_and_link_blocks_both_contribute() -> None:
    robot = _gvs()
    assert_allclose(
        robot.K_full[: robot.max_dof, : robot.max_dof], robot.params.joint.stiffness[0]
    )
    assert_allclose(
        robot.D_full[: robot.max_dof, : robot.max_dof], robot.params.joint.damping[0]
    )
    assert_allclose(
        robot.K_full[robot.max_dof :, robot.max_dof :], robot.params.link.stiffness[0]
    )
    assert_allclose(
        robot.D_full[robot.max_dof :, robot.max_dof :], robot.params.link.damping[0]
    )


def test_gvs_constructor_resolves_optional_structure_padding() -> None:
    source = _gvs()
    unresolved = GVSStructure(
        segments=source.structure.segments,
        max_dof=None,
        max_num_gauss_points=None,
        scale_rotational_basis_by_length=(
            source.structure.scale_rotational_basis_by_length
        ),
    )

    rebuilt = GVS(source.params, unresolved)

    assert rebuilt.max_dof == source.max_dof
    assert rebuilt.max_num_gauss_points == source.max_num_gauss_points
    assert rebuilt.structure.max_dof == source.max_dof
    assert rebuilt.structure.max_num_gauss_points == source.max_num_gauss_points


def test_gvs_rectangular_geometry_uses_shared_dimension_order() -> None:
    robot = _gvs()
    q = jnp.zeros(robot.num_dofs)
    tag, dimensions = robot.cross_section_geometry(q, 0.25 * robot.segment_length[0])
    expected_height = 0.03 + 0.25 * (0.02 - 0.03)

    assert int(tag) == CrossSectionGeometry.RECTANGULAR
    assert_allclose(dimensions, jnp.array([expected_height, 0.025]))
    ix, iy, iz, area = section_properties(tag, dimensions)
    expected_iy = expected_height * 0.025**3 / 12.0
    expected_iz = 0.025 * expected_height**3 / 12.0
    assert_allclose(area, expected_height * 0.025)
    assert_allclose(iy, expected_iy)
    assert_allclose(iz, expected_iz)
    assert_allclose(ix, expected_iy + expected_iz)


def test_geometry_refreshes_operators_without_overwriting_matrices() -> None:
    robot = _pcs()
    original_stiffness = robot.params.link.stiffness
    original_operator = robot.young_stiffness_operator
    section = robot.params.link.cross_section.replace(
        coefficients=1.1 * robot.params.link.cross_section.coefficients
    )
    geometry_robot = robot.update_link_params(cross_section=section)
    assert_allclose(geometry_robot.params.link.stiffness, original_stiffness)
    assert not jnp.allclose(geometry_robot.young_stiffness_operator, original_operator)
    rebuilt = geometry_robot.with_isotropic_material(_material())
    assert not jnp.allclose(rebuilt.params.link.stiffness, original_stiffness)


def test_poisson_ratio_construction_and_omitted_damping_for_all_systems() -> None:
    young_modulus = 1.0e6
    poisson_ratio = 0.45
    shear_modulus = shear_modulus_from_poisson_ratio(young_modulus, poisson_ratio)

    pcs_from_shear = PCS.from_links(
        [
            LinkSpec.circular(
                length=0.2,
                radius=0.012,
                density=1000.0,
                young_modulus=young_modulus,
                shear_modulus=shear_modulus,
                reference_strain=REFERENCE,
            )
        ]
    )
    pcs_from_poisson = PCS.from_links(
        [
            LinkSpec.circular(
                length=0.2,
                radius=0.012,
                density=1000.0,
                young_modulus=young_modulus,
                poisson_ratio=poisson_ratio,
                reference_strain=REFERENCE,
            )
        ]
    )

    planar_from_poisson = PlanarPCS.from_links(
        [
            LinkSpec.circular(
                length=0.2,
                radius=0.012,
                density=1000.0,
                young_modulus=young_modulus,
                poisson_ratio=poisson_ratio,
                reference_strain=[0.0, 0.0, 1.0],
            )
        ]
    )

    gvs_from_poisson = GVS.from_segments(
        [
            GVSSegment(
                link=LinkSpec.rectangular(
                    length=0.2,
                    height=0.03,
                    width=0.025,
                    density=1000.0,
                    young_modulus=young_modulus,
                    poisson_ratio=poisson_ratio,
                    reference_strain=REFERENCE,
                ),
                joint=JointSpec.fixed(),
                basis=StrainBasisSpec(
                    type="legendre",
                    strain_selector=("kappa_y", "sigma_x"),
                    basis_order=1,
                ),
                num_gauss_points=5,
            )
        ]
    )

    assert_allclose(
        pcs_from_poisson.params.link.stiffness,
        pcs_from_shear.params.link.stiffness,
    )
    for robot in (pcs_from_poisson, planar_from_poisson, gvs_from_poisson):
        assert_allclose(robot.params.link.damping, 0.0)


def test_poisson_material_gradients_and_disabled_damping_are_jittable() -> None:
    for robot in (_pcs(), _gvs()):
        material = _poisson_material(damping=False)

        def objective(candidate, robot=robot):
            updated = robot.with_isotropic_material(candidate)
            return jnp.sum(updated.params.link.stiffness) + jnp.sum(
                updated.params.link.damping
            )

        stiffness, damping = jax.jit(
            lambda candidate, robot=robot: robot.link_matrices_from_material(candidate)
        )(material)
        shear_stiffness_sum = jnp.sum(robot.shear_stiffness_operator, axis=(1, 2))
        expected_young_gradient = jnp.sum(
            robot.young_stiffness_operator, axis=(1, 2)
        ) + shear_stiffness_sum / (2.0 * (1.0 + material.poisson_ratio))
        expected_poisson_gradient = (
            -material.young_modulus
            * shear_stiffness_sum
            / (2.0 * (1.0 + material.poisson_ratio) ** 2)
        )
        gradient = jax.jit(jax.grad(objective))(material)

        assert_allclose(stiffness, robot.params.link.stiffness)
        assert_allclose(damping, 0.0)
        assert_allclose(gradient.young_modulus, expected_young_gradient)
        assert gradient.shear_modulus is None
        assert_allclose(gradient.poisson_ratio, expected_poisson_gradient)
        assert gradient.material_damping_coefficient is None


def test_material_gradients_and_jit_for_pcs_and_gvs() -> None:
    for robot in (_pcs(), _gvs()):
        material = _material()

        def objective(candidate, robot=robot):
            updated = robot.with_isotropic_material(candidate)
            return jnp.sum(updated.params.link.stiffness) + jnp.sum(
                updated.params.link.damping
            )

        gradient = jax.jit(jax.grad(objective))(material)
        assert_allclose(
            gradient.young_modulus,
            jnp.sum(robot.young_stiffness_operator, axis=(1, 2)),
        )
        assert_allclose(
            gradient.shear_modulus,
            jnp.sum(robot.shear_stiffness_operator, axis=(1, 2)),
        )
        assert_allclose(
            gradient.material_damping_coefficient,
            jnp.sum(robot.material_damping_operator, axis=(1, 2)),
        )


def test_poisson_material_gradient_matches_finite_difference() -> None:
    robot = _pcs()
    material = _poisson_material(damping=False)

    def objective(poisson_ratio):
        candidate = material.replace(poisson_ratio=poisson_ratio)
        return jnp.sum(robot.link_matrices_from_material(candidate)[0])

    automatic = jax.grad(objective)(material.poisson_ratio)
    step = 1e-5
    finite_difference = (
        objective(material.poisson_ratio + step)
        - objective(material.poisson_ratio - step)
    ) / (2.0 * step)
    assert_allclose(automatic[0], finite_difference, rtol=1e-7, atol=1e-7)


def test_scalar_and_per_link_material_inputs() -> None:
    robot = PCS.from_links(
        [
            LinkSpec.circular(
                length=0.2,
                radius=0.012,
                density=1000.0,
                young_modulus=1.0e6,
                shear_modulus=3.4e5,
                material_damping_coefficient=1.0e4,
                reference_strain=REFERENCE,
            ),
            LinkSpec.circular(
                length=0.15,
                radius=0.01,
                density=1050.0,
                young_modulus=1.0e6,
                shear_modulus=3.4e5,
                material_damping_coefficient=1.0e4,
                reference_strain=REFERENCE,
            ),
        ]
    )
    scalar = IsotropicMaterialParams(
        young_modulus=jnp.array(1.0e6),
        shear_modulus=jnp.array(3.4e5),
        material_damping_coefficient=jnp.array(1.0e4),
    )
    per_link = IsotropicMaterialParams(
        young_modulus=jnp.full((2,), 1.0e6),
        shear_modulus=jnp.full((2,), 3.4e5),
        material_damping_coefficient=jnp.full((2,), 1.0e4),
    )
    scalar_matrices = robot.link_matrices_from_material(scalar)
    per_link_matrices = robot.link_matrices_from_material(per_link)
    assert_allclose(scalar_matrices[0], per_link_matrices[0])
    assert_allclose(scalar_matrices[1], per_link_matrices[1])

    scalar_poisson = IsotropicMaterialParams(
        young_modulus=jnp.array(1.0e6),
        poisson_ratio=jnp.array(1.0e6 / (2.0 * 3.4e5) - 1.0),
    )
    per_link_poisson = IsotropicMaterialParams(
        young_modulus=jnp.full((2,), 1.0e6),
        poisson_ratio=jnp.full((2,), 1.0e6 / (2.0 * 3.4e5) - 1.0),
    )
    scalar_poisson_matrices = robot.link_matrices_from_material(scalar_poisson)
    per_link_poisson_matrices = robot.link_matrices_from_material(per_link_poisson)
    assert_allclose(scalar_poisson_matrices[0], per_link_poisson_matrices[0])
    assert_allclose(scalar_poisson_matrices[1], 0.0)
    assert_allclose(per_link_poisson_matrices[1], 0.0)

    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        robot.link_matrices_from_material(
            per_link.replace(young_modulus=jnp.ones((3,)))
        )
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        robot.link_matrices_from_material(
            per_link_poisson.replace(poisson_ratio=jnp.zeros((3,)))
        )


def test_material_gradient_matches_finite_difference() -> None:
    robot = _pcs()
    material = _material()

    def objective(young):
        candidate = material.replace(young_modulus=young)
        return jnp.sum(robot.link_matrices_from_material(candidate)[0])

    automatic = jax.grad(objective)(material.young_modulus)
    step = 1.0
    finite_difference = (
        objective(material.young_modulus + step)
        - objective(material.young_modulus - step)
    ) / (2.0 * step)
    assert_allclose(automatic[0], finite_difference, rtol=1e-7, atol=1e-10)


def test_log_material_optax_loop_reduces_loss() -> None:
    import optax

    robot = _pcs()
    material = _material()
    target = robot.link_matrices_from_material(
        material.replace(
            young_modulus=1.1 * material.young_modulus,
            shear_modulus=0.9 * material.shear_modulus,
            material_damping_coefficient=(1.2 * material.material_damping_coefficient),
        )
    )
    log_material = jax.tree.map(jnp.log, material)

    def decode(values):
        return jax.tree.map(jnp.exp, values)

    def loss(values):
        stiffness, damping = robot.link_matrices_from_material(decode(values))
        return jnp.mean(((stiffness - target[0]) / 100.0) ** 2) + jnp.mean(
            ((damping - target[1]) / 10.0) ** 2
        )

    optimizer = optax.adam(1e-2)
    state = optimizer.init(log_material)
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    initial = loss(log_material)
    for _ in range(25):
        _, gradient = value_and_grad(log_material)
        updates, state = optimizer.update(gradient, state, log_material)
        log_material = optax.apply_updates(log_material, updates)
    assert jnp.isfinite(loss(log_material))
    assert loss(log_material) < initial
