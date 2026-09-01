"""Tests for the reproducible Section Vd multi-start gain initialization."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# The Section Vd scripts all run in double precision, and the recovered legacy
# initialization is only reproducible there: jax.random.uniform draws a different
# number of bits for float32, so the PRNG stream diverges.
jax.config.update("jax_enable_x64", True)

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
CODE_DIR = SECTION_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_init import (  # noqa: E402
    GAIN_ORDER,
    LEGACY_KEY_ORDER,
    LEGACY_MIXED_BOX,
    LEGACY_MIXED_NOMINAL,
    LEGACY_MIXED_SEED,
    SECVD_BATCH_SIZE,
    describe_initial_gains,
    sample_initial_gains,
)

NOMINAL = {"Kp": 5e1 * jnp.ones(3), "Ki": 5e0 * jnp.ones(3), "Kd": 1e0 * jnp.ones(3)}

# The initialization recovered from legacy/soromox_synergistic.py, which is what
# issue #154 asked for. Reproducing these exact numbers is the whole point of the
# legacy_mixed_v1 scheme, so they are pinned here rather than recomputed.
#
# They also pin JAX's threefry PRNG stream. If this test starts failing after a
# JAX upgrade, the recovery is what needs revisiting -- confirm against
# legacy/reproduce_legacy_initialization.py, which checks the resulting losses
# against the archive itself -- rather than simply refreshing the numbers here.
RECOVERED_LEGACY_GAINS = {
    "Kp": [
        [40.000000, 40.000000, 40.000000],
        [31.246882, 54.342562, 47.174962],
        [32.427270, 54.866913, 57.093655],
        [49.208360, 33.935476, 45.062707],
        [36.808595, 40.015407, 52.990788],
        [26.007089, 43.655030, 21.373309],
    ],
    "Ki": [
        [20.000000, 20.000000, 20.000000],
        [25.155630, 18.881253, 21.681282],
        [29.217835, 20.634529, 12.303252],
        [16.497855, 22.253546, 22.854653],
        [12.537302, 23.591912, 15.634763],
        [29.807648, 27.303564, 10.326031],
    ],
    "Kd": [
        [2.000000, 2.000000, 2.000000],
        [2.970108, 2.202423, 2.616360],
        [2.434829, 2.092111, 2.235513],
        [2.399573, 2.007426, 2.409414],
        [2.076649, 2.401900, 2.073604],
        [2.953022, 2.156015, 2.039468],
    ],
}


def _legacy_nominal(actuators: int = 3) -> dict:
    return {
        name: value * jnp.ones(actuators)
        for name, value in LEGACY_MIXED_NOMINAL.items()
    }


def _sample_legacy(**kwargs):
    return sample_initial_gains(
        _legacy_nominal(), scheme="legacy_mixed_v1", seed=LEGACY_MIXED_SEED, **kwargs
    )


@pytest.mark.parametrize("scheme", ["log_uniform_v1", "legacy_mixed_v1"])
@pytest.mark.parametrize("batch_size", [1, SECVD_BATCH_SIZE])
def test_start_zero_is_always_the_untouched_nominal(scheme, batch_size):
    """A single-start run must stay a strict subset of a batched one."""
    gains = sample_initial_gains(NOMINAL, batch_size=batch_size, scheme=scheme)
    for name in GAIN_ORDER:
        assert gains[name].shape == (batch_size, 3)
        assert jnp.array_equal(gains[name][0], NOMINAL[name])


def test_default_batch_matches_the_legacy_archive_width():
    assert SECVD_BATCH_SIZE == 6
    assert sample_initial_gains(NOMINAL)["Kp"].shape == (6, 3)


def test_legacy_mixed_v1_reproduces_the_recovered_initialization():
    """The six gain sets behind the published Section Vd results (issue #154)."""
    gains = _sample_legacy()
    for name in GAIN_ORDER:
        expected = jnp.asarray(RECOVERED_LEGACY_GAINS[name])
        assert gains[name].shape == expected.shape
        assert jnp.allclose(gains[name], expected, atol=1e-6), name


def test_legacy_key_order_is_load_bearing():
    """Subkeys were handed out in sorted key order, not GAIN_ORDER.

    The original flattened a PyTree before splitting, so consuming the subkeys in
    the natural Kp, Ki, Kd order silently resamples every start. Guard the fact
    that the two orders genuinely differ, so the constant cannot be "tidied".
    """
    assert LEGACY_KEY_ORDER == ("Kd", "Ki", "Kp")
    assert LEGACY_KEY_ORDER != GAIN_ORDER

    gains = _sample_legacy()
    # Kp and Kd would swap streams under the natural order; their boxes differ,
    # so a wrong assignment cannot even land in the right range.
    for name in GAIN_ORDER:
        low, high = LEGACY_MIXED_BOX[name]
        assert bool(jnp.all((gains[name][1:] >= low) & (gains[name][1:] <= high)))


def test_sampling_is_reproducible_and_seed_dependent():
    first = sample_initial_gains(NOMINAL, seed=7)
    again = sample_initial_gains(NOMINAL, seed=7)
    other = sample_initial_gains(NOMINAL, seed=8)
    for name in GAIN_ORDER:
        assert jnp.array_equal(first[name], again[name])
        # Start 0 is nominal under every seed; only the sampled ones may differ.
        assert not jnp.array_equal(first[name][1:], other[name][1:])


def test_log_uniform_samples_stay_inside_their_spread():
    spread = 3.0
    gains = sample_initial_gains(NOMINAL, spread=spread)
    for name in GAIN_ORDER:
        assert bool(jnp.all(gains[name] >= NOMINAL[name] / spread - 1e-9))
        assert bool(jnp.all(gains[name] <= NOMINAL[name] * spread + 1e-9))


def test_log_uniform_gives_each_start_one_scalar_factor_per_gain():
    """log_uniform_v1 describes each start with three numbers, not 3 x m.

    Scaling the whole nominal vector also preserves any anisotropy it carries,
    which per-component sampling would destroy.
    """
    isotropic = sample_initial_gains(NOMINAL)
    for name in GAIN_ORDER:
        assert bool(jnp.all(isotropic[name] == isotropic[name][:, :1]))

    anisotropic = dict(NOMINAL, Kp=jnp.array([10.0, 20.0, 40.0]))
    ratios = sample_initial_gains(anisotropic)["Kp"] / anisotropic["Kp"]
    assert bool(jnp.allclose(ratios, ratios[:, :1]))


def test_legacy_mixed_v1_samples_each_component_independently():
    """The converse of the log_uniform_v1 property: the original drew per component."""
    gains = _sample_legacy()
    for name in GAIN_ORDER:
        sampled = gains[name][1:]
        assert not bool(jnp.all(sampled == sampled[:, :1])), name


def test_legacy_mixed_v1_ignores_the_nominal_for_sampled_starts():
    """Its box is absolute, so only start 0 depends on the caller's nominal."""
    other_nominal = {name: 3.0 * value for name, value in _legacy_nominal().items()}
    shifted = sample_initial_gains(
        other_nominal, scheme="legacy_mixed_v1", seed=LEGACY_MIXED_SEED
    )
    baseline = _sample_legacy()
    for name in GAIN_ORDER:
        assert jnp.array_equal(shifted[name][1:], baseline[name][1:])
        assert jnp.array_equal(shifted[name][0], other_nominal[name])


def test_gain_keys_are_consumed_in_a_fixed_order():
    """Reordering the input dict must not change which key each gain draws."""
    reordered = {name: NOMINAL[name] for name in reversed(GAIN_ORDER)}
    for scheme in ("log_uniform_v1", "legacy_mixed_v1"):
        baseline = sample_initial_gains(NOMINAL, scheme=scheme)
        shuffled = sample_initial_gains(reordered, scheme=scheme)
        for name in GAIN_ORDER:
            assert jnp.array_equal(baseline[name], shuffled[name])


@pytest.mark.parametrize(
    ("nominal", "kwargs"),
    [
        (NOMINAL, {"batch_size": 0}),
        (NOMINAL, {"scheme": "not_a_scheme"}),
        (NOMINAL, {"spread": 1.0}),
        (NOMINAL, {"spread": 0.5}),
        ({"Kp": jnp.ones(3), "Ki": jnp.ones(3)}, {}),
        # log_uniform_v1 is multiplicative, so a zero gain has no neighbourhood.
        (dict(NOMINAL, Ki=jnp.zeros(3)), {}),
    ],
)
def test_requests_that_cannot_produce_finite_gains_are_rejected(nominal, kwargs):
    with pytest.raises(ValueError):
        sample_initial_gains(nominal, **kwargs)


def test_absolute_box_scheme_accepts_a_zero_nominal():
    gains = sample_initial_gains(
        dict(NOMINAL, Ki=jnp.zeros(3)), scheme="legacy_mixed_v1"
    )
    assert bool(jnp.all(gains["Ki"][1:] > 0.0))


def test_description_lists_every_start_and_component():
    table = describe_initial_gains(_sample_legacy())
    assert all(name in table for name in GAIN_ORDER)
    assert len(table.splitlines()) == SECVD_BATCH_SIZE + 3
    # Per-component sampling must not be hidden behind a single printed value.
    assert "31.2469, 54.3426, 47.1750" in table
