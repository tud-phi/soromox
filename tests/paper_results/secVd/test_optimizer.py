"""Tests for the shared Section Vd optimizer builder.

``secvd_case.build_optimizer`` replaced three ``optax.yogi`` instances under a
``multi_transform`` with one ``scale_by_yogi`` and a per-family rescaling. The
claim that makes that a simplification rather than a change is that the two are
the same optimizer, so it is pinned here rather than left in a docstring.

The generators and every stage of the tuning study construct their optimizer
through this builder, for the same reason they build their problem through
``build_evaluator``: a rate measured by the study has to be the rate a generator
runs.
"""

import ast
import sys
from pathlib import Path

import jax
import optax
import pytest

jax.config.update("jax_enable_x64", True)

CODE_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
    / "code"
)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_case import (  # noqa: E402
    COMMITTED_ALPHA,
    COMMITTED_LR_RATIO,
    FAMILY_LABEL,
    METHODS,
    TUNED_ALPHA,
    YOGI_EPS,
    build_optimizer,
    describe_optimizer,
)
from secvd_init import GAIN_ORDER  # noqa: E402

LABELS = {
    "opt_ctr_params": {name: FAMILY_LABEL[name] for name in GAIN_ORDER},
    "opt_atr_params": {},
}


def committed_optimizer(alpha, ratio):
    """The three-optimizer arrangement both generators used before the builder."""
    return optax.multi_transform(
        {
            FAMILY_LABEL[name]: optax.yogi(alpha * ratio[name], eps=YOGI_EPS)
            for name in GAIN_ORDER
        },
        LABELS,
    )


def gain_tree(values):
    return {
        "opt_ctr_params": {name: values[name] for name in GAIN_ORDER},
        "opt_atr_params": {},
    }


def run(optimizer, params, gradients):
    state = optimizer.init(params)
    for grad in gradients:
        updates, state = optimizer.update(grad, state, params)
        params = optax.apply_updates(params, updates)
    return params


@pytest.fixture(scope="module")
def gradients():
    """Gradients spanning the magnitudes seen across the whole tuning study.

    The measured range runs from ``1e-4`` at the committed rate to ``0.84`` at
    the rate that diverges, so the sweep covers both ends and then some: Yogi's
    behaviour depends on where the gradient sits relative to ``eps``.
    """
    key = jax.random.PRNGKey(0)
    out = []
    for step in range(30):
        scale = 10.0 ** (-6 + step % 7)
        out.append(
            gain_tree(
                {
                    name: scale
                    * jax.random.normal(jax.random.fold_in(key, 31 * step + i), (3,))
                    for i, name in enumerate(GAIN_ORDER)
                }
            )
        )
    return out


@pytest.fixture(scope="module")
def initial_gains():
    return gain_tree(
        {
            "Kp": jax.numpy.full((3,), 50.0),
            "Ki": jax.numpy.full((3,), 5.0),
            "Kd": jax.numpy.ones(3),
        }
    )


@pytest.mark.parametrize("alpha", [COMMITTED_ALPHA, 1000.0])
def test_one_yogi_with_scaled_rates_equals_three_yogis(initial_gains, gradients, alpha):
    """The claim that makes the single-state form a simplification, not a change.

    ``scale_by_yogi``'s moment estimates never see the learning rate, so a
    per-family rate factors out of the state and can be applied afterwards. Only
    floating-point reassociation should separate the two arrangements.
    """
    harmonized = run(
        build_optimizer(alpha, ratio=COMMITTED_LR_RATIO), initial_gains, gradients
    )
    committed = run(
        committed_optimizer(alpha, COMMITTED_LR_RATIO), initial_gains, gradients
    )
    for name in GAIN_ORDER:
        assert harmonized["opt_ctr_params"][name] == pytest.approx(
            committed["opt_ctr_params"][name], rel=1e-9
        )


def test_the_ratio_actually_reaches_the_families(initial_gains, gradients):
    """A ratio applied to the wrong family would still pass an equality check."""
    moved = run(
        build_optimizer(1.0, ratio={"Kp": 1.0, "Ki": 0.0, "Kd": 0.0}),
        initial_gains,
        gradients,
    )
    assert moved["opt_ctr_params"]["Kp"] != pytest.approx(
        initial_gains["opt_ctr_params"]["Kp"]
    )
    for frozen in ("Ki", "Kd"):
        assert moved["opt_ctr_params"][frozen] == pytest.approx(
            initial_gains["opt_ctr_params"][frozen]
        )


def test_a_ratio_missing_a_family_is_rejected():
    with pytest.raises(ValueError, match="ratio must cover exactly"):
        build_optimizer(1.0, ratio={"Kp": 1.0, "Ki": 1.0})


def test_the_archived_description_carries_the_effective_rates():
    """The archive has to say what was run, not which knobs were turned."""
    described = describe_optimizer(1000.0, ratio=COMMITTED_LR_RATIO)
    for name in GAIN_ORDER:
        assert (
            f"{FAMILY_LABEL[name]}: {1000.0 * COMMITTED_LR_RATIO[name]:g}" in described
        )


def test_the_description_renders_the_effective_per_family_rates():
    """Pin the rendering itself, independently of any archive.

    Had dropping the clip and eps parameters changed this string, the shipped
    results would silently start disagreeing with the code that claims to
    reproduce them.
    """
    assert (
        describe_optimizer(1500.0, ratio=COMMITTED_LR_RATIO)
        == "scale_by_yogi(eps=1e-08) + per-family rate{P: 1500, I: 750, D: 150}"
    )


@pytest.mark.parametrize("method", METHODS)
def test_the_committed_archive_describes_the_rate_the_code_would_run(method):
    """A committed archive must not outlive the rate that produced it.

    Lowering ``TUNED_ALPHA`` without regenerating leaves an archive whose
    ``optimizer_metadata`` claims a rate the code no longer uses, which is
    exactly the drift the metadata exists to make visible. Asserting the
    agreement rather than a hand-copied literal means the two cannot separate.
    """
    import numpy as np

    archive = CODE_DIR.parent / "data" / method / "optimization_results.npz"
    if not archive.exists():
        pytest.skip(f"no committed {method} archive to compare against")
    with np.load(archive) as data:
        stored = str(np.asarray(data["optimizer_metadata"]).item())
    assert stored == describe_optimizer(TUNED_ALPHA[method], ratio=COMMITTED_LR_RATIO)


def test_the_entrypoint_does_not_hardcode_its_optimizer():
    """The generator must take the tuned rates from the shared parser.

    It previously held its own copy of the optimizer, and one such copy is how
    the collocated run silently went to a single start while the synergistic one
    ran six. A static check costs nothing and catches the next one before it
    burns a multi-hour run.
    """
    source = (CODE_DIR / "optimize_control_gains.py").read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("build_optimizer", "describe_optimizer")
    ]
    assert len(calls) == 2, "should build and describe exactly one optimizer"
    for call in calls:
        magnitude = call.args[0]
        assert isinstance(magnitude, ast.Attribute), (
            "the entrypoint passes a literal learning rate; it must come from args"
        )
        assert magnitude.attr == "alpha"
    assert "optax." not in source, "the entrypoint still constructs optax transforms"
