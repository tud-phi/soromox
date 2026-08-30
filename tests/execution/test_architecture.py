"""Dependency and public-surface tests for system execution backends."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "soromox"
EXECUTION_ROOT = PACKAGE_ROOT / "execution"
SYSTEMS_ROOT = PACKAGE_ROOT / "systems"


def test_execution_backend_has_one_public_name() -> None:
    """Expose one canonical execution type plus the package-root convenience."""

    from soromox import ExecutionBackend as RootExecutionBackend
    from soromox.execution import ExecutionBackend
    from soromox.systems import gvs

    assert RootExecutionBackend is ExecutionBackend
    assert not hasattr(sys.modules["soromox.systems"], "ExecutionBackend")
    assert not hasattr(gvs, "ExecutionBackend")


def test_execution_package_has_a_public_import_path() -> None:
    """Expose execution policy without a private compatibility package."""

    import soromox.execution as execution

    assert execution.ExecutionBackend is not None
    assert EXECUTION_ROOT.is_dir()
    legacy_execution_root = PACKAGE_ROOT / "systems" / "execution"
    assert not any(legacy_execution_root.rglob("*.py"))
    assert not (PACKAGE_ROOT / "_execution").exists()


def test_importing_soromox_does_not_load_warp() -> None:
    """The optional Warp dependency must remain behind the lazy loader."""

    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = str(PACKAGE_ROOT.parent)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import soromox; "
                "assert 'warp' not in sys.modules; "
                "assert not any(name.endswith('.executor') and "
                "'.execution.warp.' in name for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_warp_execution_does_not_import_system_implementations() -> None:
    """Executors depend on operand contracts, never concrete model classes."""

    warp_root = EXECUTION_ROOT / "warp"
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in warp_root.rglob("*.py")
    )
    forbidden = (
        "soromox.systems.gvs.core",
        "soromox.systems.pcs.pcs",
        "soromox.systems.pcs.planar_pcs",
        "soromox.systems.gvs._warp",
        "soromox.systems.pcs._warp",
    )
    for module_name in forbidden:
        assert module_name not in sources


def test_warp_array_annotations_use_bracket_syntax() -> None:
    """Keep Warp array annotations aligned with the Newton code style."""

    warp_root = EXECUTION_ROOT / "warp"
    legacy_annotation = re.compile(r"(?::|->)\s*wp\.array(?:[1-4]d)?\s*\(\s*dtype\s*=")
    for path in warp_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert legacy_annotation.search(source) is None, path


def test_system_models_only_import_the_neutral_execution_api() -> None:
    """Model modules must not reach into the Warp implementation namespace."""

    model_sources = "\n".join(
        (SYSTEMS_ROOT / relative).read_text(encoding="utf-8")
        for relative in ("gvs/core.py", "pcs/pcs.py", "pcs/planar_pcs.py")
    )
    assert "soromox.execution.warp" not in model_sources
    assert "soromox.systems.gvs._warp" not in model_sources
    assert "soromox.systems.pcs._warp" not in model_sources


def test_warp_families_only_share_through_common_modules() -> None:
    """Family executors must not depend on each other's private kernels."""

    warp_root = EXECUTION_ROOT / "warp"
    gvs_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (warp_root / "gvs").rglob("*.py")
    )
    pcs_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (warp_root / "pcs").rglob("*.py")
    )
    common_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (warp_root / "common").rglob("*.py")
    )
    assert "soromox.execution.warp.pcs" not in gvs_sources
    assert "soromox.execution.warp.gvs" not in pcs_sources
    assert "soromox.execution.warp.gvs" not in common_sources
    assert "soromox.execution.warp.pcs" not in common_sources


def test_forward_dynamics_uses_one_shared_transform_boundary() -> None:
    """System classes must not duplicate execution-routing JVP definitions."""

    model_sources = tuple(
        (SYSTEMS_ROOT / relative).read_text(encoding="utf-8")
        for relative in ("gvs/core.py", "pcs/pcs.py", "pcs/planar_pcs.py")
    )
    for source in model_sources:
        assert "_forward_dynamics_custom_jvp" not in source
        assert "evaluate_forward_dynamics(" in source
        assert "def _evaluate_forward_dynamics" not in source
