from __future__ import annotations

from pathlib import Path


def test_full_repo_requirements_include_pyside6():
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert any(
        line.strip().lower().startswith("pyside6")
        for line in requirements.splitlines()
    )


def test_generated_live_runtime_directory_is_gitignored():
    root = Path(__file__).resolve().parents[1]
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".linecaller_runtime/" in ignore.splitlines()


def test_strict_powershell_quality_gate_exists():
    root = Path(__file__).resolve().parents[1]
    gate = root / "tools" / "run_quality_gate.ps1"
    assert gate.exists()
    text = gate.read_text(encoding="utf-8")
    assert "Assert-NativeSuccess" in text
    assert "python -m pytest -q" in text