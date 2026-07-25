from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_editable_install_and_console_scripts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python -m pip install -e ." in readme
    assert 'python -m pip install -e ".[dev]"' in readme
    assert "huav-inspect --help" in readme
    assert "python -m unittest discover -s tests -v" in readme
    assert "python -m pytest" in readme


def test_root_dataset_paths_are_ignored_but_source_data_package_is_addable() -> None:
    ignored = subprocess.run(["git", "check-ignore", "-q", "data/example.npz"], cwd=ROOT, check=False)
    assert ignored.returncode == 0
    source = subprocess.run(
        ["git", "check-ignore", "-q", "src/huav/data/__init__.py"], cwd=ROOT, check=False,
    )
    assert source.returncode == 1
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "src/huav/data/__init__.py"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert tracked.returncode == 0
    assert tracked.stdout.strip().replace("\\", "/") == "src/huav/data/__init__.py"
