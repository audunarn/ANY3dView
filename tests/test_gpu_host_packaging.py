from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_extra_does_not_install_the_gpl_python_wrapper():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    gpu_requirements = "\n".join(project["optional-dependencies"]["gpu"])
    assert "tkinter-gl" not in gpu_requirements.casefold()


def test_native_tkgl_notice_and_supported_platform_assets_are_present():
    root = ROOT / "src" / "any3dview" / "gpu" / "tkgl"
    assert "grant permission to use" in (root / "license.terms").read_text(
        encoding="utf-8"
    )
    expected = (
        root / "win32" / "TkGL1.2.1" / "pkgIndex.tcl",
        root / "darwin" / "Tkgl1.2.1" / "pkgIndex.tcl",
        root / "linux-x86_64" / "Tkgl1.2.1" / "pkgIndex.tcl",
        root / "linux-aarch64" / "Tkgl1.2.1" / "pkgIndex.tcl",
    )
    assert all(path.is_file() for path in expected)
