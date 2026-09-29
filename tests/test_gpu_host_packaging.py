from __future__ import annotations

from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


class _ParentProbe:
    def __init__(self, *, mapped: bool, maps_during_idle: bool = False) -> None:
        self.mapped = mapped
        self.maps_during_idle = maps_during_idle
        self.idle_calls = 0
        self.update_calls = 0

    def winfo_ismapped(self) -> bool:
        return self.mapped

    def update_idletasks(self) -> None:
        self.idle_calls += 1
        self.mapped = self.maps_during_idle

    def update(self) -> None:
        self.update_calls += 1
        self.mapped = True


def test_windows_tkgl_host_does_not_reenter_a_mapped_parent_event_loop(monkeypatch):
    from any3dview.gpu import _tkgl

    monkeypatch.setattr(_tkgl.sys, "platform", "win32")
    parent = _ParentProbe(mapped=True)
    _tkgl._ensure_native_parent(parent)

    assert parent.idle_calls == 0
    assert parent.update_calls == 0


def test_windows_tkgl_host_maps_only_an_unmapped_parent(monkeypatch):
    from any3dview.gpu import _tkgl

    monkeypatch.setattr(_tkgl.sys, "platform", "win32")
    parent = _ParentProbe(mapped=False)
    _tkgl._ensure_native_parent(parent)

    assert parent.idle_calls == 1
    assert parent.update_calls == 1


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
