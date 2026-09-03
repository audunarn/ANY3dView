"""Fail release validation when licensing or GPU dependency policy drifts."""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    if project["license"] != "MPL-2.0":
        raise SystemExit("project licence must be MPL-2.0")
    if project["version"] != "0.5.5":
        raise SystemExit("licensing gate is scoped to the 0.5.5 release")

    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "Mozilla Public License Version 2.0" not in licence:
        raise SystemExit("LICENSE does not contain the MPL-2.0 text")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "Mozilla Public" not in readme or "License 2.0" not in readme:
        raise SystemExit("README does not declare MPL-2.0")

    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    native_notice = (
        ROOT / "src" / "any3dview" / "gpu" / "tkgl" / "license.terms"
    ).read_text(encoding="utf-8")
    if "TkGL 1.2.1" not in notices or "grant permission to use" not in native_notice:
        raise SystemExit("TkGL provenance or licence notice is incomplete")

    dependency_text = "\n".join(
        str(item)
        for group in project.get("optional-dependencies", {}).values()
        for item in group
    ).casefold()
    if "tkinter-gl" in dependency_text:
        raise SystemExit("GPL tkinter-gl Python dependency must not be restored")

    source = ROOT / "src"
    forbidden_import = "tkinter" + "_gl"
    for path in source.rglob("*.py"):
        if forbidden_import in path.read_text(encoding="utf-8"):
            raise SystemExit(f"forbidden tkinter-gl wrapper reference in {path}")


if __name__ == "__main__":
    main()
