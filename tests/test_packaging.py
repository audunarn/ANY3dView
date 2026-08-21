import pathlib
import tomllib

import any3dview


ROOT = pathlib.Path(any3dview.__file__).resolve().parents[2]


def test_version_and_dependencies_match_metadata():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == any3dview.__version__
    assert project["dependencies"] == ["numpy"]


def test_license_and_readme_exist():
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "README.md").is_file()
