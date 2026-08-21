# Release qualification

Run from any folder on Windows. These commands build frozen smoke programs but
do not publish packages.

```powershell
python -m PyInstaller --noconfirm --clean --onedir `
  --name any3dview-gpu-smoke `
  --paths C:\Github\ANY3dView\src --paths C:\Github\ANYtk3D\src `
  --collect-all tkinter_gl `
  --distpath C:\Github\ANY3dView\.pyinstaller_tmp\onedir `
  --workpath C:\Github\ANY3dView\.pyinstaller_tmp\work-onedir `
  --specpath C:\Github\ANY3dView\.pyinstaller_tmp `
  C:\Github\ANY3dView\packaging\pyinstaller_smoke.py

$env:ANY3DVIEW_BACKEND = "gpu"
& C:\Github\ANY3dView\.pyinstaller_tmp\onedir\any3dview-gpu-smoke\any3dview-gpu-smoke.exe

python -m PyInstaller --noconfirm --clean --onefile `
  --runtime-tmpdir C:\Github\ANY3dView\.pyinstaller_runtime `
  --name any3dview-auto-smoke `
  --paths C:\Github\ANY3dView\src --paths C:\Github\ANYtk3D\src `
  --collect-all tkinter_gl `
  --distpath C:\Github\ANY3dView\.pyinstaller_tmp\onefile `
  --workpath C:\Github\ANY3dView\.pyinstaller_tmp\work-onefile `
  --specpath C:\Github\ANY3dView\.pyinstaller_tmp `
  C:\Github\ANY3dView\packaging\pyinstaller_smoke.py

$env:ANY3DVIEW_BACKEND = "auto"
$env:ANY3DVIEW_DISABLE_GPU = "1"
& C:\Github\ANY3dView\.pyinstaller_tmp\onefile\any3dview-auto-smoke.exe
```

The explicit runtime extraction directory avoids a Python 3.13/PyInstaller
one-file Tcl lookup failure observed with the default short-name temporary
path.

After the repository `dist` artifacts pass all gates, these Twine commands can
also run from any folder. They are intentionally not part of automated release
qualification:

```powershell
python -m twine upload `
  C:\Github\ANY3dView\dist\any3dview-0.5.0-py3-none-any.whl `
  C:\Github\ANY3dView\dist\any3dview-0.5.0.tar.gz

python -m twine upload `
  C:\Github\ANYtk3D\dist\anytk3d-0.5.0-py3-none-any.whl `
  C:\Github\ANYtk3D\dist\anytk3d-0.5.0.tar.gz

python -m twine upload `
  C:\Github\ANYfem\dist\anyfem-0.3.0-py3-none-any.whl `
  C:\Github\ANYfem\dist\anyfem-0.3.0.tar.gz

python -m twine upload `
  C:\Github\ANYstructure\dist\anystructure-6.3.0-py3-none-any.whl `
  C:\Github\ANYstructure\dist\anystructure-6.3.0.tar.gz
```

## PyPI Trusted Publishing

All four updated repositories contain a Trusted Publishing workflow. A manual
workflow run from a branch builds and validates the distributions but does not
publish. Pushing a `v*` tag publishes only after the tag has been checked
against the package version and the `pypi` GitHub environment has approved the
publishing job.

Configure a PyPI GitHub Trusted Publisher for each project with these values:

| Project | Owner | Repository | Workflow | Environment |
| --- | --- | --- | --- | --- |
| ANY3dView | `audunarn` | `ANY3dView` | `release.yml` | `pypi` |
| ANYtk3D | `audunarn` | `ANYtk3D` | `release.yml` | `pypi` |
| ANYfem | `audunarn` | `ANYfem` | `publish.yml` | `pypi` |
| ANYstructure | `audunarn` | `ANYstructure` | `publish.yml` | `pypi` |

The workflows use short-lived OIDC credentials and do not require a
`PYPI_API_TOKEN` repository secret. Protect the `pypi` GitHub environment with
a required reviewer and protect the `v*` tag pattern before publishing.
