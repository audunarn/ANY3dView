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
python -m twine upload C:\Github\ANY3dView\dist\any3dview-0.4.0*
python -m twine upload C:\Github\ANYtk3D\dist\anytk3d-0.4.0*
```
