# ANY3dView

Backend-neutral geometry, camera, shading, clipping and selection primitives
for scientific 3D viewers.

ANY3dView contains the toolkit-independent core shared by rendering backends.
It does not create windows, process native input or depend on a GUI toolkit.
[ANYtk3D](https://github.com/audunarn/ANYtk3D) provides the compatible
Tkinter Canvas backend.

## Installation

```bash
pip install ANY3dView
```

## Core API

```python
from any3dview import Camera3D, Mesh, PickBinding, Point3D, SectionPlane

camera = Camera3D()
camera.set_target(Point3D(0, 0, 0))

plane = SectionPlane(normal=(1, 0, 0), offset=2.0)
assert plane.contains((3, 0, 0))
assert not plane.contains((1, 0, 0))

binding = PickBinding.one("element:42", "mesh.element")
```

Shape tessellation is available through `any3dview.shapes`; every builder
returns a `Mesh` without importing a renderer. Selection queries are provided
by `ProjectedSelectionIndex`, allowing a backend to expose point, directional
box and lasso selection with visible/through depth policy.

The section-plane convention retains the half-space where
`normal · point >= offset`. The normal is normalized and the offset remains a
world-space distance.

## Development

```bash
pip install -e .[dev]
pytest
python -m build
twine check dist/*
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
