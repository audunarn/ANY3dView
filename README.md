# ANY3dView

Backend-neutral geometry, retained mesh arrays, camera, shading, clipping and
selection contracts for scientific 3D viewers. The base package depends only
on NumPy and imports without Tk, OpenGL, ANYtk3D or ANYgeometry.

ANY3dView contains the toolkit-independent core shared by rendering backends.
It does not create windows or process native input during normal core imports.
[ANYtk3D](https://github.com/audunarn/ANYtk3D) provides the compatible Tk
Canvas backend. An optional ModernGL backend embeds in the same Tk application
without adding a second event loop.

## Installation

```bash
pip install ANY3dView
pip install "ANY3dView[gpu]"       # ModernGL + tkinter-gl
pip install "ANY3dView[geometry]"  # ANYgeometry adapter (Python 3.11+)
```

## Interactive demo

Run the retained-array showcase from an installed package:

```bash
pip install "ANY3dView[gpu]"  # or install ANYtk3D for the software backend
any3dview-demo --backend auto
```

Or run it directly from a source checkout; the launcher also discovers a
sibling `ANYtk3D` checkout for software fallback:

```powershell
python C:\Github\ANY3dView\run_gui.py --backend auto
```

Use `--backend gpu` to require ModernGL or `--backend software` to require
ANYtk3D. The demo includes scalar colouring, deformation animation, section
clipping and backend/fallback diagnostics. Its Renderer selector can replace
the live viewport with the GPU or Tk implementation without restarting the
application.

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

## Retained arrays

`MeshArrays` validates indexed NumPy data once. Compatible C-contiguous arrays
are retained zero-copy and must stay immutable while registered; use
`owned_copy()` when the producer cannot guarantee that lifetime.

```python
import numpy as np
from any3dview import MeshArrays

mesh = MeshArrays(
    positions=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
    triangles=np.asarray([[0, 1, 2]], np.uint32),
    element_ids=np.asarray([42], np.uint64),
    element_scalars=np.asarray([180.0], np.float32),
)

handle = viewer.add_mesh_arrays(mesh)
handle.update_element_scalars(np.asarray([205.0], np.float32))
handle.set_selected_elements([0])
```

`MeshHandle` also supports positions, displacements, deformation scale,
active masks, transforms, visibility, local chunk replacement and idempotent
removal. Independent generation counters let backends update only changed
buffers or display batches. Cross-thread producers can call
`viewer.submit_update(handle.update_displacements, immutable_array)`; the
callback runs on the viewer's owning Tk thread.

Packed CSR owner tables avoid allocating owner objects per primitive.
`EntityHandle` or `PickOwner` values are materialized only for selection hits.

## Backends

```python
from any3dview import create_viewer

viewer = create_viewer(parent, backend="auto")
```

`backend="gpu"` requires OpenGL 3.3 and raises `GPUUnavailableError` with
diagnostics on failure. `backend="software"` lazily imports ANYtk3D. `auto`
tries GPU first and falls back to software while retaining diagnostics.

The GPU path provides persistent indexed buffers, frustum culling,
camera-relative float32 positions, derivative flat normals, instanced
screen-space lines, point markers, node and element result fields,
deformation, active and selection masks, sorted alpha, integer point picking,
and visible/through rectangle and lasso queries. Rendering is demand-driven.

## ANYgeometry adapter

```python
from any3dview.adapters.anygeometry import DisplayPolicy, GeometryLayer

layer = GeometryLayer(model, DisplayPolicy(mode="combined"))
viewer.add_layer(layer)
```

The optional adapter consumes ANYgeometry 0.2.2/schema 4 public records and
change sets. Stable chunks, entity-generation tessellation caches, bounded
cross-thread polling, revision-gap resynchronization and replacement-lineage
selection keep geometry ownership separate from display data. Geometry,
structural, topology-debug, relationships and combined policies are available.

Shape tessellation is available through `any3dview.shapes`; every builder
returns a `Mesh` without importing a renderer. Selection queries are provided
by `ProjectedSelectionIndex`, allowing a backend to expose point, directional
box and lasso selection with visible/through depth policy.

The section-plane convention retains the half-space where
`normal · point >= offset`. The normal is normalized and the offset remains a
world-space distance.

## Performance qualification

The standalone benchmark records platform, driver, commit, scene, median/p95
CPU and GPU frame times, upload and cached-pick timings, draw calls, upload
counts and array memory as JSON:

```powershell
$env:PYTHONPATH = "C:\Github\ANY3dView\src"
python C:\Github\ANY3dView\benchmarks\run_gpu.py `
  --output C:\Github\ANY3dView\benchmark-results\reference-current.json
```

It exercises approximately one million opaque triangles, the same scene with
structural edges, one million scalar values and one million displacement
vectors at 1920x1080 with a two-second warm-up and ten-second orbit samples.

## Development

```bash
pip install -e .[dev]
pytest
python -m build
twine check dist/*
```

Native tkinter-gl lifecycle tests are opt-in:

```powershell
$env:ANY3DVIEW_RUN_GUI_TESTS = "1"
python -m pytest tests/test_gpu_widget.py
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
