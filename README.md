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
pip install "ANY3dView[gpu]"       # ModernGL + tkinter-gl + Pillow capture
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
Incremental chunks can carry their own stable primitive ownership without
changing the legacy `handle.chunks` view:

```python
handle.add_chunk("crack-tip", local_mesh, owners=local_owner_table)
handle.replace_chunk("crack-tip", updated_mesh)  # preserves ownership
handle.set_chunk_ownership("crack-tip", replacement_owner_table)
```

`handle.chunk_records` and `handle.chunk_ownership(id)` expose the optional
chunk-local table and resolver. Triangle, line and point CSR spans are checked
against each chunk, so replacement cannot silently rebind local primitive
indices. Passing `None` to `set_chunk_ownership()` explicitly clears the
semantic mapping; a handle-level legacy tag remains a stable fallback binding.

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
deformation, active masks, distinct compact selection/preselection masks for
triangles, lines and points, sorted alpha, cached integer point picking, and
visible/through rectangle and lasso queries. Rendering is demand-driven. Text,
legends, rulers and interaction overlays use a cached Pillow-generated OpenGL
atlas rather than child Tk label widgets.
It also implements the established ANYtk3D scene surface (`add_faces`, lines,
markers, text, shape builders, camera presets, legends, highlighting,
animation and image capture), so existing scenes can be populated without a
renderer-specific branch. Backend-specific pixels and Tk Canvas item IDs are
not part of that portable contract.

`tkinter-gl` 1.1 has no OpenGL 3.3 profile token. On Windows, ANY3dView prefers
its OpenGL 4.1 core profile because recent NVIDIA drivers can terminate a
process inside the driver when TkGL and ModernGL share a legacy compatibility
context. Set `ANY3DVIEW_GL_PROFILE=legacy` before creating the viewer only when
using Windows hardware limited to OpenGL 3.3/4.0. ModernGL remains the final
version gate, and unsupported contexts fail with an actionable diagnostic.

`ViewerBackend`, `ViewerCapabilities`, `Pick` and `ViewerState` describe the
shared integration boundary. Applications can switch renderers
transactionally by populating a candidate and copying the view policy:

```python
candidate = create_viewer(parent, backend="gpu")
populate_scene(candidate)
candidate.apply_view_state(current.export_view_state())
candidate.pack(fill="both", expand=True)
current.destroy()
```

Both bundled backends expose `backend_name`, `event_widget`, `viewport_size`,
`project_point(s)`, `screen_ray()` and `unproject_to_plane()`. These replace
direct access to Tk canvas dimensions or private projection methods.

## LLM-safe viewer commands

ANY3dView 0.5.2 exposes a vendor-neutral JSON Schema 2020-12 command manifest.
It contains only JSON-safe camera, display, section, semantic selection,
visibility, observation, and undo/redo operations; it accepts no callbacks,
expressions, or executable objects.

```python
from any3dview import ViewerCommand, ViewerCommandController

controller = ViewerCommandController(viewer, entity_exists=model.contains_ref)
result = controller.execute(ViewerCommand(
    "viewer.visibility.isolate",
    {"entities": [{
        "source": "model", "model_id": str(model.uuid),
        "kind": "face", "key": 42,
    }]},
))
assert result.status == "ok"
```

`viewer_command_manifest()` is suitable for native tool calling or a strictly
validated JSON fallback and has no dependency on an LLM vendor. Hosts assign
trusted queue priorities, may pause AI work during pointer gestures, and get a
bounded 50-step transactional history. `set_model_identity()` clears history
when an application model changes; `revalidate()` removes stale same-model
semantic references after a revision.

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

The standalone benchmark records platform, driver, commit plus dirty-tree GPU
source digest, scene, median/p95 CPU and GPU frame times, upload and cached-pick
timings, camera-motion upload deltas, Python allocation deltas, a defined
30-second no-redraw idle sample, per-frame draw calls and array memory as JSON:

```powershell
$env:PYTHONPATH = "C:\Github\ANY3dView\src"
python C:\Github\ANY3dView\benchmarks\run_gpu.py `
  --output C:\Github\ANY3dView\benchmark-results\reference-current.json
```

It exercises approximately one million opaque triangles, the same scene with
structural edges, one million scalar values and one million displacement
vectors at 1920x1080 with a two-second warm-up per render scene and ten-second
orbit samples. Field latency is conservatively measured from the retained
handle update through a rendered frame and an OpenGL completion barrier after
the unchanged topology has already been synchronized.

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
