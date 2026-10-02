# Viewer host adapters

ANY3DView separates renderer behavior from desktop widget ownership.
`ViewerBackend` remains the renderer-neutral drawing, camera, selection and
retained-mesh contract. `ViewerHostAdapter` creates the concrete widget for a
desktop toolkit.

`create_viewer(parent, backend=..., host=...)` uses the supplied host. Omitting
`host` selects `TkViewerHostAdapter`, preserving the existing GPU/Tk and
ANYtk3D fallback behavior.

`QtViewerHostAdapter` is available from `any3dview.qt` with the optional `qt`
extra. It returns a QWidget implementing the same retained scene, camera,
selection and animation contracts. Use `ANY3dView[gpu,qt]` for OpenGL rendering;
`ANY3dView[qt]` supplies the QPainter software backend.

```python
from any3dview import create_viewer
from any3dview.qt import QtViewerHostAdapter

viewer = create_viewer(parent, backend="auto", host=QtViewerHostAdapter())
layout.addWidget(viewer)
```

Create and mutate widgets on the Qt GUI thread. For a startup child, create the
software backend first and request Automatic after the containing window is
exposed, as ANYfem does. Explicit GPU requests report initialization failure;
Automatic preserves the retained scene when falling back to software. Connect
`render_error` for diagnostics and `backend_changed` to reinstall any application
event filters on `event_widget` after a runtime fallback. Call `destroy()` during
application teardown to stop timers and release the host's resources.

`export_view_state()` / `apply_view_state()` transfer camera, display, lighting,
selection and visibility settings. The application repopulates scene geometry
before replacing its old widget. `capture_image()` returns the rendered viewport
in physical pixels, including the Qt device-pixel ratio. QOpenGLWidget owns
context creation and presentation; clients must not swap its buffers themselves.

Qt remains optional: importing the headless viewer contracts does not import Qt
or Tk. Software rendering does not advertise GPU-only picking, shader
deformation, line occlusion or stippled transparency.

## Verification

Pull requests and main pushes run headless Python/platform and geometry matrices,
Tk desktop regression, and installed Qt host checks. Windows Qt CI exercises
software rendering. Linux Qt CI requires a real Qt/XCB context through Mesa under
Xvfb and runs both software and GPU contracts. Physical GPU/driver acceptance is
recorded separately. Enable `ANY3DVIEW_RUN_QT_TESTS=1` for Qt tests and additionally
`ANY3DVIEW_RUN_QT_GPU_TESTS=1` when a usable GPU context is required.

## Visual animation rates

The shared retained viewer accepts positive finite fractional rates in
`play_animation(fps=...)`, including 0.5 frames/s. The host schedules the
nearest millisecond interval, with a one-millisecond minimum. Invalid rates
and intervals beyond the host's signed 32-bit timer range are refused before
changing an active animation. Playback timing changes
presentation only; cached scene frames and engineering result times remain
unchanged. Coordinated Qt migration evidence is maintained in ANYfem's
`docs/QT_CONVERSION.md`; hardware qualification remains separate.
