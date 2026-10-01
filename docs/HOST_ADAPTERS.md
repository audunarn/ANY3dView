# Viewer host adapters

ANY3DView separates renderer behavior from desktop widget ownership.
`ViewerBackend` remains the renderer-neutral drawing, camera, selection and
retained-mesh contract. `ViewerHostAdapter` creates the concrete widget for a
desktop toolkit.

`create_viewer(parent, backend=..., host=...)` uses the supplied host. Omitting
`host` selects `TkViewerHostAdapter`, preserving the existing GPU/Tk and
ANYtk3D fallback behavior.

A future Qt adapter should implement `toolkit_name` and `create_viewer`, return
a QWidget satisfying `ViewerBackend`, and keep GPU fallback decisions inside
the adapter. It must not add Qt imports to ANY3DView's contracts, arrays,
selection, retained-mesh, camera or scene modules.
# Visual animation rates

The shared retained viewer accepts positive finite fractional rates in
`play_animation(fps=...)`, including 0.5 frames/s. The host schedules the
nearest millisecond interval, with a one-millisecond minimum. Invalid rates
and intervals beyond the host's signed 32-bit timer range are refused before
changing an active animation. Playback timing changes
presentation only; cached scene frames and engineering result times remain
unchanged. Coordinated Qt migration evidence is maintained in ANYfem's
`docs/QT_CONVERSION.md`; hardware qualification remains separate.
