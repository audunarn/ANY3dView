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
