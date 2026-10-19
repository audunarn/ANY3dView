# TkGL native component provenance

- Component: TkGL 1.2.1
- Upstream: <https://github.com/3-manifolds/TkGL>
- Licence: the permissive terms in `license.terms`
- Distribution source: the platform artifacts shipped in the `tkinter-gl`
  1.1 wheel, whose bundled `tk/README` identifies these files as compiled
  TkGL and directs recipients to the TkGL project.
- Integrated: 2026-09-03

The native files and their licence notice are redistributed unchanged. The
GPL-licensed Python wrapper from `tkinter-gl` is not included. ANY3dView uses
its own small wrapper around TkGL's public Tcl commands (`tkgl`,
`makecurrent`, `swapbuffers`, `glversion`, and `extensions`).
