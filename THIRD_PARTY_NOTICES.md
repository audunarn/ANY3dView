# Third-party notices

ANY3dView 0.5.5 includes the native TkGL 1.2.1 component in the optional GPU
backend. TkGL is distributed under its own permissive licence. Its required
notice and full terms are preserved verbatim at
`src/any3dview/gpu/tkgl/license.terms` and are included in both the wheel and
source distribution.

The Python wrapper in `any3dview.gpu._tkgl` is an original ANY3dView
implementation of the public TkGL Tcl widget interface. The `tkinter-gl`
Python package is neither copied nor installed by ANY3dView 0.5.5.

ANY3dView also depends on NumPy and optionally uses ModernGL, Pillow,
ANYgeometry, and mapbox-earcut. These packages are not bundled and remain
subject to the licence terms of their respective distributions.

The optional `qt` extra installs PySide6 dynamically. PySide6 is not copied into
ANY3dView wheels or source archives. This candidate uses the LGPL-3.0-only
licensing option for Qt for Python and the LGPL-compatible Qt Widgets/OpenGL
Widgets modules. PySide6's installed distribution supplies its notices and
license texts. The alternative GPL route is not selected, and this notice does
not authorize adding GPL-only Qt modules. A future bundled application installer
requires a separate review of its LGPL notices and replacement/relinking route.

The experimental `webgpu` extra installs wgpu-py 0.32.0, whose installed Python
distribution supplies a BSD-2-Clause license. The original WGSL renderer here is
ANY3dView code. wgpu-native is upstream MIT/Apache-2.0; its native distribution
and transitive components require their own notices. These dependencies are not
copied into ANY3dView source archives or wheels. This notice is not a complete
native-wheel or bundled-application license audit. The optional Qt and Pillow
terms above continue to apply. Rendercanvas 2.7.2 was investigated (installed
BSD-2-Clause license) but is not used or required by this implementation.
