# ANY3dView 0.5.5 qualification

Date: 2026-09-03

## Automated verification

- Core and renderer tests: 154 passed, 9 opt-in GUI tests skipped.
- Native TkGL lifecycle tests on Python 3.13: 9 passed.
- Native TkGL lifecycle tests on Python 3.14: 9 passed.
- Coordinated ANYtk3D source tests: 88 passed, 85 GUI tests skipped.
- Coordinated wheel installation: import, shared class identity, and native GPU
  creation/destruction passed.
- Coordinated source-distribution installation: import, shared class identity,
  and native GPU creation/destruction passed.
- `twine check`: wheel and source distribution passed.

The opt-in ANYtk3D GUI suite passed 167 tests; its desktop screenshot test was
inconclusive because the operating system denied `ImageGrab.grab` in the test
session. This does not affect rendering, interaction, packaging, or the GPU
host replacement.

## Performance verification

The full 1920 x 1080 reference benchmark used 999,698 opaque triangles on an
NVIDIA GeForce RTX 3090 Ti. It recorded:

- Opaque: 3,635 FPS; CPU median 0.093 ms; GPU median 0.110 ms.
- Structural edges: 969 FPS; CPU median 0.128 ms; GPU median 0.769 ms.
- Scalar upload: 9.27 ms.
- Displacement upload: 52.10 ms.
- Cached pick median: 0.034 ms.
- Camera-motion geometry uploads: 0.
- Thirty-second idle CPU: 0.0%; rendered frames: 0.

All established release gates pass. The machine-readable result is
`benchmark-results/gpu-0.5.5.json`.
