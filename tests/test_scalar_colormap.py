"""Actual shader endpoint pixels must preserve scalar colour distinctions."""

import numpy as np
import pytest

from any3dview import Camera3D, MeshArrays, MeshHandle, Point3D
from any3dview.core import get_color_stops, parse_color

moderngl = pytest.importorskip("moderngl")


@pytest.mark.parametrize("location", ["element", "node"])
def test_scalar_endpoints_clamp_palette_and_switch_without_geometry_upload(location):
    from any3dview.gpu.renderer import ModernGLRenderer

    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"no standalone OpenGL 3.3 context: {error}")
    renderer = ModernGLRenderer(context)
    framebuffer = context.simple_framebuffer((128, 128), components=4)
    try:
        values = np.array([10, 30], dtype=np.float32)
        fields = {f"{location}_scalars": values if location == "element" else np.repeat(values, 3)}
        mesh = MeshArrays(
            np.array([[-2.,-.5,0.],[-1.,-.5,0.],[-1.5,.5,0.],
                      [1.,-.5,0.],[2.,-.5,0.],[1.5,.5,0.]],dtype=np.float32),
            np.array([[0,1,2],[3,4,5]],dtype=np.uint32),
            **fields,
        )
        handle = MeshHandle(mesh)
        renderer.add_mesh(handle, cull_backface=False, lit=False)
        camera = Camera3D()
        stops = get_color_stops()
        endpoints = [np.array(parse_color(row[1]),dtype=int) for row in (stops[0],stops[-1])]

        def draw_and_sample():
            renderer.render(camera,(128,128),target=framebuffer,shading_enabled=False,samples=0)
            pixels=[]
            for point in (Point3D(-1.5,-.1,0.),Point3D(1.5,-.1,0.)):
                projected=camera.project_point(point,128,128)
                assert projected is not None
                x,y=(int(round(value)) for value in projected)
                pixels.append(np.frombuffer(framebuffer.read(viewport=(x,127-y,1,1),components=3,alignment=1),dtype=np.uint8).astype(int))
            return pixels

        pixels = draw_and_sample()
        for actual, expected in zip(pixels,endpoints):
            assert np.max(np.abs(actual-expected)) <= 2, (actual,expected)
        uploaded = renderer.geometry_uploads
        generations = handle.generations
        if location == "element":
            handle.update_element_scalars(values[::-1])
        else:
            handle.update_node_scalars(np.repeat(values[::-1],3))
        pixels = draw_and_sample()
        for actual, expected in zip(pixels,endpoints[::-1]):
            assert np.max(np.abs(actual-expected)) <= 2, (actual,expected)
        assert renderer.geometry_uploads == uploaded
        assert handle.generations.topology == generations.topology
        assert handle.generations.position == generations.position
        assert handle.generations.scalar > generations.scalar
    finally:
        framebuffer.release()
        renderer.release()
        context.release()
