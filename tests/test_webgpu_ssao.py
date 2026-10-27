"""SSAO contact oracle, neutral cases, lifecycle and interaction policy."""
import os
import numpy as np
import pytest
from any3dview.webgpu.ssao import validated_settings


def test_invalid_settings_refused():
    for values in (('unknown',1,24),('Low',float('nan'),24),('High',-1,24),('Low',1,0),('Low',1,float('inf'))):
        with pytest.raises(ValueError):validated_settings(*values)


@pytest.mark.skipif(os.environ.get('ANY3DVIEW_TEST_WEBGPU')!='1',reason='opt-in hardware')
def test_ssao_contact_flat_plane_strength_radius_interaction_and_cleanup(tmp_path):
    from PySide6.QtWidgets import QApplication
    from any3dview.qt_webgpu import QtWebGPUViewer
    from PIL import Image
    app=QApplication.instance() or QApplication([]);v=QtWebGPUViewer(width=400,height=320,shading=False)
    try:
        v.surface.resize(400,320)
        floor=[[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]]
        v.add_faces([floor],colors=['#d8e0e8'],lit=False);v.fit_to_scene(redraw=False)
        def image(quality,strength=.8,radius=24):
            v.set_render_quality(samples=4,ambient_occlusion=quality,ao_strength=strength,ao_radius=radius)
            return np.asarray(v.capture_image())
        flat=image('Off');flat_ao=image('High')
        np.testing.assert_array_equal(flat,flat_ao)
        faces=[[[x,y,z] for x,y,z in coords] for coords in (
            ((-.6,-.6,0),(.6,-.6,0),(.6,-.6,1),(-.6,-.6,1)),
            ((.6,-.6,0),(.6,.6,0),(.6,.6,1),(.6,-.6,1)),
            ((.6,.6,0),(-.6,.6,0),(-.6,.6,1),(.6,.6,1)),
            ((-.6,.6,0),(-.6,-.6,0),(-.6,-.6,1),(-.6,.6,1)),
            ((-.6,-.6,1),(.6,-.6,1),(.6,.6,1),(-.6,.6,1))) ]
        existing=set(v._entries)
        v.add_faces(faces,colors=['#d8e0e8']*5,lit=False,outline='',opacity=0.)
        np.testing.assert_array_equal(image('High'),flat_ao)
        for key in set(v._entries)-existing:v._entries[key]['handle'].remove()
        v.add_faces(faces,colors=['#d8e0e8']*5,lit=False)
        off=image('Off');passes=v._webgpu.ao.passes
        image('Off');assert v._webgpu.ao.passes==passes
        low=image('Low');high=image('High')
        difference=off[:,:,:3].astype(int)-high[:,:,:3].astype(int)
        assert difference.max()>5 and (difference>2).sum()>100
        np.testing.assert_array_equal(off[2,2],high[2,2])
        np.testing.assert_array_equal(image('High',0),off)
        assert not np.array_equal(image('High',radius=8),image('High',radius=48))
        full=v._webgpu.ao.textures['linear']
        v._drag='orbit';image('High');assert v.renderer_diagnostics['effective_ambient_occlusion']=='Low'
        redraw=[];original=v.redraw;v.redraw=lambda:redraw.append(True)
        v._release(None);assert redraw==[True] and not v._drag
        v.redraw=original
        image('High');assert v.renderer_diagnostics['effective_ambient_occlusion']=='High'
        assert v._webgpu.ao.textures['linear'] is full
        v._zoom(1.);image('High');assert v.renderer_diagnostics['effective_ambient_occlusion']=='Low'
        v.after_cancel(v._wheel_finish_after_id);v._finish_wheel_zoom()
        image('High');assert v.renderer_diagnostics['effective_ambient_occlusion']=='High'
        state=(v._webgpu.samples,v._webgpu.ao_quality)
        with pytest.raises(ValueError):v.set_render_quality(samples=1,ambient_occlusion='bad')
        assert (v._webgpu.samples,v._webgpu.ao_quality)==state
        v.set_section_plane((1,0,0),0);image('High');v.clear_section_plane()
        v.surface.resize(420,300);image('Low')
        for name,pixels in (('ssao-off',off),('ssao-low',low),('ssao-high',high)):
            Image.fromarray(pixels).save(tmp_path/f'{name}.png')
    finally:
        v._drag='';v.destroy();app.processEvents()
    assert not v._webgpu.ao.pool and not v._webgpu.ao.buffers
