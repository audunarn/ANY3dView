"""Callout layout is deterministic, bounded and independent of source anchors."""
import os
import numpy as np
import pytest
from any3dview.callouts import layout_callouts,_overlap

measure=lambda text,font:(len(text)*7.,14.)
font=('Segoe UI',9,'bold')

def test_deterministic_nonoverlap_elision_viewport_and_no_anchor_mutation():
    items=[{'key':f'pin:{i:03}','anchor':(160.,110.),'text':'Full engineering caption '+str(i)+' value 123456789 '*8,'font':font} for i in range(100)]
    before=[dict(i) for i in items]
    placements,omitted=layout_callouts(items,320,240,measure)
    reverse=layout_callouts(list(reversed(items)),320,240,measure)
    assert reverse==(placements,omitted) and items==before
    assert 0<len(placements)<=64 and omitted==100-len(placements)
    assert all(p.anchor==(160.,110.) and p.text.endswith('\u2026') for p in placements)
    for i,p in enumerate(placements):
        x0,y0,x1,y1=p.rect
        assert 0<=x0<x1<=320 and 0<=y0<y1<=240
        assert not any(_overlap(p.rect,q.rect) for q in placements[:i])
        assert not (x0<=p.anchor[0]<=x1 and y0<=p.anchor[1]<=y1)

def test_small_viewport_and_long_label_are_bounded():
    item={'key':'a','anchor':(12,16),'text':'word '*10000,'font':font}
    assert layout_callouts([item],10,10,measure)==((),1)
    placements,omitted=layout_callouts([item],100,80,measure)
    assert len(placements)+omitted==1
    if placements:assert len(placements[0].text)<=11

@pytest.mark.skipif(os.environ.get('ANY3DVIEW_RUN_QT_TESTS')!='1',reason='Qt opt-in')
@pytest.mark.parametrize('ratio',[1.,2.])
def test_callouts_shared_hud_plain_labels_groups_clipping_and_dpi(ratio):
    from PySide6.QtWidgets import QApplication
    from any3dview.qt import QtViewer
    from any3dview.webgpu.presentation import render_hud_image
    app=QApplication.instance() or QApplication([])
    v=QtViewer(backend='software',width=320,height=240)
    try:
        v.surface.resize(320,240);v.set_top_view();v.set_axis_indicator(False)
        v._sync_pixel_ratio=lambda:ratio
        v._pixel_ratio=ratio
        # Projection here is explicitly physical pixels, as native/bitmap cameras are.
        v.project_point=lambda point:(160*ratio,110*ratio,10.)
        v.add_text((0,0,0),'plain caption',color='#ff0000')
        with v.annotation_group() as group:
            v.add_text((0,0,0),'Full pinned field value: 123.456 Pa '*12,callout=True,callout_key='pin:01')
            v.add_text((0,0,0),'other pin',callout=True,callout_key='pin:02')
        original=[(x['point'],x['text']) for x in v._world_text]
        image=render_hud_image(v,round(320*ratio),round(240*ratio))
        assert len(v._callout_placements)==2 and v._callout_omitted==0
        assert [(x['point'],x['text']) for x in v._world_text]==original
        first=v._callout_placements[0]
        assert first.key=='pin:01' and first.text.endswith('\u2026')
        # Logical backplate positions become physical image pixels at either DPR.
        pixels=np.asarray(image.constBits()).reshape(image.height(),image.width(),4)
        x0,y0,x1,y1=first.rect
        assert pixels[round((y1-2)*ratio),round((x1-2)*ratio),3]>0
        assert image.width()==round(320*ratio)
        group.remove();assert len(v._world_text)==1 and not v._world_text[0]['callout']
        v.set_section_plane((1,0,0),1.)
        v.add_text((0,0,0),'clipped pin',callout=True,callout_key='hidden')
        render_hud_image(v,round(320*ratio),round(240*ratio))
        assert not v._callout_placements
    finally:v.destroy();app.processEvents()


@pytest.mark.skipif(os.environ.get('ANY3DVIEW_TEST_WEBGPU')!='1',reason='GPU opt-in')
def test_webgpu_callouts_capture_high_resolution_and_clipping():
    from PySide6.QtWidgets import QApplication
    from any3dview.qt_webgpu import QtWebGPUViewer
    from any3dview.arrays import MeshArrays
    app=QApplication.instance() or QApplication([]);v=QtWebGPUViewer(width=300,height=240)
    try:
        v.surface.resize(300,240);v.set_top_view()
        v.add_mesh_arrays(MeshArrays([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],[[0,1,2],[0,2,3]]),cull_backface=False)
        v.fit_to_scene(redraw=False)
        v.add_text((0,0,0),'Pinned engineering field value '*12,callout=True,callout_key='inspection:01')
        assert v.capture_image().size==(300,240) and len(v._callout_placements)==1
        assert v.capture_image(width=600,height=480).size==(600,480) and len(v._callout_placements)==1
        v.set_section_plane((1,0,0),.5);v.capture_image();assert not v._callout_placements
    finally:v.destroy();app.processEvents()
