"""Software exposure caching and optional hover scheduling, without windows."""
from types import SimpleNamespace
import pytest
pytest.importorskip('PySide6')
from PySide6.QtCore import QObject,QCoreApplication
from PySide6.QtGui import QImage,QPainter,QColor
from any3dview.core import Camera3D
from any3dview import qt

@pytest.mark.parametrize('ratio',[1.,1.5,2.])
def test_cached_image_matches_direct_paint_and_only_rebuilds_on_change(monkeypatch,ratio):
    width,height=80,60;calls=[]
    def image():
        out=QImage(round(width*ratio),round(height*ratio),QImage.Format_RGBA8888)
        out.setDevicePixelRatio(ratio);return out
    destination=image();surface=SimpleNamespace(_cached_image=None,_cached_key=None,
        width=lambda:width,height=lambda:height,devicePixelRatioF=lambda:ratio)
    owner=SimpleNamespace(_closed=False,_software_revision=0,camera=Camera3D(),bg='white')
    surface.owner=owner
    def draw(painter):
        calls.append(1);painter.scale(1/ratio,1/ratio)
        painter.fillRect(0,0,round(width*ratio),round(height*ratio),QColor('white'))
        painter.fillRect(13,17,37,23,QColor('red'));painter.end()
    owner._paint_software=draw
    direct=image();draw(QPainter(direct));calls.clear()
    def painter_factory(target):return QPainter(destination if target is surface else target)
    painter_factory.Antialiasing=QPainter.Antialiasing
    monkeypatch.setattr(qt,'QPainter',painter_factory)
    qt._PaintSurface.paintEvent(surface,None)
    assert bytes(destination.constBits())==bytes(direct.constBits())
    for _ in range(5):qt._PaintSurface.paintEvent(surface,None)
    assert len(calls)==1
    owner._software_revision+=1;qt._PaintSurface.paintEvent(surface,None)
    assert len(calls)==2
    owner.camera.orbit(.1,0);qt._PaintSurface.paintEvent(surface,None)
    assert len(calls)==3
    owner._closed=True;qt._PaintSurface.paintEvent(surface,None)
    assert len(calls)==3

class Hover(QObject):
    _queue_hover=qt.QtViewer._queue_hover
    _cancel_hover=qt.QtViewer._cancel_hover
    _deliver_hover=qt.QtViewer._deliver_hover
    def __init__(self):
        super().__init__();self._closed=False;self._timers=set();self._hover_timer=None;self._pending_hover=None
        self._drag='';self._selection_dragging=False
        self.surface=SimpleNamespace(underMouse=lambda:True)
        self.hits=[];self._hover_select=lambda e:self.hits.append((e.x,e.y))

@pytest.fixture
def hover():
    app=QCoreApplication.instance() or QCoreApplication([])
    h=Hover();yield h
    h._cancel_hover()


def test_hover_coalesces_latest_position_and_reuses_timer(hover):
    hover._queue_hover(SimpleNamespace(x=1,y=2));timer=hover._hover_timer
    hover._queue_hover(SimpleNamespace(x=3,y=4))
    assert hover.hits==[] and hover._timers=={timer}
    assert timer.isSingleShot() and timer.interval()==120
    hover._deliver_hover();assert hover.hits==[(3,4)]

@pytest.mark.parametrize('cancel',['cancel','leave','replace','drag','selection','close'])
def test_obsolete_hover_never_builds_index(hover,cancel):
    hover._queue_hover(SimpleNamespace(x=1,y=2))
    if cancel=='cancel':hover._cancel_hover()
    elif cancel=='leave':hover.surface.underMouse=lambda:False
    elif cancel=='replace':hover.surface=SimpleNamespace(underMouse=lambda:True)
    elif cancel=='drag':hover._drag='orbit'
    elif cancel=='selection':hover._selection_dragging=True
    elif cancel=='close':hover._closed=True
    hover._deliver_hover();assert hover.hits==[]


def test_mouse_press_cancels_hover_then_preserves_exact_click_dispatch():
    from PySide6.QtCore import Qt
    calls=[]
    owner=SimpleNamespace(_cancel_hover=lambda:calls.append('cancel'),_press_select=lambda e:calls.append(('click',e)))
    event=SimpleNamespace(button=lambda:Qt.LeftButton,accept=lambda:calls.append('accepted'))
    surface=SimpleNamespace(owner=owner,setFocus=lambda:None,_event=lambda e:'exact-event')
    qt._InputSurface.mousePressEvent(surface,event)
    assert calls==['cancel',('click','exact-event'),'accepted']


def test_actual_software_scene_highlight_visibility_and_section_match_uncached(monkeypatch):
    import numpy as np
    from any3dview import MeshArrays
    from any3dview.retained_viewer import RetainedViewer
    v=RetainedViewer();v.initialize_state(width=120,height=90)
    v._software_revision=0;v._renderer=qt._SoftwareResources();v._hud=qt._PainterHud()
    v._software_prep={};v._prune_software_prep=qt.QtViewer._prune_software_prep.__get__(v)
    v._software_chunk_prep=qt.QtViewer._software_chunk_prep.__get__(v)
    v._software_prep_signature=qt.QtViewer._software_prep_signature.__get__(v)
    v._render_hud=lambda size:None  # Scene renderer comparison; no font engine/window.
    surface=SimpleNamespace(owner=v,_cached_image=None,_cached_key=None,
        width=lambda:120,height=lambda:90,devicePixelRatioF=lambda:1.,update=lambda:None)
    v.surface=surface
    v._host=SimpleNamespace(make_current=lambda:None,framebuffer_size=lambda:(120,90),request_redraw=lambda:qt.QtViewer.request_redraw(v))
    v._paint_software=lambda painter:qt.QtViewer._paint_software(v,painter)
    mesh=MeshArrays(np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]]),np.array([[0,1,2],[0,2,3]],dtype=np.uint32))
    handle=v.add_mesh_arrays(mesh,tags='face',color='#123456',cull_backface=False)
    destination=QImage(120,90,QImage.Format_RGBA8888)
    def painter_factory(target):return QPainter(destination if target is surface else target)
    painter_factory.Antialiasing=QPainter.Antialiasing
    monkeypatch.setattr(qt,'QPainter',painter_factory)
    for change in (lambda:None,lambda:v.set_highlight(['face']),lambda:v.set_section_plane((1,0,0),.5),lambda:handle.set_visible(False)):
        change()
        expected=QImage(120,90,QImage.Format_RGBA8888)
        v._paint_software(QPainter(expected))
        qt._PaintSurface.paintEvent(surface,None)
        assert bytes(destination.constBits())==bytes(expected.constBits())
        count=v._renderer.frame_count
        qt._PaintSurface.paintEvent(surface,None)
        assert v._renderer.frame_count==count

@pytest.mark.parametrize('mode,selecting,redraws',[('',False,0),('orbit',False,1),('',True,1)])
def test_cancel_idle_focus_does_not_invalidate_scene_but_active_overlay_does(mode,selecting,redraws):
    from any3dview.retained_viewer import RetainedViewer
    v=RetainedViewer();v.initialize_state();calls=[];v.redraw=lambda:calls.append(1)
    v._drag=mode;v._selection_dragging=selecting;v._selection_press=(1,2)
    v._cancel_interaction()
    assert len(calls)==redraws
    assert v._drag=='' and not v._selection_dragging and v._selection_press is None
