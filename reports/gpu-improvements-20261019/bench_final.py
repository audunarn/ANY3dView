"""Same-source barge comparison; display tessellation only, no solver/mesher."""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--out', required=True)
p.add_argument('--backend', choices=['gpu','software'], required=True)
p.add_argument('--frames', type=int, default=12)
p.add_argument('--oracle', action='store_true')
p.add_argument('--profile', action='store_true')
p.add_argument('--spacing', type=float, default=.7)
a=p.parse_args()
sys.path.insert(0, str(Path('reports/performance-20261019/anyfem-src').resolve()))
sys.path.insert(0, 'C:/Github/ANYloads/src')
sys.path.insert(0, str(Path(a.source).resolve()))
import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, QPointF, Qt, QEvent
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from any3dview.qt import QtViewer
from any3dview.core import Point3D
from anyfem.application.examples import floating_barge_section
from anyfem.ui.scene import build_geometry_scene, FacePatch
from anyfem.presentation.viewport import SceneViewport
from anyfem.selection import MeshEntityRef

out=Path(a.out); out.mkdir(parents=True, exist_ok=True)
q=QApplication([])
v=QtViewer(backend=a.backend); v.resize(1000,700); v.show(); q.processEvents()
class View(SceneViewport):
    def __init__(self):
        self.initialize_state(None); self.canvas=v; self._point3d=Point3D
        self._commercial_selection=True
    def _apply_highlight(self): pass
    def _clear_hover_state(self): pass
view=View()
scene=build_geometry_scene(floating_barge_section())
dense=scene.copy(); dense.faces=[]; eid=0
for patch in scene.faces:
    if patch.ref is None or patch.ref.kind!='face':
        dense.faces.append(patch); continue
    pts=np.concatenate(patch.polygons); lo=pts.min(0); hi=pts.max(0)
    i,j=np.argsort(hi-lo)[-2:]
    ni=max(1,int(np.ceil((hi[i]-lo[i])/a.spacing)))
    nj=max(1,int(np.ceil((hi[j]-lo[j])/a.spacing)))
    polygons=[]; owners=[]
    for ii in range(ni):
        for jj in range(nj):
            vertices=[]
            for u,t in [(ii,jj),(ii+1,jj),(ii+1,jj+1),(ii,jj+1)]:
                pt=lo.copy(); pt[i]=lo[i]+u*(hi[i]-lo[i])/ni
                pt[j]=lo[j]+t*(hi[j]-lo[j])/nj; vertices.append(pt)
            polygons.append(np.array(vertices)); eid+=1
            owners.append((patch.ref,MeshEntityRef('element',eid),MeshEntityRef('element_face',(eid,0))))
    dense.faces.append(FacePatch(ref=patch.ref,polygons=polygons,colors=['#cbd5e1']*len(polygons),outline='#334155',polygon_owners=owners))
view._scene=dense; view._render_scene(dense,reset_view=True); q.processEvents()
fit_distance=float(np.linalg.norm(np.asarray(v.camera.position.to_tuple())-v.camera.target.to_tuple()))
def paint():
    if a.backend=='gpu':
        v.surface.grabFramebuffer()
        v._renderer.ctx.finish()
    else: v.surface.repaint()
def timing(fn, count):
    values=[]
    for i in range(count):
        start=time.perf_counter(); fn(i); values.append((time.perf_counter()-start)*1000)
    return dict(samples_ms=values,median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
def save(name):
    v.capture_image().save(out/(name+'.png'))
def orbit(i):
    v.camera.set_orbit(azimuth=.6+i*.018,elevation=.35+i*.003,distance=fit_distance)
    v._renderer.pick_dirty=True; v.redraw(); paint()
if a.oracle:
    h=next(e['handle'] for e in v._entries.values() if e['handle'].mesh.triangle_count)
    records={}
    def capture(name):
        v.redraw(); paint(); save(name)
        records[name]=[[(hit.key,hit.primitive) for hit in v.query_point(x,y,front_only=True)]
                       for y in range(250,451,100) for x in range(300,701,100)]
    orbit(0); capture('unchanged')
    h.set_selected_elements([0]); capture('selection')
    transform=np.eye(4); transform[2,3]=.1; h.set_transform(transform); capture('transform')
    displacement=np.zeros_like(h.mesh.positions); displacement[:,2]=.05
    h.update_displacements(displacement); capture('deformation')
    active=np.ones(h.mesh.element_count,dtype=np.uint8); active[::2]=0
    h.set_active_elements(active); capture('active')
    v.set_section_plane(Point3D(1.,0.,0.),0.); capture('section')
    v.clear_section_plane(); h.set_visible(False); capture('hidden')
    h.set_visible(True); h.update_positions(h.mesh.positions+np.array([.02,0.,0.])); capture('positions')
    (out/'oracle.json').write_text(json.dumps(records,indent=2))
    v.destroy(); q.processEvents(); print('8 mutation images and 120 pick samples'); sys.exit(0)
orbit(0); save('initial')
frames_before=v._renderer.frame_count
result={'backend':v.backend_name,'source':str(Path(a.source).resolve()),'python':sys.version,
        'platform':platform.platform(),'viewport':v.viewport_size,'dpr':v.pixel_ratio,
        'fixture':'Current Examples floating_barge_section; planar display tessellation at0.7m, beams retained; no scientific mesh',
        'triangles':sum(e['handle'].mesh.triangle_count for e in v._entries.values()),
        'entries':len(v._entries),'orbit':timing(orbit,a.frames)}
result['orbit']['rendered_frames']=v._renderer.frame_count-frames_before
assert result['orbit']['rendered_frames'] >= a.frames, 'Timing omitted actual rendering'
save('orbit')
if a.profile:
    import cProfile, pstats
    profile=cProfile.Profile(); profile.enable()
    for _ in range(3): orbit(a.frames-1)
    profile.disable(); profile.dump_stats(out/'orbit.prof')
    with (out/'profile.txt').open('w') as stream:
        pstats.Stats(profile,stream=stream).sort_stats('cumtime').print_stats(30)
result['diagnostics']=v.diagnostic_snapshot()

# Diagnostic timings: same scene, completed framebuffer, synchronous real picks.
probe={}
for label, dirty in [('cached_pick',False),('dirty_pick',True)]:
    def query(i):
        if dirty: v._renderer.pick_dirty=True
        v.query_point(500+i%5,350,front_only=True)
    query(0)
    probe[label]=timing(query,20)
probe['draw_calls']=v._renderer.draw_calls
probe['spacing']=a.spacing
probe['actual_triangles']=sum(c.mesh.triangle_count for g in v._renderer._groups.values() for _,c in g.chunks)
probe['chunks']=sum(len(g.chunks) for g in v._renderer._groups.values())
keys=[]
for yy in range(200,501,50):
    for xx in range(300,701,50):
        for hit in v.query_point(xx,yy,front_only=True):
            if hit.key not in keys: keys.append(hit.key)
probe['hit_key_count']=len(keys)
if keys:
    def highlight(i):
        v._set_hit_preselection(keys[i%len(keys)] if i%2 else None)
        paint()
    probe['highlight_and_paint']=timing(highlight,12)
v._set_hit_preselection(None)
if keys:
    pick_passes=[0]; old_render_pick=v._renderer.render_pick
    def count_pick(*args,**kwargs):
        pick_passes[0]+=1
        return old_render_pick(*args,**kwargs)
    v._renderer.render_pick=count_pick
    v.query_point(500,350,front_only=True)
    pick_passes[0]=0
    def highlight_pick(i):
        v._set_hit_preselection(keys[i%len(keys)] if i%2 else None)
        v.query_point(500,350,front_only=True)
    probe['highlight_then_pick']=timing(highlight_pick,20)
    probe['highlight_pick_passes']=pick_passes[0]
    v._renderer.render_pick=old_render_pick
    v._set_hit_preselection(None)

result['gpu_sweep']=probe

def hits():
    return [[(h.key,h.primitive) for h in v.query_point(x,y,front_only=True)]
            for y in range(200,501,50) for x in range(300,701,50)]
result['picks']=hits()
# Measure index rebuild across pure scalar edits, independent of GPU pick route.
handle=next(e['handle'] for e in v._entries.values() if e['handle'].mesh.triangle_count)
v._projected_selection_index()
def field(i):
    handle.update_element_scalars(np.full(handle.mesh.element_count,float(i)))
    v._projected_selection_index()
result['field_index']=timing(field,4)
# Cached animation: topology-identical scalar frames should avoid reupload.
v.begin_animation_cache()
for i in range(3):
    handle.update_element_scalars(np.full(handle.mesh.element_count,float(i)))
    displacement=np.zeros_like(handle.mesh.positions); displacement[:,2]=.02*i
    handle.update_displacements(displacement)
    v.capture_animation_frame()
uploads=getattr(v._renderer,'geometry_uploads',0)
def animate(i):
    v._show_animation_frame(i%3); paint()
result['animation_first']=timing(animate,6)
result['animation']=timing(animate,30)
if a.profile:
    ap=cProfile.Profile(); ap.enable()
    for i in range(6): animate(i)
    ap.disable(); ap.dump_stats(out/'animation.prof')
    with (out/'animation-profile.txt').open('w') as stream:
        pstats.Stats(ap,stream=stream).sort_stats('cumtime').print_stats(30)
result['animation']['geometry_uploads']=getattr(v._renderer,'geometry_uploads',0)-uploads
save('animation'); result['animation_picks']=hits(); v.stop_animation()
# Burst uses actual surface mouse handler, no synthetic direct pick bypass.
calls=[0]; original=v._hover_select
def hover(event): calls[0]+=1; return original(event)
v._hover_select=hover
v.surface.underMouse=lambda:True  # Deterministic synthetic hover, independent of desktop cursor.
start=time.perf_counter()
for i in range(100):
    ev=QMouseEvent(QEvent.MouseMove,QPointF(400+i,300),QPointF(400+i,300),Qt.NoButton,Qt.NoButton,Qt.NoModifier)
    v.surface.mouseMoveEvent(ev)
timer=getattr(v,'_hover_timer',None)
if timer is not None and timer.isActive():
    # Explicit batch boundary makes this a dispatch-count comparison;
    # desktop focus/leave events must not masquerade as reduced pick work.
    timer.stop(); timer.timeout.emit()
loop=QEventLoop(); QTimer.singleShot(160,loop.quit); loop.exec()
result['hover']={'calls':calls[0],'burst_and_settle_ms':(time.perf_counter()-start)*1000,
                 'explicit_batch_delivery':True}
polls=[0]
timer=v._update_poll_id
if timer is not None and hasattr(timer,'timeout'): timer.timeout.connect(lambda:polls.__setitem__(0,polls[0]+1))
loop=QEventLoop(); QTimer.singleShot(200,loop.quit); loop.exec()
result['idle']={'window_ms':200,'timer_ticks':polls[0]}
result['hashes']={name:hashlib.sha256((Path(a.source)/'any3dview'/name).read_bytes()).hexdigest()
                  for name in ['qt.py','retained_viewer.py','scheduler.py','gpu/renderer.py']}
(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
v.destroy(); q.processEvents()
print(json.dumps({k:result[k] for k in ['backend','triangles','orbit','field_index','animation','hover','idle']}))
