import sys,time,json,runpy,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--swap',type=int);p.add_argument('--out');a=p.parse_args()
runpy.run_path('C:/Github/ANYfem/run_gui.py',run_name='diagnostic_launcher')
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import any3dview.qt as qt
original=qt.QSurfaceFormat
# Experimental control only; no product default is changed.
def format_factory():
 f=original();f.setSwapInterval(a.swap);return f
format_factory.CoreProfile=original.CoreProfile
qt.QSurfaceFormat=format_factory
from anyfem.ui.qt.app import QtFemWindow
from anyfem.application.examples import floating_barge_section
app=QApplication([]);w=QtFemWindow(project=floating_barge_section());w.setWindowTitle('Bounded presentation diagnosis');w.showMaximized()
rows=[];beats=[];requests={};state={'id':0,'rendered':None};last=[None];effective={}
def setup():
 v=w.viewport.canvas;s=v.event_widget
 effective.update(backend=v.backend_name,context_swap=s.context().format().swapInterval(),window_swap=w.windowHandle().format().swapInterval(),size=v.viewport_size)
 render=v._renderer.render
 def measured(*args,**kw):
  identity=state['id'];start=time.perf_counter_ns();result=render(*args,**kw)
  # This identity belongs to the camera actually submitted to render, not
  # arbitrary preceding hover/button events.
  state['rendered']=(identity,start,time.perf_counter_ns());return result
 v._renderer.render=measured
 def swapped():
  if state['rendered']:
   i,start,end=state['rendered'];state['rendered']=None
   if i in requests:rows.append(dict(id=i,request=requests[i],render_start=start,render_end=end,swapped=time.perf_counter_ns()))
 s.frameSwapped.connect(swapped)
 def tick():
  now=time.perf_counter_ns()
  if last[0] is not None:beats.append((now-last[0])/1e6)
  last[0]=now;state['id']+=1;requests[state['id']]=now;v.camera.orbit(.008,.003);v.redraw()
 timer=QTimer(w);timer.timeout.connect(tick);timer.start(16);state['timer']=timer
 QTimer.singleShot(4000,finish)
def finish():
 state['timer'].stop()
 import numpy as np
 def stats(x):return {'count':len(x),'median_ms':float(np.median(x)),'p95_ms':float(np.percentile(x,95)),'max_ms':max(x)} if x else {}
 result=dict(effective=effective,requests=len(requests),frames=rows,heartbeat=stats(beats),request_to_swap=stats([(r['swapped']-r['request'])/1e6 for r in rows]),render_cpu=stats([(r['render_end']-r['render_start'])/1e6 for r in rows]))
 Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='frames'}));w.session.mark_saved();w.close();app.quit()
QTimer.singleShot(1300,setup);app.exec()
