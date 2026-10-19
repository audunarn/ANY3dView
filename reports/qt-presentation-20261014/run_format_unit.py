import subprocess,time,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
label=sys.argv[1]
cmd=[sys.executable,'-m','pytest','tests/test_qt_surface_format.py','tests/test_qt_deferred_failure.py','-q','-p','no:cacheprovider']
start=time.perf_counter()
with (root/f'{label}.log').open('w') as out:
 p=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
 try:code=p.wait(timeout=30)
 except subprocess.TimeoutExpired:
  subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=out,stderr=out);code='timeout'
receipt={'command':cmd,'status':code,'process_wall_s':time.perf_counter()-start,'category':'fake/unit; no QApplication or window'}
(root/f'{label}-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
