import subprocess,time,json,sys,hashlib,os
from pathlib import Path
root=Path(__file__).resolve().parent
budget=73.4644437; used=0.
for swap in (1,0):
 cmd=[sys.executable,str(root/'probe.py'),'--swap',str(swap),'--out',str(root/f'swap{swap}.json')]
 start=time.perf_counter();status=None
 with (root/f'swap{swap}.log').open('w') as out:
  process=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
  try:status=process.wait(timeout=min(20,budget-used))
  except subprocess.TimeoutExpired:
   subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=out,stderr=out);status='timeout'
 elapsed=time.perf_counter()-start;used+=elapsed
 receipt=dict(command=cmd,python=sys.executable,probe_sha256=hashlib.sha256((root/'probe.py').read_bytes()).hexdigest(),status=status,process_wall_s=elapsed,cumulative_gui_s=526.5355563+used,remaining_gui_s=budget-used)
 (root/f'swap{swap}-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
 if status!=0:break
