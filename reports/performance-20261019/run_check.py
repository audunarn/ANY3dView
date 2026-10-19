"""Bounded command with exact process-wall accounting and retained output."""
import json, pathlib, subprocess, sys, time
out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=False)
timeout=float(sys.argv[2]); cmd=sys.argv[3:]
start=time.perf_counter()
with (out/'stdout.log').open('w',encoding='utf8') as stdout, (out/'stderr.log').open('w',encoding='utf8') as stderr:
    child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr)
    expired=False
    try: code=child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        expired=True; subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=stdout,stderr=stderr); code=child.wait()
receipt=dict(command=cmd,timeout_s=timeout,exit_code=code,timed_out=expired,process_wall_s=time.perf_counter()-start)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt)); sys.exit(code or int(expired))
