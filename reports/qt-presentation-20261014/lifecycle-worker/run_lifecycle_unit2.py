"""Lifecycle unit/regression runs with unrounded wall and cumulative budget.

Budget: 30 s total worker check budget. The old rounded receipts (2.154,
1.955, 2.051) plus 0.0005 s per-run accounting allowance are consumed; this
harness spends only the remainder, with each process timeout bounded by what
is left (and by 15 s). Walls are exact perf_counter floats; nothing is
rounded and no old run is repeated to reconstruct precision. On timeout the
whole process tree is killed and partial output is preserved in the receipt.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOTAL_BUDGET = 30.0
OLD_SPENT_UPPER = 2.154 + 1.955 + 2.051 + 3 * 0.0005  # 6.1615 (rounded receipts)
RUN = sys.argv[1] if len(sys.argv) > 1 else "run1"
TESTS = sys.argv[2:] or ["tests/test_qt_resource_lifecycle.py"]
CMD = [sys.executable, "-m", "pytest", *TESTS, "-p", "no:cacheprovider", "-q"]
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
SOURCES = ["src/any3dview/qt.py", "tests/test_qt_resource_lifecycle.py"]

new_spent = 0.0
for prior_receipt in sorted(HERE.glob("lifecycle-unit2-*-receipt.json")):
    try:
        new_spent += float(json.loads(
            prior_receipt.read_text(encoding="utf-8"))["wall_seconds_raw"])
    except Exception:
        pass
remaining_before = TOTAL_BUDGET - OLD_SPENT_UPPER - new_spent
timeout_seconds = max(1.0, min(15.0, remaining_before))

start = time.perf_counter()
proc = subprocess.Popen(
    CMD, cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, creationflags=0x00000200)  # CREATE_NEW_PROCESS_GROUP
timed_out = False
try:
    stdout, stderr = proc.communicate(timeout=timeout_seconds)
    code = proc.returncode
except subprocess.TimeoutExpired:
    timed_out = True
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                   capture_output=True, text=True)
    stdout, stderr = proc.communicate()  # process tree is dead; drains pipes
    code = proc.returncode
wall = time.perf_counter() - start

receipt = {
    "command": " ".join(CMD),
    "cwd": str(ROOT),
    "python": sys.executable,
    "env": {"PYTHONDONTWRITEBYTECODE": "1"},
    "timeout_seconds": timeout_seconds,
    "timed_out": timed_out,
    "exit": code,
    "wall_seconds_raw": wall,
    "wall_seconds_repr": repr(wall),
    "budget": {
        "total": TOTAL_BUDGET,
        "old_spent_upper": OLD_SPENT_UPPER,
        "new_spent_before": new_spent,
        "remaining_before": remaining_before,
        "remaining_after": TOTAL_BUDGET - OLD_SPENT_UPPER - new_spent - wall,
    },
    "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in SOURCES},
    "stdout": stdout,
    "stderr": stderr,
}
(HERE / f"lifecycle-unit2-{RUN}-receipt.json").write_text(
    json.dumps(receipt, indent=1), encoding="utf-8")
print(stdout[-6000:])
if stderr.strip():
    print("STDERR:", stderr[-2000:])
print(f"exit={code} timed_out={timed_out} wall_raw={wall!r}s "
      f"remaining_after={receipt['budget']['remaining_after']!r}s")
