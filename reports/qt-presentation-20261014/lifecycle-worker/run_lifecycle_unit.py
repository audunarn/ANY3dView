"""Focused lifecycle unit run with exact command/hash/exit/wall receipt."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = sys.argv[1] if len(sys.argv) > 1 else "run1"
TESTS = sys.argv[2:] or ["tests/test_qt_resource_lifecycle.py"]
CMD = [sys.executable, "-m", "pytest", *TESTS, "-p", "no:cacheprovider", "-q"]
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
SOURCES = ["src/any3dview/qt.py", "tests/test_qt_resource_lifecycle.py"]

start = time.perf_counter()
timed_out = False
try:
    proc = subprocess.run(CMD, cwd=ROOT, env=ENV, capture_output=True,
                          text=True, timeout=20)
    stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
except subprocess.TimeoutExpired as expired:
    timed_out = True
    stdout = expired.stdout or ""
    stderr = expired.stderr or ""
    code = None
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
wall = time.perf_counter() - start

receipt = {
    "command": " ".join(CMD),
    "cwd": str(ROOT),
    "python": sys.executable,
    "env": {"PYTHONDONTWRITEBYTECODE": "1"},
    "timeout_seconds": 20,
    "timed_out": timed_out,
    "exit": code,
    "wall_seconds": round(wall, 3),
    "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in SOURCES},
    "stdout": stdout,
    "stderr": stderr,
}
(HERE / f"lifecycle-unit-{RUN}-receipt.json").write_text(
    json.dumps(receipt, indent=1), encoding="utf-8")
print(stdout[-6000:])
if stderr.strip():
    print("STDERR:", stderr[-2000:])
print(f"exit={code} timed_out={timed_out} wall={wall:.3f}s")
