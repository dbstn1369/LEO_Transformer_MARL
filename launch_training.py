"""
Launch run_all.py as a fully detached Windows process.
This process survives even if the parent terminal/session closes.

Usage: python launch_training.py
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS = fully independent process
# CREATE_NO_WINDOW = no console window popup
DETACHED = (
    subprocess.CREATE_NEW_PROCESS_GROUP |
    subprocess.DETACHED_PROCESS |
    subprocess.CREATE_NO_WINDOW
)

log_file = open("logs/pipeline.log", "w")
target = sys.argv[1] if len(sys.argv) > 1 else "run_all.py"
proc = subprocess.Popen(
    [sys.executable, "-u", target],
    stdout=log_file,
    stderr=subprocess.STDOUT,
    creationflags=DETACHED,
    close_fds=True,
)
print(f"Launched {target} as detached process PID={proc.pid}")
print(f"Log: logs/pipeline.log")
print(f"Check:   tasklist | findstr python")
