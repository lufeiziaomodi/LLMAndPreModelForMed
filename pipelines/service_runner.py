import subprocess
from typing import Dict, List


def run_python_script(script: str, args: List[str]) -> Dict[str, object]:
    cmd = ["python", script] + [str(x) for x in args]
    completed = subprocess.run(cmd, check=False)
    return {
        "command": cmd,
        "return_code": int(completed.returncode),
        "ok": completed.returncode == 0,
    }
