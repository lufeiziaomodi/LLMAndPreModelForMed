from typing import Any, Dict

from pipelines.contracts import TrainingRunConfig
from pipelines.service_runner import run_python_script


def run_training(section: Dict[str, Any]) -> Dict[str, Any]:
    conf = TrainingRunConfig.from_section(section)
    script = conf.script
    if not script:
        raise ValueError("training.script is required")

    return run_python_script(script=script, args=conf.args)
