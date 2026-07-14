from typing import Any, Dict

from pipelines.contracts import InferenceRunConfig
from pipelines.service_runner import run_python_script


def run_inference(section: Dict[str, Any]) -> Dict[str, Any]:
    conf = InferenceRunConfig.from_section(section)
    script = conf.script
    if not script:
        raise ValueError("inference.script is required")

    result = run_python_script(script=script, args=conf.args)
    output_json = ""
    metrics_json = ""
    args = list(conf.args)
    for idx, value in enumerate(args):
        if value == "--output_json" and idx + 1 < len(args):
            output_json = str(args[idx + 1])
        if value == "--metrics_json" and idx + 1 < len(args):
            metrics_json = str(args[idx + 1])

    result["output_json"] = output_json
    result["metrics_json"] = metrics_json
    return result