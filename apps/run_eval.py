import argparse
import os

from pipelines.classification_pipeline import run_classification_eval
from pipelines.explanation_pipeline import run_explanation_eval
from pipelines.io_utils import load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation pipelines from config")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()

    conf = load_config(args.config)
    exp = conf.get("experiment", {})
    output_dir = exp.get("output_dir", "results/experiments/default")
    os.makedirs(output_dir, exist_ok=True)

    results = {"experiment": exp, "status": "ok"}

    if conf.get("classification_eval", {}).get("enabled", False):
        results["classification_eval"] = run_classification_eval(conf["classification_eval"])

    if conf.get("explanation_eval", {}).get("enabled", False):
        judge = conf.get("judge", {})
        results["explanation_eval"] = run_explanation_eval(conf["explanation_eval"], judge)

    save_json(os.path.join(output_dir, "eval_results.json"), results)
    print(f"Saved eval results to {output_dir}")


if __name__ == "__main__":
    main()
