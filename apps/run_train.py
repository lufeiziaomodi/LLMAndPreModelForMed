import argparse
import os

from pipelines.io_utils import load_config, save_json
from pipelines.training_pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training pipeline from config")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()

    conf = load_config(args.config)
    exp = conf.get("experiment", {})
    output_dir = exp.get("output_dir", "results/experiments/default")
    os.makedirs(output_dir, exist_ok=True)

    train_conf = conf.get("training", {})
    if not train_conf.get("enabled", False):
        raise ValueError("training.enabled must be true")

    result = run_training(train_conf)
    save_json(os.path.join(output_dir, "train_results.json"), result)

    if not result.get("ok", False):
        raise SystemExit(result.get("return_code", 1))

    print(f"Saved training results to {output_dir}")


if __name__ == "__main__":
    main()
