import argparse
from pipelines.experiment_pipeline import execute_experiment
from pipelines.io_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()

    conf = load_config(args.config)
    summary = execute_experiment(conf)
    print(f"Experiment completed: {summary['run_id']}")
    print(f"Outputs: {summary['output_dir']}")


if __name__ == "__main__":
    main()
