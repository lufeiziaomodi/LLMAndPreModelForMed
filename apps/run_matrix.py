import argparse

from pipelines.io_utils import load_config
from pipelines.matrix_pipeline import run_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matrix experiments from config")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON matrix config")
    args = parser.parse_args()

    conf = load_config(args.config)
    result = run_matrix(conf)
    print(f"Matrix completed. Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
