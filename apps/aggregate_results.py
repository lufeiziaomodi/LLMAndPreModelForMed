import argparse
import os

from pipelines.results_aggregator import collect_compare_rows, write_aggregate_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate compare_summary.csv files")
    parser.add_argument("--root", default="data/reports", help="Root experiments/report directory (default: data/reports)")
    parser.add_argument("--output", default="data/reports/aggregate_compare_summary.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = collect_compare_rows(args.root)
    out = write_aggregate_csv(args.output, rows)
    print(f"Found {len(rows)} rows")
    print(f"Saved aggregate summary to {out}")


if __name__ == "__main__":
    main()
