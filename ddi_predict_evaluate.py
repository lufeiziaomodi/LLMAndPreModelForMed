"""Backward-compatible entrypoint for evaluation workflows.

Legacy command:
    python ddi_predict_evaluate.py

Recommended new command:
    python -m evaluate.cli.main --mode classify
"""

from evaluate.cli.main import main


if __name__ == "__main__":
    main()