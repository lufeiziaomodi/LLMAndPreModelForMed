import argparse
import os

from evaluate.config import EvaluationConfig
from evaluate.cli.workflows import run_classify, run_faithfulness, run_full, run_judge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DDI evaluation workflows")
    parser.add_argument("--mode", choices=["classify", "faithfulness", "judge", "full"], default="full")
    parser.add_argument("--test-data", default="data/test_augmented_data.csv")
    parser.add_argument("--predictions-file", default=None)
    parser.add_argument("--output-dir", default="results/ddi_evaluation")
    parser.add_argument("--model-id", default="models/google/medgemma-27b-text-it")
    parser.add_argument("--max-new-tokens", type=int, default=256)

    parser.add_argument("--use-judge", action="store_true")
    parser.add_argument("--judge-model-id", default="qwen-max")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--judge-max-retries", type=int, default=4)
    parser.add_argument("--judge-retry-delay", type=float, default=2.0)
    parser.add_argument("--judge-verbose", action="store_true")
    parser.add_argument("--judge-checkpoint-every", type=int, default=50)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    api_key = args.judge_api_key or os.getenv("DASHSCOPE_API_KEY")

    cfg = EvaluationConfig(
        mode=args.mode,
        test_data_path=args.test_data,
        predictions_file=args.predictions_file,
        output_dir=args.output_dir,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        use_judge=args.use_judge,
        judge_model_id=args.judge_model_id,
        judge_api_key=api_key,
        judge_base_url=args.judge_base_url,
        judge_max_retries=args.judge_max_retries,
        judge_retry_delay=args.judge_retry_delay,
        judge_verbose=args.judge_verbose,
        judge_checkpoint_every=args.judge_checkpoint_every,
    )
    cfg.validate()

    if cfg.mode == "classify":
        result = run_classify(cfg)
        print(
            f"Classification done: Micro-F1={result['metrics']['micro_f1']:.4f}, "
            f"Macro-F1={result['metrics']['macro_f1']:.4f}, "
            f"Weighted-F1={result['metrics']['weighted_f1']:.4f}"
        )
        return

    if cfg.mode == "faithfulness":
        result = run_faithfulness(cfg)
        summary = result["summary"]
        print(
            f"Faithfulness done: coverage={summary['coverage_mean']:.4f}, "
            f"hallucination={summary['hallucination_mean']:.4f}, "
            f"consistency={summary['consistency_mean']:.4f}"
        )
        return

    if cfg.mode == "judge":
        if not cfg.use_judge:
            raise ValueError("--use-judge is required when mode=judge")
        result = run_judge(cfg)
        print(f"Judge done: mechanism_overall={result['summary']['mechanism_overall_score_mean']:.4f}")
        return

    result = run_full(cfg)
    print("Full workflow completed.")
    print(f"Tag: {result['tag']}")


if __name__ == "__main__":
    main()
