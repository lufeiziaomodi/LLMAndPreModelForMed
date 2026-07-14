# Configs

This folder stores reusable experiment configuration files.

## Suggested workflow

1. Copy `configs/experiments/template_dual_track.yaml` to a new file.
2. Edit model, data, and stage toggles.
3. Run unified entrypoint:

```bash
python apps/run_experiment.py --config configs/experiments/template_dual_track.yaml
```

4. Run matrix experiments:

```bash
python apps/run_matrix.py --config configs/experiments/matrix_dual_track.yaml
```

5. Aggregate all compare summaries across runs:

```bash
python apps/aggregate_results.py --root results/experiments --output results/experiments/aggregate_compare_summary.csv
```

## Stage toggles

- `training.enabled`: run training stage.
- `classification_eval.enabled`: run classification metrics stage.
- `explanation_eval.enabled`: run faithfulness metrics stage.
- `judge.enabled`: run qwen-max mechanism-oriented LLM-as-judge on explanation outputs.

## Mechanism-first evaluation (recommended)

If your goal is mechanism discovery (not only label agreement), keep this pattern:

- `explanation_eval.enabled: true` to produce deterministic grounding signals.
- `judge.enabled: true` to score mechanism-level quality with rubric dimensions.
- Judge no longer consumes faithfulness diagnostics; it only reads the raw explanation-side inputs.

Recommended interpretation order:

- First check grounding floor (`coverage_mean`, `hallucination_mean`, `query_coverage_mean`).
- Then compare mechanism-level judge dimensions.

For the full Chinese guideline and scoring framework, see:

- [docs/mechanism_evaluation_framework_zh.md](../docs/mechanism_evaluation_framework_zh.md)

## API key

Set `DASHSCOPE_API_KEY` in environment or `judge.api_key` in config.

## max_new_tokens Policy

To keep experiments comparable, use a unified token budget by task type:

- Classification (`label_only_*`): `128`
- Reasoning + label (`full`, `reasoning_without_kg`): `512`
- Explanation-only (`explanation_*`): `768`

In `inference_llama3_ddi.py`, pass `--max_new_tokens -1` to enable this automatic profile.
