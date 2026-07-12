from dataclasses import dataclass, field
from typing import List, Optional


DEFAULT_DDI_TYPES = ["effect", "mechanism", "advise", "int", "false"]


@dataclass
class EvaluationConfig:
    mode: str = "full"
    test_data_path: str = "data/test_augmented_data.csv"
    predictions_file: Optional[str] = None
    predictions_labels_file: Optional[str] = None
    label_predictions_file: Optional[str] = None
    output_dir: str = "results/ddi_evaluation"

    model_id: str = "models/google/medgemma-27b-text-it"
    max_new_tokens: int = 256

    ddi_types: List[str] = field(default_factory=lambda: list(DEFAULT_DDI_TYPES))

    compute_coverage: bool = True
    compute_hallucination: bool = True
    compute_consistency: bool = True

    use_judge: bool = False
    judge_model_id: str = "qwen-max"
    judge_api_key: Optional[str] = None
    judge_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    judge_batch_size: int = 5
    judge_max_retries: int = 4
    judge_retry_delay: float = 2.0
    judge_verbose: bool = True
    judge_checkpoint_every: int = 50

    temperature: float = 0.0
    top_p: float = 1.0

    def validate(self) -> None:
        valid_modes = {"classify", "faithfulness", "judge", "full"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.use_judge and not self.judge_api_key:
            raise ValueError("judge_api_key is required when use_judge=True")
