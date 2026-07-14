from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TrainingRunConfig:
    enabled: bool
    script: str
    args: List[str]

    @classmethod
    def from_section(cls, section: Dict[str, Any]) -> "TrainingRunConfig":
        return cls(
            enabled=bool(section.get("enabled", False)),
            script=str(section.get("script", "")),
            args=[str(x) for x in section.get("args", [])],
        )


@dataclass
class InferenceRunConfig:
    enabled: bool
    script: str
    args: List[str]

    @classmethod
    def from_section(cls, section: Dict[str, Any]) -> "InferenceRunConfig":
        return cls(
            enabled=bool(section.get("enabled", False)),
            script=str(section.get("script", "")),
            args=[str(x) for x in section.get("args", [])],
        )


@dataclass
class ClassificationEvalConfig:
    enabled: bool
    model_name: str
    model_id: str
    test_data: str
    output_dir: str
    max_new_tokens: int

    @classmethod
    def from_section(cls, section: Dict[str, Any]) -> "ClassificationEvalConfig":
        return cls(
            enabled=bool(section.get("enabled", False)),
            model_name=str(section.get("model_name", "classification_model")),
            model_id=str(section.get("model_id", "models/google/medgemma-27b-text-it")),
            test_data=str(section.get("test_data", "data/raw/test_augmented_data.csv")),
            output_dir=str(section.get("output_dir", "data/reports/default/classification")),
            max_new_tokens=int(section.get("max_new_tokens", 256)),
        )


@dataclass
class JudgeConfig:
    enabled: bool
    model_id: str
    api_key: Optional[str]
    base_url: str
    max_retries: int
    retry_delay: float
    verbose: bool
    checkpoint_every: int
    output_dir: str

    @classmethod
    def from_section(cls, section: Dict[str, Any], default_output_dir: str) -> "JudgeConfig":
        return cls(
            enabled=bool(section.get("enabled", False)),
            model_id=str(section.get("model_id", "qwen-max")),
            api_key=section.get("api_key"),
            base_url=str(section.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            max_retries=int(section.get("max_retries", 4)),
            retry_delay=float(section.get("retry_delay", 2.0)),
            verbose=bool(section.get("verbose", True)),
            checkpoint_every=int(section.get("checkpoint_every", 50)),
            output_dir=str(section.get("output_dir", default_output_dir)),
        )


@dataclass
class ExplanationEvalConfig:
    enabled: bool
    model_name: str
    predictions_file: str
    labels_file: str
    label_predictions_file: str
    output_dir: str

    @classmethod
    def from_section(cls, section: Dict[str, Any]) -> "ExplanationEvalConfig":
        return cls(
            enabled=bool(section.get("enabled", False)),
            model_name=str(section.get("model_name", "explanation_model")),
            predictions_file=str(section.get("predictions_file", "")),
            labels_file=str(section.get("labels_file", "")),
            label_predictions_file=str(section.get("label_predictions_file", "")),
            output_dir=str(section.get("output_dir", "data/reports/default/explanation_eval")),
        )
