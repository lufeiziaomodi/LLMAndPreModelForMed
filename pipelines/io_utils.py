import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def make_run_id(prefix: str = "exp") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return os.getenv(var_name, "")

        return _ENV_PATTERN.sub(_replace, value)
    return value


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Prefer YAML when available; fallback to JSON.
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return _expand_env_vars(data)
    except Exception:
        pass

    try:
        data = json.loads(text)
    except Exception as exc:
        raise ValueError(
            "Failed to parse config. Provide valid YAML (requires pyyaml) or JSON."
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("Config root must be an object")
    return _expand_env_vars(data)
