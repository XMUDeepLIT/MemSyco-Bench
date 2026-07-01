from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .base import BaselineEvalConfig, REPO_ROOT


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def get_baseline_config_path(method: str) -> Path:
    return CONFIG_DIR / f"{method}.json"


def build_baseline_eval_config(
    method: str,
    *,
    baseline_config_path: Path | None = None,
    top_k: int | None = None,
    config_path: Path | None = None,
    save_root: Path | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    llm_model: str | None = None,
) -> BaselineEvalConfig:
    data = _load_config(method, baseline_config_path)

    resolved_config_path = _resolve_optional_path(config_path or _optional_path(data.get("config_path")))
    resolved_save_root = _resolve_path(save_root or _optional_path(data.get("save_root")))

    return BaselineEvalConfig(
        method=method,
        top_k=int(top_k if top_k is not None else data.get("top_k", 10)),
        config_path=resolved_config_path,
        save_root=resolved_save_root or (REPO_ROOT / "output_data" / "baseline_eval_memory"),
        api_key=_first_nonempty(api_key, _env_value(data.get("api_key_env")), data.get("api_key")),
        base_url=_first_nonempty(base_url, _env_value(data.get("base_url_env")), data.get("base_url")),
        llm_model=_first_nonempty(llm_model, _env_value(data.get("llm_model_env")), data.get("llm_model")),
        embedding_model=_first_nonempty(
            _env_value(data.get("embedding_model_env")),
            data.get("embedding_model"),
        ),
        embedding_dims=_optional_int(
            _first_nonempty(
                _env_value(data.get("embedding_dims_env")),
                data.get("embedding_dims"),
            )
        ),
        embedding_api_key=_first_nonempty(
            _env_value(data.get("embedding_api_key_env")),
            data.get("embedding_api_key"),
            api_key,
            _env_value(data.get("api_key_env")),
            data.get("api_key"),
        ),
        embedding_base_url=_first_nonempty(
            _env_value(data.get("embedding_base_url_env")),
            data.get("embedding_base_url"),
            base_url,
            _env_value(data.get("base_url_env")),
            data.get("base_url"),
        ),
    )


def _load_config(method: str, baseline_config_path: Path | None) -> dict[str, Any]:
    path = baseline_config_path or get_baseline_config_path(method)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline config not found for {method}: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Baseline config must be a JSON object: {path}")
    config_method = data.get("method")
    if config_method and config_method != method:
        raise ValueError(f"Baseline config method mismatch: expected {method!r}, got {config_method!r}")
    return data


def _optional_path(value: Any) -> Path | None:
    if not value:
        return None
    return Path(str(value))


def _resolve_optional_path(path: Path | None) -> Path | None:
    resolved = _resolve_path(path)
    return resolved


def _resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _env_value(name: Any) -> str | None:
    if not name:
        return None
    return os.environ.get(str(name)) or None


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
