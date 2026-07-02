from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from ..base import BaselineContext, BaselineEvalConfig, REPO_ROOT
from ..common import (
    format_retrieved_memories,
    jsonable_memories,
    parse_dialogue_to_messages,
    retry_embedding_query,
    sample_user_id,
    timestamp_for_turn,
)

TOOLKIT_VENDOR_DIR = REPO_ROOT / "baselines" / "toolkit" / "vendor"
DEFAULT_TOOLKIT_CONFIG_DIR = TOOLKIT_VENDOR_DIR / "configs"
VENDORED_MEM0_PARENT_DIR = TOOLKIT_VENDOR_DIR / "memories" / "layers" / "baselines"
LIGHTMEM_SRC_DIR = REPO_ROOT / "baselines" / "lightmem" / "vendor" / "src"

TOOLKIT_METHODS = ("NaiveRAG", "A-MEM", "MemZero")


def build_lightmem_toolkit_context(
    method: str,
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    if method not in TOOLKIT_METHODS:
        raise ValueError(f"Unsupported toolkit baseline method: {method!r}")

    _prepare_import_path()
    config_dict = _load_method_config(method, eval_config)
    user_id = sample_user_id(method, sample_key, prior_dialogue, user_question)
    save_dir = str((eval_config.save_root / method / user_id).resolve())

    config_dict["user_id"] = user_id
    config_dict["save_dir"] = save_dir
    if "collection_name" in config_dict or method in {"NaiveRAG", "MemZero"}:
        config_dict["collection_name"] = user_id
    if eval_config.llm_model:
        config_dict["llm_model"] = _format_llm_model_for_method(method, eval_config.llm_model)
    if eval_config.embedding_model:
        config_dict["retriever_name_or_path"] = eval_config.embedding_model
    if eval_config.embedding_dims is not None:
        config_dict["embedding_model_dims"] = eval_config.embedding_dims
    _apply_api_overrides(method, config_dict, eval_config)

    from memories import CONFIG_MAPPING, MEMORY_LAYERS_MAPPING  # type: ignore

    config = CONFIG_MAPPING[method](**config_dict)
    layer = MEMORY_LAYERS_MAPPING[method](config)

    for idx, message in enumerate(parse_dialogue_to_messages(prior_dialogue)):
        add_kwargs: dict[str, Any] = {"timestamp": timestamp_for_turn(idx)}
        if method == "NaiveRAG":
            add_kwargs["turn_index"] = idx
        layer.add_message(message, **add_kwargs)

    retrieved = retry_embedding_query(
        lambda: layer.retrieve(user_question, k=eval_config.top_k),
        method=method,
    )
    return BaselineContext(
        context_text=format_retrieved_memories(retrieved),
        retrieved_memories=jsonable_memories(retrieved),
        user_id=user_id,
        save_dir=save_dir,
        method=method,
        top_k=eval_config.top_k,
    )


def _prepare_import_path() -> None:
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    if not TOOLKIT_VENDOR_DIR.is_dir():
        raise FileNotFoundError(f"Toolkit vendor directory not found: {TOOLKIT_VENDOR_DIR}")
    if not VENDORED_MEM0_PARENT_DIR.is_dir():
        raise FileNotFoundError(f"Vendored Mem0 directory not found: {VENDORED_MEM0_PARENT_DIR}")
    import_paths = (
        str(VENDORED_MEM0_PARENT_DIR),
        str(TOOLKIT_VENDOR_DIR),
        str(LIGHTMEM_SRC_DIR.resolve()),
    )
    for path in reversed(import_paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    _drop_nonvendored_mem0_modules()


def _drop_nonvendored_mem0_modules() -> None:
    mem0_module = sys.modules.get("mem0")
    if mem0_module is None:
        return
    module_file = getattr(mem0_module, "__file__", None)
    if module_file and Path(module_file).resolve().is_relative_to(VENDORED_MEM0_PARENT_DIR.resolve()):
        return
    for name in list(sys.modules):
        if name == "mem0" or name.startswith("mem0."):
            sys.modules.pop(name, None)


def _load_method_config(method: str, eval_config: BaselineEvalConfig) -> dict[str, Any]:
    config_path = eval_config.config_path
    if config_path is None:
        config_path = DEFAULT_TOOLKIT_CONFIG_DIR / f"{method}.json"
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Toolkit config must be a JSON object: {config_path}")
    return dict(data)


def _apply_api_overrides(method: str, config: dict[str, Any], eval_config: BaselineEvalConfig) -> None:
    api_key = (eval_config.api_key or "").strip()
    base_url = (eval_config.base_url or "").strip()
    embedding_api_key = (eval_config.embedding_api_key or api_key).strip()
    embedding_base_url = (eval_config.embedding_base_url or base_url).strip()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        config["api_key"] = api_key
    if base_url:
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
        config["base_url"] = base_url
    if embedding_api_key:
        config["embedding_api_key"] = embedding_api_key
    if embedding_base_url:
        config["embedding_base_url"] = embedding_base_url

    if method == "A-MEM":
        if not config.get("api_key") and api_key:
            config["api_key"] = api_key
        if not config.get("base_url") and base_url:
            config["base_url"] = base_url


def _format_llm_model_for_method(method: str, model: str) -> str:
    return model
