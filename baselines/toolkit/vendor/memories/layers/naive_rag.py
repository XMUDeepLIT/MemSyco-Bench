from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional, Literal, Union

from pydantic import BaseModel, Field, model_validator

from .base import BaseMemoryLayer

from mem0.memory.main import Memory  # type: ignore

from collections.abc import Mapping
from types import MappingProxyType


logger = logging.getLogger(__name__)


class NaiveRAGConfig(BaseModel):
    """Default configuration for NaiveRAG (aligned with common
    `memory_construction` / `memory_search` parameter conventions)."""

    # ===== Fields overridden/injected by general scripts =====
    user_id: str = Field(..., description="The user id of the memory system.")

    # General scripts set: config['save_dir'] = f"{layer_type}/{user_id}"
    save_dir: str = Field(
        default="vector_store/naive_rag",
        description="The directory to persist vector store and config.",
    )

    # ===== Field names aligned with memzero (to share config.json) =====
    retriever_name_or_path: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name/path (HF or OpenAI embedding model).",
    )

    embedding_model_dims: int = Field(
        default=384,
        description="Embedding dimension.",
    )

    use_gpu: str = Field(
        default="cpu",
        description="Device for embedding model, e.g. 'cpu' or 'cuda'.",
    )

    llm_backend: Literal["openai", "ollama"] = Field(
        default="openai",
        description="LLM backend provider (kept for consistency).",
    )

    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name (kept for consistency).",
    )

    # ===== Vector store / embedding provider =====
    vector_store_provider: Literal["qdrant", "chroma"] = Field(
        default="qdrant",
        description="Vector store provider for mem0.",
    )

    # If not provided, defaults to user_id
    collection_name: Optional[str] = Field(
        default=None,
        description="Vector store collection name; defaults to user_id.",
    )

    embedder_provider: Literal["huggingface", "openai"] = Field(
        default="huggingface",
        description="Embedder provider.",
    )

    embedding_api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI-compatible embedding providers.",
    )

    embedding_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for OpenAI-compatible embedding providers.",
    )

    # Key: whether Qdrant enables on-disk persistence
    # mem0 docs: Qdrant has an on_disk option, default may be False
    qdrant_on_disk: bool = Field(
        default=True,
        description="Enable Qdrant persistent storage (on_disk).",
    )

    @model_validator(mode="after")
    def _validate_and_fill(self) -> "NaiveRAGConfig":
        if os.path.isfile(self.save_dir):
            raise AssertionError(
                f"Provided path ({self.save_dir}) should be a directory, not a file"
            )
        if not self.collection_name:
            self.collection_name = self.user_id
        return self


class NaiveRAGLayer(BaseMemoryLayer):
    """Naive RAG: one dialogue turn -> one chunk -> one vector (no LLM memory extraction)."""

    layer_type: str = "NaiveRAG"

    def __init__(self, config: NaiveRAGConfig) -> None:
        self.config = config
        self.memory_config = self._build_memory_config()
        self._next_turn_index = 0

        try:
            self.memory_layer = Memory.from_config(self.memory_config)  # type: ignore
            logger.info(
                f"NaiveRAGLayer initialized for user={self.config.user_id}, "
                f"vs_provider={self.config.vector_store_provider}, save_dir={self.config.save_dir}, "
                f"collection={self.config.collection_name}, on_disk={self.config.qdrant_on_disk}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize NaiveRAG(mem0): {e}")
            raise RuntimeError(f"Failed to initialize NaiveRAG(mem0): {e}") from e

    def _build_memory_config(self) -> Dict[str, Any]:
        """Build mem0 configuration dict."""
        # embedder
        if self.config.embedder_provider == "huggingface":
            embedder_cfg: Dict[str, Any] = {
                "provider": "huggingface",
                "config": {
                    "model": self.config.retriever_name_or_path,
                    "embedding_dims": self.config.embedding_model_dims,
                    "model_kwargs": {"device": self.config.use_gpu},
                },
            }
        elif self.config.embedder_provider == "openai":
            embedder_cfg = {
                "provider": "openai",
                "config": {
                    "model": self.config.retriever_name_or_path,
                    "embedding_dims": self.config.embedding_model_dims,
                    "api_key": self.config.embedding_api_key or os.environ.get("OPENAI_API_KEY"),
                    "openai_base_url": (
                        self.config.embedding_base_url
                        or os.environ.get("OPENAI_API_BASE")
                        or os.environ.get("OPENAI_BASE_URL")
                    ),
                },
            }
        else:
            raise ValueError(f"Unsupported embedder_provider: {self.config.embedder_provider}")

        vector_store_cfg: Dict[str, Any] = {
            "collection_name": self.config.collection_name,
            "embedding_model_dims": self.config.embedding_model_dims,
            "path": self.config.save_dir,
        }

        # Key: enable Qdrant on_disk; otherwise "file created but points=0" occurs often
        if self.config.vector_store_provider == "qdrant":
            vector_store_cfg["on_disk"] = self.config.qdrant_on_disk

        return {
            "llm": {
                "provider": self.config.llm_backend,
                "config": {
                    "model": self.config.llm_model,
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "openai_base_url": os.environ.get("OPENAI_API_BASE"),
                },
            },
            "vector_store": {
                "provider": self.config.vector_store_provider,
                "config": vector_store_cfg,
            },
            "embedder": embedder_cfg,
        }

    # ==================== Persistence ====================

    def _save_config(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        config_path = os.path.join(self.config.save_dir, "config.json")

        # Only save reproducible/loadable fields (exclude api_key)
        config_dict = {
            "layer_type": self.layer_type,
            "user_id": self.config.user_id,
            "save_dir": self.config.save_dir,
            "retriever_name_or_path": self.config.retriever_name_or_path,
            "embedding_model_dims": self.config.embedding_model_dims,
            "use_gpu": self.config.use_gpu,
            "llm_backend": self.config.llm_backend,
            "llm_model": self.config.llm_model,
            "vector_store_provider": self.config.vector_store_provider,
            "collection_name": self.config.collection_name,
            "embedder_provider": self.config.embedder_provider,
            "embedding_api_key": self.config.embedding_api_key,
            "embedding_base_url": self.config.embedding_base_url,
            "qdrant_on_disk": self.config.qdrant_on_disk,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)

    def save_memory(self) -> None:
        """
        In most cases, Qdrant/Chroma persist data on write.
        We still write `config.json` to ensure search phase can rebuild.
        Also best-effort trigger `persist`/`close` where available.
        """
        try:
            os.makedirs(self.config.save_dir, exist_ok=True)
            self._save_config()

            # Best-effort: trigger underlying persistence (implementation differs by version)
            vs = getattr(self.memory_layer, "vector_store", None)
            if vs is not None:
                if hasattr(vs, "persist"):
                    try:
                        vs.persist()
                    except Exception:
                        pass
                client = getattr(vs, "client", None)
                if client is not None and hasattr(client, "close"):
                    try:
                        client.close()
                    except Exception:
                        pass

            logger.info(
                f"NaiveRAG saved config for user {self.config.user_id} at {self.config.save_dir}"
            )
        except Exception as e:
            logger.error(f"Error saving NaiveRAG config for user {self.config.user_id}: {e}")
            raise RuntimeError(f"Error saving NaiveRAG config for user {self.config.user_id}: {e}") from e

    def load_memory(self, user_id: Optional[str] = None) -> bool:
        """
        General scripts depend on this return value:
        - memory_construction: if not rerun and load_memory=True, skip rebuild
        - memory_search: when strict=True, return False will raise an error
        """
        if user_id is None:
            user_id = self.config.user_id

        # Prefer rebuilding from config.json (if present)
        config_path = os.path.join(self.config.save_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

                # Honor the provided user_id (general scripts call this way)
                cfg["user_id"] = user_id
                if not cfg.get("collection_name"):
                    cfg["collection_name"] = user_id

                self.config = NaiveRAGConfig(**cfg)
                self.memory_config = self._build_memory_config()
                self.memory_layer = Memory.from_config(self.memory_config)  # type: ignore
            except Exception as e:
                logger.warning(f"[NaiveRAG] Failed to rebuild from config.json: {e}")

        # Finally: use get_all to check if any data exists
        has_any = self._has_any_memory(user_id=user_id)
        logger.info(f"[NaiveRAG] load_memory(user_id={user_id}) -> {'FOUND' if has_any else 'EMPTY'}")
        return has_any

    def _has_any_memory(self, user_id: str) -> bool:
        """
        Handle compatibility for different mem0 versions' `get_all` return structures.
        """
        try:
            existing = self.memory_layer.get_all(user_id=user_id, limit=1)  # type: ignore
        except TypeError:
            # Some versions switched to filters
            existing = self.memory_layer.get_all(filters={"AND": [{"user_id": user_id}]}, limit=1)  # type: ignore
        except Exception as e:
            logger.warning(f"[NaiveRAG] get_all failed for user {user_id}: {e}")
            return False

        if isinstance(existing, dict):
            results = existing.get("results") or existing.get("memories") or existing.get("data") or []
            return bool(results)
        if isinstance(existing, list):
            return len(existing) > 0
        return False

    def _to_jsonable(self, obj: Any) -> Any:
        """
        Convert any Python object into a type acceptable by `json.dump`.
        - Scalars/None are returned as-is
        - list/tuple/set are processed recursively
        - dict / Mapping / mappingproxy are processed recursively
        - Other complex types are converted to `str(obj)`
        """
        # Simple scalars / None: return directly
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # list / tuple / set: recurse
        if isinstance(obj, (list, tuple, set)):
            return [self._to_jsonable(i) for i in obj]

        # dict / Mapping / mappingproxy: recurse key/values
        if isinstance(obj, (dict, Mapping, MappingProxyType)):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}

        # Convert others to string to ensure json.dump does not fail
        return str(obj)

    # ==================== Write ====================

    @staticmethod
    def _format_turn_chunk(message: Dict[str, str]) -> str:
        role = str(message.get("role") or "user").strip().lower()
        if role == "assistant":
            label = "Assistant"
        elif role == "system":
            label = "System"
        else:
            label = "User"
        content = str(message.get("content") or "").strip()
        return f"{label}: {content}"

    def add_message(self, message: Dict[str, str], **kwargs) -> None:
        """Index one dialogue turn as a single RAG chunk."""
        content = message.get("content")
        if content is None:
            raise KeyError("message must contain 'content'")

        chunk_text = self._format_turn_chunk(message)
        if not chunk_text.split(":", 1)[-1].strip():
            return

        raw_role = str(message.get("role") or "user").strip().lower()
        turn_index = kwargs.get("turn_index", self._next_turn_index)
        self._next_turn_index = int(turn_index) + 1

        metadata: Dict[str, Any] = {
            "raw_role": raw_role,
            "chunk_type": "turn",
            "turn_index": turn_index,
        }
        if "timestamp" in kwargs and kwargs["timestamp"] is not None:
            metadata["timestamp"] = kwargs["timestamp"]
        name = message.get("name")
        if name is not None:
            metadata["name"] = name

        self.memory_layer.add(
            messages=[{"role": "user", "content": chunk_text}],
            user_id=self.config.user_id,
            infer=False,
            metadata=metadata or None,
        )

    def add_messages(self, messages: List[Dict[str, str]], **kwargs) -> None:
        for m in messages:
            self.add_message(m, **kwargs)

    # ==================== Retrieval ====================

    def retrieve(
        self, query: str, k: int = 10, **kwargs
    ) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        res = self.memory_layer.search(
            query=query,
            user_id=self.config.user_id,
            limit=k,
        )

        if isinstance(res, dict):
            results = res.get("results") or res.get("memories") or res.get("data") or []
        elif isinstance(res, list):
            results = res
        else:
            results = []

        outputs: List[Dict[str, Union[str, Dict[str, Any]]]] = []
        for item in results:
            content = item.get("memory", "")
            metadata = {kk: vv for kk, vv in item.items() if kk != "memory"}
            out: Dict[str, Union[str, Dict[str, Any]]] = {
                "content": content,
                "metadata": metadata,
            }
            used_content = {
                "Turn": metadata.get("turn_index"),
                "Role": metadata.get("raw_role"),
                "Time": item.get("timestamp") or metadata.get("timestamp"),
                "Content": content,
            }
            out["used_content"] = "\n".join(
                f"{kk}: {vv}" for kk, vv in used_content.items() if vv is not None
            )

            outputs.append(out)

        return outputs


    # ==================== Other interfaces ====================

    def delete(self, memory_id: str) -> bool:
        try:
            self.memory_layer.delete(memory_id)  # type: ignore
            return True
        except Exception as e:
            logger.error(f"[NaiveRAG] delete error: {e}")
            return False

    def delete_all(self) -> bool:
        try:
            self.memory_layer.delete_all(user_id=self.config.user_id)  # type: ignore
            return True
        except Exception as e:
            logger.error(f"[NaiveRAG] delete_all error: {e}")
            return False

    def update(self, memory_id: str, **kwargs) -> bool:
        data = kwargs.get("data") or kwargs.get("content")
        if data is None:
            logger.error("[NaiveRAG] update requires 'data' or 'content'")
            return False
        try:
            self.memory_layer.update(memory_id, data)  # type: ignore
            return True
        except Exception as e:
            logger.error(f"[NaiveRAG] update error: {e}")
            return False
