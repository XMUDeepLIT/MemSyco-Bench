from typing import Dict, Optional, Literal, Any
import os
import json
import logging
import time
from abc import ABC, abstractmethod
from litellm import completion


logger = logging.getLogger(__name__)


def _method_logging_enabled() -> bool:
    return os.getenv("AMEM_LOG", "").lower() in {"1", "true", "yes", "on"} or os.getenv(
        "MEMORY_METHOD_LOG", ""
    ).lower() in {"1", "true", "yes", "on"}


def _configure_method_logging() -> None:
    if not _method_logging_enabled():
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.setLevel(logging.INFO)

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            _configure_method_logging()
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            base_url = os.getenv('OPENAI_API_BASE')
            if base_url is None:
                self.client = OpenAI(api_key=api_key)
            else:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        max_tokens = int(os.getenv("AMEM_LLM_MAX_TOKENS", "4096"))
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if response_format:
            params["response_format"] = _portable_response_format(response_format)

        started_at = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**params)
        except Exception as exc:
            if "response_format" not in str(exc):
                raise
            params.pop("response_format", None)
            response = self.client.chat.completions.create(**params)
        usage = getattr(response, "usage", None)
        logger.info(
            "[amem.llm] model=%s response_format=%s prompt_chars=%d prompt_tokens=%s completion_tokens=%s total_tokens=%s elapsed_ms=%.1f",
            self.model,
            bool(response_format),
            len(prompt),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "total_tokens", None),
            (time.perf_counter() - started_at) * 1000,
        )
        return response.choices[0].message.content

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai", "ollama"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama'")
            
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)


def _portable_response_format(response_format: dict) -> dict:
    if response_format and response_format.get("type") == "json_schema":
        return {"type": "json_object"}
    return response_format
