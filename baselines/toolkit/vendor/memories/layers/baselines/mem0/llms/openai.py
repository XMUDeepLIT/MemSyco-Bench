import json
import logging
import os
import time
from typing import Dict, List, Optional, Union

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.openai import OpenAIConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json


token_logger = logging.getLogger("mem0.token")


def _token_logging_enabled() -> bool:
    enabled = os.getenv("MEM0_TOKEN_LOG", "").lower() in {"1", "true", "yes", "on"}
    if enabled:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        token_logger.setLevel(logging.INFO)
    return enabled


def _message_char_count(messages: List[Dict[str, str]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)


class OpenAILLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, OpenAIConfig, Dict]] = None):
        # Convert to OpenAIConfig if needed
        if config is None:
            config = OpenAIConfig()
        elif isinstance(config, dict):
            config = OpenAIConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, OpenAIConfig):
            # Convert BaseLlmConfig to OpenAIConfig
            config = OpenAIConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-4o-mini"

        if os.environ.get("OPENROUTER_API_KEY"):  # Use OpenRouter
            self.client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url
                or os.getenv("OPENROUTER_API_BASE")
                or "https://openrouter.ai/api/v1",
            )
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = self.config.openai_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a JSON response based on the given messages using OpenAI.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional OpenAI-specific parameters.

        Returns:
            json: The generated response.
        """
        params = self._get_supported_params(messages=messages, **kwargs)
        
        params.update({
            "model": self.config.model,
            "messages": messages,
        })

        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_params = {}
            if self.config.models:
                openrouter_params["models"] = self.config.models
                openrouter_params["route"] = self.config.route
                params.pop("model")

            if self.config.site_url and self.config.app_name:
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
                openrouter_params["extra_headers"] = extra_headers

            params.update(**openrouter_params)
        
        else:
            openai_specific_generation_params = ["store"]
            for param in openai_specific_generation_params:
                if hasattr(self.config, param):
                    params[param] = getattr(self.config, param)
            
        if response_format:
            params["response_format"] = response_format
        if response_format and "max_tokens" in params:
            params["max_tokens"] = max(int(params["max_tokens"] or 0), int(os.environ.get("MEMZERO_LLM_MAX_TOKENS", "4096")))
        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        extra_body = dict(params.get("extra_body") or {})
        extra_body.setdefault("thinking", {"type": "disabled"})
        params["extra_body"] = extra_body
        started_at = time.perf_counter()
        response = self.client.chat.completions.create(**params)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        parsed_response = self._parse_response(response, tools)
        if _token_logging_enabled():
            usage = getattr(response, "usage", None)
            token_logger.info(
                "[mem0.llm] model=%s response_format=%s tools=%s messages=%d input_chars=%d "
                "prompt_tokens=%s completion_tokens=%s total_tokens=%s elapsed_ms=%.1f",
                params.get("model") or params.get("models"),
                bool(response_format),
                bool(tools),
                len(messages),
                _message_char_count(messages),
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
                getattr(usage, "total_tokens", None),
                elapsed_ms,
            )
        if self.config.response_callback:
            try:
                self.config.response_callback(self, response, params)
            except Exception as e:
                # Log error but don't propagate
                logging.error(f"Error due to callback: {e}")
                pass
        return parsed_response
