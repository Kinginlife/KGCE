
from typing import Any, Literal, Optional, List

from pydantic import BaseModel

from kgce.core.backend_model import BackendModel
from .gemini_model import GeminiModel
from .openai_model import OpenAIModel, OpenAIModelJSON, SGlangOpenAIModelJSON
from .qwenvl_model import QwenVLModel, QwenVLModelJSON, SGlangQwenVLModelJSON  # 导入 QwenVLModel 相关模型。
class BackendModelConfig(BaseModel):
    model_class: Literal["openai", "gemini","qwenvl","deepseek"]
    """Specify the model class to be used. Different model classese use different
    APIs.
    """

    model_name: str | None = None
    """Specify the model name to be used. This value is directly passed to the API, 
    check model provider API documentation for more details.
    """

    model_platform: str | None = None
    """Required for CamelModel. Otherwise, it is ignored. Please check CAMEL
    documentation for more details.
    """

    history_messages_len: int = 0
    """Number of rounds of previous messages to be used in the model input. 0 means no
    history.
    """

    parameters: dict[str, Any] = {}
    """Additional parameters to be passed to the model."""

    json_structre_output: bool = False
    """If True, the model generate action through JSON without using "tool call" or
    "function call". SGLang model only supports JSON output. OpenAI model supports both.
    Other models do not support JSON output.
    """

    tool_call_required: bool = True
    """Specify if the model enforce each round to generate tool/function calls."""

    base_url: str | None = None
    """Specify the base URL of the API. Only used in OpenAI and SGLang currently."""

    api_key: str | None = None
    """Specify the API key to be used. Only used in OpenAI and SGLang currently."""

    app_id: str | None = None

    api_endpoint: str | None=None


def create_backend_model(model_config: BackendModelConfig) -> BackendModel:
    if model_config.model_class == "OpenAIModel":
        model_config.model_class = "openai"
    match model_config.model_class:
        case "gemini":
            if model_config.base_url is not None or model_config.api_key is not None:
                raise Warning(
                    "base_url and api_key are not supported for GeminiModel currently."
                )
            if model_config.json_structre_output:
                raise Warning(
                    "json_structre_output is not supported for GeminiModel currently."
                )
            return GeminiModel(
                model=model_config.model_name,
                parameters=model_config.parameters,
                history_messages_len=model_config.history_messages_len,
                api_key=model_config.api_key,
                api_endpoint=model_config.api_endpoint,
                tool_call_required=model_config.tool_call_required,
            )
        case "openai":
            if not model_config.json_structre_output:
                return OpenAIModel(
                    model=model_config.model_name,
                    parameters=model_config.parameters,
                    history_messages_len=model_config.history_messages_len,
                    base_url=model_config.base_url,
                    api_key=model_config.api_key,
                    tool_call_required=model_config.tool_call_required,
                )
            else:
                return OpenAIModelJSON(
                    model=model_config.model_name,
                    parameters=model_config.parameters,
                    history_messages_len=model_config.history_messages_len,
                    base_url=model_config.base_url,
                    api_key=model_config.api_key,
                )
        case "qwenvl":
            return QwenVLModel(
                model=model_config.model_name,
                api_key=model_config.api_key,
                base_url=model_config.base_url,
                parameters=model_config.parameters,
                history_messages_len=model_config.history_messages_len,
                tool_call_required=model_config.tool_call_required,
            )

        case _:
            raise ValueError(f"Unsupported model name: {model_config.model_name}")
