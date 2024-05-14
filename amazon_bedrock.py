import json
import os
from typing import Any, List, Mapping, Optional, Type
from cat.mad_hatter.decorators import tool, hook, plugin
from langchain_aws import ChatBedrock
from cat.factory.llm import LLMSettings
from pydantic import BaseModel, ConfigDict

from cat.mad_hatter.mad_hatter import MadHatter


class AmazonBedrockSettings(BaseModel):
    guardrails_trace: bool = True
    guardrails_id: str
    guardrails_version: str


@plugin
def settings_model():
    return AmazonBedrockSettings


class CustomBedrock(ChatBedrock):
    def __init__(self, **kwargs: Any) -> None:
        print(kwargs)
        guardrails = {}
        with open(os.path.join(plugin_path, "settings.json")) as f:
            settings = json.load(f)
            if bool(settings):
                guardrails = {
                    "id": settings["guardrails_id"],
                    "version": settings["guardrails_version"],
                    "trace": settings["guardrails_trace"],
                }

        kwargs["model_kwargs"] = {"temperature": kwargs["temperature"]}
        input_kwargs = {
            "region_name": kwargs["region_name"],
            "model_id": kwargs["model_id"],
            "endpoint_url": kwargs["endpoint_url"],
            "credentials_profile_name": kwargs["credentials_profile_name"],
            "streaming": True,
            "guardrails": guardrails,
        }
        super().__init__(**input_kwargs)


class AmazonBedrockConfig(LLMSettings):
    region_name: str
    model_id: str
    endpoint_url: str
    credentials_profile_name: str
    temperature: float = 0.7
    streaming: bool = True

    _pyclass: Type = CustomBedrock

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Amazon Bedrock",
            "description": "Configuration for Amazon Bedrock",
            "link": "https://aws.amazon.com/bedrock/",
        }
    )


@hook
def factory_allowed_llms(allowed, cat) -> List:
    global plugin_path
    plugin_path = MadHatter().get_plugin().path
    allowed.append(AmazonBedrockConfig)
    return allowed
