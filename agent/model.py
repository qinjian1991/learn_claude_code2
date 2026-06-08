from core.config import settings
from langchain_anthropic import ChatAnthropic

from agent.tools import TOOLS


BASE_MODEL = ChatAnthropic(
    base_url=settings.anthropic_base_url,
    api_key=settings.anthropic_api_key,
    model_name=settings.anthropic_model,
    max_tokens=settings.model_max_tokens,
    streaming=True,
    temperature=settings.model_temperature,
)
MODEL = BASE_MODEL.bind_tools(TOOLS)
