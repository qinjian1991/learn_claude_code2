from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from threading import Thread
from typing import Any

from langchain_core.tools import StructuredTool

from agent.tools.permission_registry import register_permission_rules
from core.config import settings
from core.constants import WORKDIR


logger = logging.getLogger("agent")
MCP_TOOL_NAMES: set[str] = set()


def load_mcp_config(path: str | Path | None = None) -> dict[str, Any] | None:
    config_path = Path(path or settings.mcp_config_path)
    if not config_path.is_absolute():
        config_path = WORKDIR / config_path

    if not config_path.exists():
        logger.info("MCP config not found: %s", config_path)
        return None

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Invalid MCP config %s: %s", config_path, exc)
        return None

    if not isinstance(config, dict):
        logger.warning("MCP config %s must contain a JSON object.", config_path)
        return None

    servers = config.get("mcpServers")
    if not isinstance(servers, dict) or not servers:
        return None

    return config


def discover_mcp_tools(
    config: dict[str, Any] | None = None,
    *,
    client_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> list[StructuredTool]:
    config = config if config is not None else load_mcp_config()
    if not config:
        return []

    try:
        client_factory = client_factory or _fastmcp_client_factory()
        tools = _run_async(_list_mcp_tools(client_factory(config)))
    except Exception as exc:
        logger.warning("MCP tool discovery failed: %s", exc)
        return []

    wrapped_tools = [
        _to_langchain_tool(tool, config, client_factory)
        for tool in tools
    ]
    _register_mcp_permission_rules([tool.name for tool in wrapped_tools])
    logger.info("Discovered %s MCP tool(s)", len(wrapped_tools))
    return wrapped_tools


def format_mcp_result(result: Any) -> str:
    if bool(getattr(result, "is_error", False)):
        return f"MCP tool error: {_content_text(result) or result}"

    data = getattr(result, "data", None)
    if data is not None:
        return _json_or_string(data)

    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return _json_or_string(structured)

    content_text = _content_text(result)
    if content_text:
        return content_text

    return _json_or_string(result)


async def _list_mcp_tools(client: Any) -> list[Any]:
    async with client:
        return list(
            await asyncio.wait_for(
                client.list_tools(),
                timeout=settings.mcp_discovery_timeout_seconds,
            )
        )


async def _call_mcp_tool(
    tool_name: str,
    args: dict[str, Any],
    config: dict[str, Any],
    client_factory: Callable[[dict[str, Any]], Any],
) -> str:
    client = client_factory(config)
    async with client:
        result = await client.call_tool(
            tool_name,
            args,
            timeout=settings.mcp_tool_timeout_seconds,
            raise_on_error=False,
        )
    return format_mcp_result(result)


def _to_langchain_tool(
    mcp_tool: Any,
    config: dict[str, Any],
    client_factory: Callable[[dict[str, Any]], Any],
) -> StructuredTool:
    tool_name = _field(mcp_tool, "name")
    description = _field(mcp_tool, "description") or f"MCP tool: {tool_name}"
    input_schema = _field(mcp_tool, "inputSchema") or {
        "type": "object",
        "properties": {},
    }

    def call_tool(**kwargs: Any) -> str:
        return _run_async(_call_mcp_tool(tool_name, kwargs, config, client_factory))

    call_tool.__name__ = f"call_mcp_{tool_name}"
    call_tool.__doc__ = description

    return StructuredTool.from_function(
        func=call_tool,
        name=tool_name,
        description=description,
        args_schema=input_schema,
        infer_schema=False,
    )


def _fastmcp_client_factory() -> Callable[[dict[str, Any]], Any]:
    try:
        from fastmcp import Client
    except ImportError as exc:
        raise RuntimeError(
            "fastmcp is not installed. Install requirements before enabling MCP."
        ) from exc

    return Client


def _register_mcp_permission_rules(tool_names: list[str]) -> None:
    new_tool_names = [name for name in tool_names if name not in MCP_TOOL_NAMES]
    if not new_tool_names:
        return

    MCP_TOOL_NAMES.update(new_tool_names)
    register_permission_rules([
        {
            "tools": [tool_name],
            "action": "approval_required",
            "check": lambda args: True,
            "message": "MCP tool call requires approval.",
            "risk_summary": _mcp_risk_summary(tool_name),
        }
        for tool_name in new_tool_names
    ])


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _content_text(result: Any) -> str:
    content = getattr(result, "content", None) or []
    text_parts = [
        str(block.text)
        for block in content
        if getattr(block, "text", None) is not None
    ]
    return "\n".join(text_parts)


def _json_or_string(value: Any) -> str:
    if isinstance(value, str):
        return value

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _mcp_risk_summary(tool_name: str) -> Callable[[dict[str, Any]], str]:
    return lambda args: f"{tool_name}: {_json_or_string(args)}"


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:
            result["error"] = exc

    thread = Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result.get("value")
