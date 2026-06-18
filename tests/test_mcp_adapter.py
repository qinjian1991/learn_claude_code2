import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agent.nodes import call_tools
from agent.tools.mcp_adapter import (
    discover_mcp_tools,
    format_mcp_result,
    load_mcp_config,
)
from agent.tools.permissions import evaluate_tool_permission, refresh_permission_rules


def tool_call(name, args=None, call_id="call_test"):
    return {
        "name": name,
        "args": args or {},
        "id": call_id,
        "type": "tool_call",
    }


class FakeContent:
    def __init__(self, text):
        self.text = text


class FakeResult:
    def __init__(self, data=None, content=None, is_error=False):
        self.data = data
        self.content = content or []
        self.is_error = is_error
        self.structured_content = None


class FakeMcpClient:
    list_tools_result = []
    call_tool_result = FakeResult(data={"ok": True})
    called = []

    def __init__(self, config):
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def list_tools(self):
        return self.list_tools_result

    async def call_tool(self, name, arguments, timeout=None, raise_on_error=True):
        self.called.append(
            {
                "name": name,
                "arguments": arguments,
                "timeout": timeout,
                "raise_on_error": raise_on_error,
            }
        )
        return self.call_tool_result


class FakeLangChainTool:
    def __init__(self):
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return "MCP EXECUTED"


class McpConfigTest(unittest.TestCase):
    def test_missing_config_returns_no_config(self):
        missing = Path(tempfile.gettempdir()) / "missing-mcp-config-for-test.json"
        self.assertIsNone(load_mcp_config(missing))

    def test_invalid_config_returns_no_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mcp.json"
            path.write_text("{invalid", encoding="utf-8")

            self.assertIsNone(load_mcp_config(path))


class McpToolAdapterTest(unittest.TestCase):
    def setUp(self):
        FakeMcpClient.called = []
        FakeMcpClient.call_tool_result = FakeResult(data={"ok": True})
        FakeMcpClient.list_tools_result = [
            SimpleNamespace(
                name="weather_get_forecast",
                description="Get a forecast.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            )
        ]

    def test_discovers_mcp_tool_with_schema_and_preserved_name(self):
        tools = discover_mcp_tools(
            {"mcpServers": {"weather": {"command": "python"}}},
            client_factory=FakeMcpClient,
        )

        self.assertEqual([tool.name for tool in tools], ["weather_get_forecast"])
        self.assertIn("city", tools[0].args)

    def test_mcp_tool_invokes_client_and_formats_structured_result(self):
        tools = discover_mcp_tools(
            {"mcpServers": {"weather": {"command": "python"}}},
            client_factory=FakeMcpClient,
        )

        result = tools[0].invoke({"city": "London"})

        self.assertEqual(result, '{"ok": true}')
        self.assertEqual(FakeMcpClient.called[0]["name"], "weather_get_forecast")
        self.assertEqual(FakeMcpClient.called[0]["arguments"], {"city": "London"})
        self.assertFalse(FakeMcpClient.called[0]["raise_on_error"])

    def test_format_result_falls_back_to_text_content(self):
        result = format_mcp_result(FakeResult(content=[FakeContent("plain text")]))

        self.assertEqual(result, "plain text")

    def test_format_result_surfaces_tool_errors(self):
        result = format_mcp_result(
            FakeResult(content=[FakeContent("failed")], is_error=True)
        )

        self.assertEqual(result, "MCP tool error: failed")

    def test_mcp_tools_require_approval_by_default(self):
        discover_mcp_tools(
            {"mcpServers": {"weather": {"command": "python"}}},
            client_factory=FakeMcpClient,
        )
        refresh_permission_rules()

        decision = evaluate_tool_permission(
            tool_call("weather_get_forecast", {"city": "London"})
        )

        self.assertEqual(decision["action"], "approval_required")
        self.assertEqual(decision["reason"], "MCP tool call requires approval.")
        self.assertIn("weather_get_forecast", decision["risk_summary"])
        self.assertIn("London", decision["risk_summary"])

    def test_approved_mcp_call_executes(self):
        fake_tool = FakeLangChainTool()
        message = AIMessage(
            content="",
            tool_calls=[
                tool_call("weather_get_forecast", {"city": "London"}, "call_mcp")
            ],
        )
        state = {
            "messages": [message],
            "tool_permission_decisions": [
                {
                    "tool_call_id": "call_mcp",
                    "tool_name": "weather_get_forecast",
                    "action": "allow",
                    "reason": "Approved by user.",
                    "risk_summary": "weather_get_forecast: London",
                }
            ],
        }

        with patch.dict(
            "agent.nodes.TOOLS_BY_NAME",
            {"weather_get_forecast": fake_tool},
        ):
            result = call_tools(state)

        self.assertEqual(fake_tool.calls, [{"city": "London"}])
        self.assertEqual(result["messages"][-1].content, "MCP EXECUTED")

    def test_rejected_mcp_call_does_not_execute(self):
        fake_tool = FakeLangChainTool()
        message = AIMessage(
            content="",
            tool_calls=[
                tool_call("weather_get_forecast", {"city": "London"}, "call_mcp")
            ],
        )
        state = {
            "messages": [message],
            "tool_permission_decisions": [
                {
                    "tool_call_id": "call_mcp",
                    "tool_name": "weather_get_forecast",
                    "action": "rejected",
                    "reason": "Rejected by user.",
                    "risk_summary": "weather_get_forecast: London",
                }
            ],
        }

        with patch.dict(
            "agent.nodes.TOOLS_BY_NAME",
            {"weather_get_forecast": fake_tool},
        ):
            result = call_tools(state)

        self.assertEqual(fake_tool.calls, [])
        self.assertIn("rejected", result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
