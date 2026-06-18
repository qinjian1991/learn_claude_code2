from agent.tools.edit_file import edit_file
from agent.tools.glob import glob_files
from agent.tools.load_skill import load_skill
from agent.tools.mcp_adapter import discover_mcp_tools
from agent.tools.panorama_summary import request_panorama_summary
from agent.tools.powershell import powershell
from agent.tools.read_file import read_file
from agent.tools.spawn_subagent import spawn_subagent
from agent.tools.todo_write import run_todo_write
from agent.tools.write_file import write_file


SPECIAL_PANORAMA_SUMMARY_TOOL = request_panorama_summary.name
LOCAL_TOOLS = [
    powershell,
    read_file,
    load_skill,
    write_file,
    edit_file,
    glob_files,
    spawn_subagent,
    request_panorama_summary,
    run_todo_write,
]
TOOLS = LOCAL_TOOLS + discover_mcp_tools()
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
