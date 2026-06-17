from agent.tools.edit_file import edit_file
from agent.tools.glob import glob_files
from agent.tools.panorama_summary import request_panorama_summary
from agent.tools.powershell import powershell
from agent.tools.read_file import read_file
from agent.tools.spawn_subagent import spawn_subagent
from agent.tools.write_file import write_file


SPECIAL_PANORAMA_SUMMARY_TOOL = request_panorama_summary.name
TOOLS = [
    powershell,
    read_file,
    write_file,
    edit_file,
    glob_files,
    spawn_subagent,
    request_panorama_summary,
]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
