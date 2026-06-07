from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    call_tools,
    exit_agent_loop,
    invoke_model,
    prepare_model_input,
    process_model_response,
    should_continue,
)
from agent.persistence import CHECKPOINTER, STORE
from agent.state import AgentState, RuntimeContext


def build_graph():
    builder = StateGraph(AgentState, context_schema=RuntimeContext)
    builder.add_node("prepare_model_input", prepare_model_input)
    builder.add_node("invoke_model", invoke_model)
    builder.add_node("process_model_response", process_model_response)
    builder.add_node("tools", call_tools)
    builder.add_node("exit_agent_loop", exit_agent_loop)
    builder.add_edge(START, "prepare_model_input")
    builder.add_edge("prepare_model_input", "invoke_model")
    builder.add_edge("invoke_model", "process_model_response")
    builder.add_conditional_edges(
        "process_model_response",
        should_continue,
        ["tools", "exit_agent_loop"],
    )
    builder.add_edge("tools", "prepare_model_input")
    builder.add_edge("exit_agent_loop", END)
    return builder.compile(
        checkpointer=CHECKPOINTER,
        store=STORE,
    )


GRAPH = build_graph()
