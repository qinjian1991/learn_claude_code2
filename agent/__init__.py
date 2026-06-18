__all__ = ["Agent", "GRAPH", "build_graph"]


def __getattr__(name):
    if name == "Agent":
        from agent.client import Agent

        return Agent

    if name in {"GRAPH", "build_graph"}:
        from agent.graph import GRAPH, build_graph

        return {"GRAPH": GRAPH, "build_graph": build_graph}[name]

    raise AttributeError(f"module 'agent' has no attribute {name!r}")
