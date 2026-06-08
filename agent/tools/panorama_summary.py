from langchain_core.tools import tool


@tool
def request_panorama_summary(reason: str) -> str:
    """Request a full conversation summary when the context is getting noisy or long."""
    return "Panorama summary requested."
