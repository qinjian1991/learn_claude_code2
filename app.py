import streamlit as st

from agent import Agent
from core.config import settings
from thread_memory import get_or_create_default_thread_id, rotate_default_thread_id


st.set_page_config(
    page_title="LangGraph Agent",
    page_icon="",
    layout="centered",
)


def get_active_agent() -> Agent:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = get_or_create_default_thread_id()

    agent = st.session_state.get("agent")
    if (
        not isinstance(agent, Agent)
        or not hasattr(agent, "stream")
        or agent.thread_id != st.session_state.thread_id
    ):
        agent = create_agent(st.session_state.thread_id)
        st.session_state.agent = agent
    return agent


def create_agent(thread_id: str) -> Agent:
    try:
        return Agent(thread_id=thread_id)
    except TypeError as exc:
        if "thread_id" not in str(exc):
            raise
        agent = Agent()
        agent.thread_id = thread_id
        return agent


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Settings")
        st.caption("Runtime configuration")
        st.text_input("Model", value=settings.anthropic_model, disabled=True)
        st.text_input(
            "Base URL",
            value=settings.anthropic_base_url or "Default Anthropic endpoint",
            disabled=True,
        )
        st.number_input(
            "Temperature",
            value=float(settings.model_temperature),
            disabled=True,
        )

        if st.button("Clear chat", use_container_width=True):
            get_active_agent().reset()
            st.session_state.messages = []
            st.session_state.thread_id = rotate_default_thread_id()
            st.session_state.agent = create_agent(st.session_state.thread_id)
            st.rerun()


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_pending_tool_approval() -> bool:
    agent = get_active_agent()
    pending = agent.get_pending_interrupt()
    if not pending or pending.get("type") != "tool_approval":
        return False

    st.warning(pending.get("message") or "Approve these tool calls?")
    for index, tool_call in enumerate(pending.get("tool_calls") or [], start=1):
        tool_name = tool_call.get("tool_name") or "unknown"
        reason = tool_call.get("reason") or "Approval required."
        risk_summary = tool_call.get("risk_summary") or ""
        with st.expander(f"{index}. {tool_name}", expanded=True):
            st.markdown(f"**Reason:** {reason}")
            if risk_summary:
                st.code(risk_summary, language="text")

    approve_column, reject_column = st.columns(2)
    with approve_column:
        approve = st.button("Approve", type="primary", use_container_width=True)
    with reject_column:
        reject = st.button("Reject", use_container_width=True)

    if approve or reject:
        approved = bool(approve)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": "Approved tool calls." if approved else "Rejected tool calls.",
            }
        )
        with st.chat_message("assistant"):
            response = st.write_stream(agent.resume_tool_approval(approved))
        if response:
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
        st.rerun()

    return True


def main() -> None:
    init_state()
    render_sidebar()

    st.title("LangGraph Agent")

    if not settings.anthropic_api_key:
        st.error("Set ANTHROPIC_API_KEY in .env before chatting.")
        st.stop()

    render_chat_history()

    if render_pending_tool_approval():
        return

    prompt = st.chat_input("Ask the agent...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(stream_agent_response(prompt))

    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})

    if get_active_agent().get_pending_interrupt():
        st.rerun()


def stream_agent_response(prompt: str):
    try:
        yield from get_active_agent().stream(prompt)
    except Exception as exc:
        yield f"Agent error: {exc}"


if __name__ == "__main__":
    main()
