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


def main() -> None:
    init_state()
    render_sidebar()

    st.title("LangGraph Agent")

    if not settings.anthropic_api_key:
        st.error("Set ANTHROPIC_API_KEY in .env before chatting.")
        st.stop()

    render_chat_history()

    prompt = st.chat_input("Ask the agent...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(stream_agent_response(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})


def stream_agent_response(prompt: str):
    try:
        yield from get_active_agent().stream(prompt)
    except Exception as exc:
        yield f"Agent error: {exc}"


if __name__ == "__main__":
    main()
