"""Streamlit interface for the AI Engineer Mentor Agent (powered by Groq).

Teaches Python and PostgreSQL (SQL) from absolute zero.
"""
import json
import os

import streamlit as st

from ai_engineer_agent import (
    DEFAULT_MODEL, LEVELS, LESSONS, MODELS, ORDER,
    AgentError, AIEngineerAgent, make_client, next_lesson,
)

st.set_page_config(page_title="AI Engineer Mentor", page_icon="🧠", layout="centered")


def get_api_key() -> str:
    """Secrets first (Streamlit Cloud), then environment, then the sidebar box."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets file locally
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "") or st.session_state.get("typed_key", "")


def completed_ids() -> set:
    return {lid for lid in LESSONS if st.session_state.get(f"done_{lid}")}


# ---------- sidebar: settings ----------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", MODELS, index=MODELS.index(DEFAULT_MODEL))
    level = st.selectbox("Explanation level", list(LEVELS), index=0)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.4, 0.1)

    if not get_api_key():
        st.text_input("Groq API key", type="password", key="typed_key",
                      help="Only kept for this browser session. On Streamlit Cloud, use Secrets instead.")

    if st.button("New conversation", use_container_width=True):
        st.session_state.pop("agent", None)
        st.session_state.pop("messages", None)
        st.rerun()

    # ---------- sidebar: progress save / load ----------
    # The app keeps nothing on the server, so save your progress file and load it next time.
    st.divider()
    st.subheader("Your progress")
    uploaded = st.file_uploader("Load saved progress", type="json", key="progress_file")
    if uploaded is not None and st.session_state.get("last_upload") != uploaded.file_id:
        try:
            data = json.load(uploaded)
            saved = set(data.get("completed", []))
            for lid in LESSONS:
                st.session_state[f"done_{lid}"] = lid in saved
            st.session_state["last_upload"] = uploaded.file_id
            st.success("Progress loaded.")
        except (ValueError, AttributeError):
            st.error("That file is not a valid progress file.")

    # ---------- sidebar: lessons ----------
    done = completed_ids()
    nxt = next_lesson(done)
    st.progress(len(done) / len(LESSONS), text=f"{len(done)} of {len(LESSONS)} lessons done")
    start_prompt = None
    if nxt:
        if st.button(f"▶ Start next: {nxt['title']}", use_container_width=True, type="primary"):
            start_prompt = ("start", nxt["id"])
    else:
        st.success("Both tracks complete! Ask what comes next.")

    for track in ("Python", "PostgreSQL"):
        with st.expander(f"{track} lessons", expanded=False):
            for lid in ORDER:
                lesson = LESSONS[lid]
                if lesson["track"] != track:
                    continue
                c1, c2 = st.columns([0.8, 0.2])
                c1.checkbox(lesson["title"], key=f"done_{lid}")
                if c2.button("Go", key=f"go_{lid}", help="Start this lesson"):
                    start_prompt = ("start", lid)

    st.download_button("Save progress", json.dumps({"completed": sorted(completed_ids())}, indent=2),
                       "mentor_progress.json", "application/json", use_container_width=True)

    st.caption("Your messages are sent to Groq to generate replies. "
               "This app stores nothing on a server; closing the tab clears the chat, "
               "so save your progress file.")

# ---------- state ----------
api_key = get_api_key()
st.title("🧠 AI Engineer Mentor")
st.caption("Learn Python and PostgreSQL from zero, one small step at a time. "
           "It tells you when it does not know.")

if not api_key:
    st.info("Add your Groq API key to continue. On Streamlit Cloud: App settings → Secrets → "
            "`GROQ_API_KEY = \"gsk_...\"`. Locally: use the sidebar box or `.streamlit/secrets.toml`.")
    st.stop()

if "agent" not in st.session_state:
    try:
        st.session_state.agent = AIEngineerAgent(make_client(api_key), model, level, temperature)
    except AgentError as e:
        st.error(str(e))
        st.stop()
    st.session_state.messages = []

agent: AIEngineerAgent = st.session_state.agent
# apply sidebar changes to the live agent
agent.client = make_client(api_key)
agent.model, agent.level, agent.temperature = model, level, temperature
agent.completed = completed_ids()

# ---------- chat ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if not st.session_state.messages:
    st.markdown("**New here?** Press **Start next** in the sidebar and I'll teach you from the very "
                "first step. Or ask anything, for example: `What is a variable?` · "
                "`What is a database?` · `Why does this code fail?`")

prompt = st.chat_input("Ask a question, or paste your code or output here")
if start_prompt:
    prompt = agent.lesson_request(start_prompt[1])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(agent.chat(prompt))
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except AgentError as e:
            st.error(str(e))
            st.session_state.messages.pop()  # drop the unanswered question

if st.session_state.messages:
    transcript = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in st.session_state.messages)
    st.sidebar.download_button("Download chat (.md)", transcript, "mentor_chat.md", "text/markdown",
                               use_container_width=True)
