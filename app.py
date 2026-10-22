"""Streamlit interface for the AI Mentor Agent (powered by Groq).

Teaches Python and PostgreSQL (SQL) from absolute zero.
"""
import html
import json
import os

import streamlit as st

from ai_engineer_agent import (
    DEFAULT_MODEL, LEVELS, LESSONS, MODELS, ORDER,
    AgentError, AIEngineerAgent, make_client, next_lesson,
)

st.set_page_config(
    page_title="AI Mentor",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",  # keeps the main page clean, like DS Mentor
)

# ---------- look and feel: cream page, serif title, terracotta accent ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@600;700&display=swap');

:root {
    --bg: #FCFAF6;
    --accent: #B85410;
    --ink: #2B2118;
    --muted: #A89B8A;
    --line: #E9E3D8;
}

.stApp { background: var(--bg); color: var(--ink); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp textarea, .stApp input { font-weight: 500; }
.stApp strong, .stApp b, .stApp h1, .stApp h2, .stApp h3 { font-weight: 700; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #F5F1E8; }
.block-container { padding-top: 3.5rem; max-width: 760px; }

/* Header */
.mentor-header { text-align: center; margin-bottom: 2.2rem; }
.mentor-logo {
    width: 40px; height: 40px; border-radius: 50%;
    border: 1.5px solid var(--accent);
    display: inline-flex; align-items: center; justify-content: center;
    vertical-align: middle; margin-right: 10px;
}
.mentor-title {
    font-family: 'Lora', serif; font-weight: 700; font-size: 2rem;
    color: var(--ink); vertical-align: middle;
}
.mentor-sub { color: var(--muted); font-size: 0.82rem; font-weight: 500; margin-top: 0.5rem; }

/* Your message: right side, terracotta bubble */
.user-row { display: flex; justify-content: flex-end; margin: 6px 0 18px; }
.user-bubble {
    background: var(--accent); color: #fff; padding: 10px 18px;
    border-radius: 16px 16px 0 16px; max-width: 80%;
    font-size: 0.95rem; font-weight: 600; line-height: 1.4; white-space: pre-wrap;
}

/* Mentor reply: left side, white card */
div[class*="st-key-asst_"] {
    background: #fff; border: 1px solid var(--line);
    border-radius: 14px; padding: 12px 18px; max-width: 88%;
    margin: 6px 0 18px;
}
div[class*="st-key-asst_"] p { margin-bottom: 0.4rem; }

/* Chat box */
[data-testid="stChatInput"] > div {
    background: #fff; border: 1px solid var(--line); border-radius: 10px;
}
[data-testid="stChatInput"] textarea { background: #F0F2F6; border-radius: 8px; }
/* + (upload) and send arrow: black icons, bold, white background */
[data-testid="stChatInput"] button,
[data-testid="stChatInput"] button:hover,
[data-testid="stChatInput"] button:focus,
[data-testid="stChatInput"] button:active,
[data-testid="stChatInput"] button:disabled {
    background: #fff !important; background-color: #fff !important;
    border: 1.5px solid #000 !important; border-radius: 8px !important;
    color: #000 !important; opacity: 1 !important;
}
[data-testid="stChatInput"] button svg { color: #000 !important; }
[data-testid="stChatInput"] button svg path {
    stroke: #000; stroke-width: 0.8px; stroke-linejoin: round;
}
[data-testid="stChatInput"] button span {
    color: #000 !important; font-weight: 700 !important;
    font-variation-settings: 'wght' 700; -webkit-text-stroke: 0.4px #000;
}

/* Calm focus outline (no red) */
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--accent) !important; box-shadow: none !important;
}

/* Hide the footer / "Hosted with Streamlit" badge and keep the bottom area cream */
footer, [data-testid="stFooter"], [class*="viewerBadge"], [class*="_viewerBadge"],
[class*="_profileContainer"], [class*="StatusWidget"] { display: none !important; visibility: hidden !important; }
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: var(--bg) !important; }
</style>
""",
    unsafe_allow_html=True,
)


def get_api_key() -> str:
    """Secrets first (Streamlit Cloud), then environment, then the sidebar box."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets file locally
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "") or st.session_state.get("typed_key", "")


def completed_ids() -> set:
    return {lid for lid in LESSONS if st.session_state.get(f"done_{lid}")}


def render_user(text: str) -> None:
    st.markdown(
        f'<div class="user-row"><div class="user-bubble">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


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

# ---------- header ----------
st.markdown(
    """
<div class="mentor-header">
  <div>
    <span class="mentor-logo">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#B85410"
           stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="7.5" cy="8" r="1.5"/><circle cx="16.5" cy="8" r="1.5"/>
        <circle cx="12" cy="17" r="1.5"/>
        <path d="M9 8h6M8.2 9.3l2.9 6.2M15.8 9.3l-2.9 6.2"/>
      </svg>
    </span><span class="mentor-title">AI Mentor</span>
  </div>
  <div class="mentor-sub">Your personal AI Engineer mentor — ask anything.</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- state ----------
api_key = get_api_key()

if not api_key:
    st.info("Add your Groq API key to continue. On Streamlit Cloud: App settings → Secrets → "
            "`GROQ_API_KEY = \"gsk_...\"`. Locally: open the sidebar and use the key box, "
            "or add `.streamlit/secrets.toml`.")
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
for i, m in enumerate(st.session_state.messages):
    if m["role"] == "user":
        render_user(m.get("display", m["content"]))
    else:
        with st.container(key=f"asst_{i}"):
            st.markdown(m["content"])

chat_value = st.chat_input(
    "Ask your AI Engineer question...",
    accept_file=True,  # shows the + button inside the prompt bar
    file_type=["py", "txt", "sql", "csv", "md", "json"],
)

prompt, display = None, None
if chat_value:
    text = (chat_value["text"] or "").strip()
    prompt, display = text, text
    for f in chat_value["files"]:
        try:
            body = f.getvalue().decode("utf-8")[:20000]
        except UnicodeDecodeError:
            continue
        prompt += f"\n\n[Attached file: {f.name}]\n```\n{body}\n```"
        display += f"\n📎 {f.name}"
    prompt, display = prompt.strip(), display.strip()
if start_prompt:
    prompt = agent.lesson_request(start_prompt[1])
    display = prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "display": display})
    render_user(display)

    with st.container(key=f"asst_{len(st.session_state.messages)}"):
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
