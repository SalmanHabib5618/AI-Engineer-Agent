"""AI Engineer Mentor Agent: core logic (no Streamlit imports, so it is easy to test).

Rules from the PRD that live in code, not in the model:
  * intent routing picks a response style for each request
  * consent flow: when the model says it cannot answer reliably, the agent
    waits for your yes/no before explaining any alternative
  * clear errors, and no silent fallback to another provider
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator

CONSENT_MARKER = "[[NEEDS_CONSENT]]"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
# OpenAI GPT models (need an OpenAI API key). Check platform.openai.com/docs/models for current names.
OPENAI_MODELS = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-5-mini"]
MODELS = GROQ_MODELS + OPENAI_MODELS
PROVIDER_NAMES = {"groq": "Groq", "openai": "OpenAI"}


def provider_of(model: str) -> str:
    """'openai' for GPT models, otherwise 'groq'."""
    return "openai" if model in OPENAI_MODELS or model.startswith(("gpt-", "o1", "o3", "o4")) else "groq"


def _fixed_temperature(model: str) -> bool:
    """GPT-5 and o-series models only accept the default temperature, so we do not send one."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))
MAX_HISTORY_MESSAGES = 14  # keeps requests small and fast

# ---------------------------------------------------------------------------
# Curriculum: Python and PostgreSQL/SQL from absolute zero
# ---------------------------------------------------------------------------
CURRICULUM: list[dict] = [
    # Python
    {"id": "py01", "track": "Python", "title": "Setup and your first program",
     "goals": "Install Python and a code editor, open a terminal, run a .py file, use print()."},
    {"id": "py02", "track": "Python", "title": "Variables and data types",
     "goals": "int, float, str, bool, naming variables, type(), comments."},
    {"id": "py03", "track": "Python", "title": "Operators, input and strings",
     "goals": "Arithmetic, input(), f-strings, common string methods."},
    {"id": "py04", "track": "Python", "title": "Decisions with if / elif / else",
     "goals": "Comparisons, and / or / not, indentation rules."},
    {"id": "py05", "track": "Python", "title": "Loops",
     "goals": "for, while, range(), break and continue."},
    {"id": "py06", "track": "Python", "title": "Lists and tuples",
     "goals": "Indexing, slicing, list methods, looping over a list."},
    {"id": "py07", "track": "Python", "title": "Dictionaries and sets",
     "goals": "Key/value data, JSON-like structures, membership tests."},
    {"id": "py08", "track": "Python", "title": "Functions",
     "goals": "def, parameters, return values, scope, docstrings."},
    {"id": "py09", "track": "Python", "title": "Errors and debugging",
     "goals": "Reading tracebacks, try/except, print debugging."},
    {"id": "py10", "track": "Python", "title": "Files and JSON",
     "goals": "Read and write text files, the with statement, the json module."},
    {"id": "py11", "track": "Python", "title": "Modules, pip and virtual environments",
     "goals": "import, pip install, venv, requirements.txt."},
    {"id": "py12", "track": "Python", "title": "Classes and objects",
     "goals": "class, __init__, methods, why objects are useful."},
    {"id": "py13", "track": "Python", "title": "Project: a command-line app",
     "goals": "Combine everything into a small, tested command-line program."},
    # PostgreSQL and SQL
    {"id": "db01", "track": "PostgreSQL", "title": "What a database is, and installing PostgreSQL",
     "goals": "Tables, rows, columns; install PostgreSQL; use psql or pgAdmin; CREATE DATABASE."},
    {"id": "db02", "track": "PostgreSQL", "title": "Creating tables and data types",
     "goals": "CREATE TABLE, INTEGER, TEXT, NUMERIC, DATE, BOOLEAN, primary keys."},
    {"id": "db03", "track": "PostgreSQL", "title": "Inserting and reading data",
     "goals": "INSERT, SELECT, choosing columns, aliases."},
    {"id": "db04", "track": "PostgreSQL", "title": "Filtering and sorting",
     "goals": "WHERE, AND / OR, LIKE, IN, ORDER BY, LIMIT."},
    {"id": "db05", "track": "PostgreSQL", "title": "Changing data safely",
     "goals": "UPDATE, DELETE, why WHERE matters, practising safely."},
    {"id": "db06", "track": "PostgreSQL", "title": "Summaries and grouping",
     "goals": "COUNT, SUM, AVG, MIN, MAX, GROUP BY, HAVING."},
    {"id": "db07", "track": "PostgreSQL", "title": "Keys, constraints and relationships",
     "goals": "Primary and foreign keys, NOT NULL, UNIQUE, one-to-many design."},
    {"id": "db08", "track": "PostgreSQL", "title": "Joins",
     "goals": "INNER JOIN, LEFT JOIN, joining two and three tables."},
    {"id": "db09", "track": "PostgreSQL", "title": "Subqueries and CTEs",
     "goals": "Nested queries and WITH (common table expressions)."},
    {"id": "db10", "track": "PostgreSQL", "title": "Indexes and query plans",
     "goals": "What an index is, CREATE INDEX, reading EXPLAIN."},
    {"id": "db11", "track": "PostgreSQL", "title": "Transactions",
     "goals": "BEGIN, COMMIT, ROLLBACK, the idea of ACID."},
    {"id": "db12", "track": "PostgreSQL", "title": "PostgreSQL from Python",
     "goals": "psycopg, connecting, parameterised queries, a small Python + PostgreSQL project."},
]
LESSONS: dict[str, dict] = {l["id"]: l for l in CURRICULUM}

# Suggested study order: the two tracks are interleaved.
ORDER = ["py01", "py02", "py03", "db01", "py04", "py05", "db02", "db03", "py06", "py07",
         "db04", "db05", "py08", "py09", "db06", "db07", "py10", "py11", "db08", "db09",
         "py12", "db10", "db11", "py13", "db12"]

# After both tracks: the next stage of the AI Engineer path (taught at a brief level for now).
NEXT_STAGE = ("APIs and the requests library, NumPy and pandas, maths for machine learning, "
              "machine learning basics, building apps with an LLM API such as Groq, "
              "embeddings and RAG (including pgvector in PostgreSQL), agents, evaluation.")


def next_lesson(completed: set[str]) -> dict | None:
    """First lesson in the suggested order that is not completed yet."""
    for lid in ORDER:
        if lid not in completed:
            return LESSONS[lid]
    return None


LEVELS = {
    "Beginner": "The learner is a beginner. Explain simply first, define every new term, "
    "and go deeper only when asked or when they show readiness.",
    "Intermediate": "The learner knows the basics. Be concise, use correct terminology, "
    "and include short code or maths where useful.",
    "Advanced": "The learner is advanced. Skip basics, go straight to technical depth and trade-offs.",
}

SYSTEM_PROMPT = f"""You are AI Engineer Mentor, a patient teacher for one learner who is starting
from absolute zero. The learner has NO prior knowledge of Python, SQL or PostgreSQL and is working
toward becoming an AI Engineer. Your main job is to teach Python and PostgreSQL (with SQL) from
scratch, step by step. The learner finished FSc Pre-Engineering, so school-level maths is fine.

How to teach:
- Use plain words. Define every new term the first time you use it. Never assume knowledge.
- Teach ONE small idea at a time with a tiny, complete example. Show exact commands and code in
  code blocks.
- After each idea, give a short "Try it" task and STOP. Wait for the learner's answer or output.
- Check their work, explain mistakes kindly and clearly, then continue.
- After about three ideas, give a quick check of 3 short questions. When they pass, tell them they
  can tick the lesson as complete in the sidebar.
- Never dump a whole lesson at once. Keep each reply short enough to finish in a few minutes.
- You cannot run code. The learner runs code on their own laptop (or in an online editor) and pastes
  the result back. For setup steps, ask which operating system they use (Windows, Mac or Linux) once
  if you do not know, then give steps for that system.
- Use PostgreSQL syntax for SQL. Use small practice data such as students, courses or books.
- If asked about topics beyond Python and PostgreSQL, answer briefly and honestly, and say where
  it fits later in the path: {NEXT_STAGE}

Honesty rules (strict):
- Never invent citations, links, model names, release dates, benchmark numbers or library features.
- You cannot browse the internet. For recent or fast-changing facts, say your knowledge may be
  outdated and point to the official documentation.
- Separate what is well established from your inference, and say when you are unsure.
- Never claim you ran, tested or verified code. You can only reason about it.

When you cannot answer the question reliably:
1. Say briefly what you could not establish.
2. Give any part you can support.
3. Offer up to three alternatives (a related concept, a prerequisite, or a way to verify).
4. Ask which one they want, then STOP. Do not explain any alternative yet.
5. Put this exact token alone on the very last line: {CONSENT_MARKER}
Use this only when you truly cannot answer. If the question is merely vague, ask one short
clarifying question instead (no token). If it is clear, just answer.

For debugging: ask what they expected, find the likely cause, give the smallest fix, explain why it
works, and say how to check it. For practice, give a challenge and do not reveal the full solution
unless asked."""

# (intent, pattern, style hint). First match wins.
_ROUTES: list[tuple[str, str, str]] = [
    ("lesson", r"(^\s*(teach me|start|continue|resume|next)\b.*\blesson\b|\bfrom (zero|scratch)\b|\bteach me\b)",
     "Teach step by step: state the goal, teach ONE small idea with a tiny example, give a 'Try it' task and wait."),
    ("debug", r"\b(error|traceback|exception|bug|fails?|failing|not working|doesn'?t work|crash)\b",
     "Use the debugging structure: expected behaviour, cause, minimal fix, why it works, how to verify."),
    ("quiz", r"\b(quiz|test me|mcq|question me)\b",
     "Run a quiz: ask ONE question at a time, wait for the answer, then grade it and explain."),
    ("assess", r"\b(check my|evaluate my|review my|is my (answer|code) (right|correct)|grade)\b",
     "Evaluate the learner's actual work. Separate correct, incomplete and incorrect parts."),
    ("practice", r"\b(exercise|practi[sc]e|challenge|drill|homework)\b",
     "Give a suitable exercise with a hint available on request. Do not give the full solution."),
    ("roadmap", r"\b(roadmap|what should i (learn|study)|next topic|study plan|today'?s plan|learning path)\b",
     "Give a short, ordered plan with reasons, using the learner's progress below. Python and PostgreSQL lessons alternate."),
    ("project", r"\b(build|project|create an? (app|agent|bot)|chatbot)\b",
     "Break the work into small, testable milestones that fit a learner's current skills."),
    ("code", r"\b(explain (this )?(code|function)|what does this (code|function)|write (a )?(function|script|code))\b",
     "Explain the objective, show clear code, describe key parts, and say how to run and check it."),
    ("career", r"\b(career|job|hire|hiring|skills? (do|does)|portfolio|interview)\b",
     "Be realistic. Do not promise job-readiness or timelines."),
    ("concept", r"\b(what is|what are|explain|how does|how do|why)\b",
     "Explain simply first, then add depth if useful."),
]

_YES = re.compile(r"^\s*(yes|yeah|yep|yup|sure|ok(ay)?|please( do)?|go ahead|do it|y|"
                  r"(the )?(first|second|third|1st|2nd|3rd|1|2|3)( one| option)?|option\s*[123])\b", re.I)
_NO = re.compile(r"^\s*(no|nope|nah|n|not now|no thanks|don'?t|do not|skip|never ?mind)\b", re.I)


class AgentError(Exception):
    """A problem with a friendly message that is safe to show to the user."""


def detect_intent(text: str) -> tuple[str, str]:
    """Return (intent, style hint) for a user message."""
    for intent, pattern, hint in _ROUTES:
        if re.search(pattern, text, re.I):
            return intent, hint
    if len(text.split()) <= 3 and re.match(r"^\s*(hi|hello|hey|thanks|thank you|salam|assalam)", text, re.I):
        return "chat", "Reply briefly and warmly."
    return "general", "Answer directly and helpfully."


def classify_consent_reply(text: str) -> str:
    """Return 'yes', 'no', 'unclear' or 'new' for a reply to a consent question."""
    if _NO.match(text):
        return "no"
    if _YES.match(text):
        return "yes"
    if len(text.split()) <= 4 and "?" not in text:
        return "unclear"
    return "new"  # a longer message is treated as a new request


def make_client(api_key: str, provider: str = "groq"):
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise AgentError("The 'openai' package is not installed. Add it to requirements.txt.") from e
        return OpenAI(api_key=api_key)
    try:
        from groq import Groq
    except ImportError as e:  # pragma: no cover
        raise AgentError("The 'groq' package is not installed. Check requirements.txt.") from e
    return Groq(api_key=api_key)


class AIEngineerAgent:
    def __init__(self, client, model: str = DEFAULT_MODEL, level: str = "Beginner",
                 temperature: float = 0.4):
        self.client = client
        self.model = model
        self.level = level
        self.temperature = temperature
        self.history: list[dict] = []
        self.awaiting_consent = False
        self.completed: set[str] = set()
        self.current: str | None = None

    # ---- public API -------------------------------------------------
    def reset(self) -> None:
        self.history.clear()
        self.awaiting_consent = False
        self.current = None

    def lesson_request(self, lesson_id: str) -> str:
        """Message to send when the learner starts (or continues) a lesson."""
        self.current = lesson_id
        lesson = LESSONS[lesson_id]
        return f"Teach me lesson {lesson_id}: {lesson['title']}, from zero."

    def _progress_note(self) -> str:
        done = [f"{LESSONS[i]['title']}" for i in ORDER if i in self.completed]
        nxt = next_lesson(self.completed)
        lines = ["Learner progress: " + (", ".join(done) if done else "nothing completed yet") + "."]
        if self.current in LESSONS:
            cur = LESSONS[self.current]
            lines.append(f"Current lesson: {cur['title']}. Goals: {cur['goals']}")
        if nxt:
            lines.append(f"Suggested next lesson: {nxt['title']}.")
        else:
            lines.append("Both tracks are complete. Move on to: " + NEXT_STAGE)
        return "\n".join(lines)

    def chat(self, user_text: str) -> Iterator[str]:
        """Yield the reply in chunks. Updates history only if the reply succeeds."""
        user_text = user_text.strip()
        extra_note = None

        if self.awaiting_consent:
            verdict = classify_consent_reply(user_text)
            if verdict == "no":
                reply = "Okay, I won't switch topics. Ask me anything else when you're ready."
                self.awaiting_consent = False
                self._remember(user_text, reply)
                yield reply
                return
            if verdict == "unclear":
                reply = ("I'm not sure if that was a yes. Do you want me to explain one of the "
                         "alternatives I offered? Please say yes, no, or name the one you want.")
                self._remember(user_text, reply)
                yield reply
                return
            if verdict == "yes":
                extra_note = ("The learner approved an alternative you offered in your last message. "
                              "Explain the one they named, or the first one if they did not name one. "
                              "Stay honest about uncertainty.")
            # 'new' falls through and is answered as a normal request
            self.awaiting_consent = False

        intent, hint = detect_intent(user_text)
        system = (f"{SYSTEM_PROMPT}\n\n{LEVELS.get(self.level, LEVELS['Beginner'])}\n"
                  f"{self._progress_note()}\nThis request looks like: {intent}. {hint}")
        if extra_note:
            system += f"\n{extra_note}"

        messages = [{"role": "system", "content": system}]
        messages += self.history[-MAX_HISTORY_MESSAGES:]
        messages.append({"role": "user", "content": user_text})

        full: list[str] = []
        for piece in self._hide_marker(self._stream(messages), full):
            yield piece

        raw = "".join(full)
        self.awaiting_consent = CONSENT_MARKER in raw
        self._remember(user_text, raw.replace(CONSENT_MARKER, "").rstrip())

    # ---- internals --------------------------------------------------
    def _remember(self, user_text: str, reply: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})

    def _stream(self, messages: list[dict]) -> Iterator[str]:
        try:
            kwargs = {"model": self.model, "messages": messages, "stream": True}
            if not _fixed_temperature(self.model):
                kwargs["temperature"] = self.temperature
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except AgentError:
            raise
        except Exception as e:  # map provider errors to friendly messages
            raise AgentError(_friendly_error(e, provider_of(self.model))) from e

    @staticmethod
    def _hide_marker(chunks: Iterable[str], sink: list[str]) -> Iterator[str]:
        """Pass chunks through but never show the consent marker, even if split across chunks."""
        hold = len(CONSENT_MARKER)
        buf = ""
        for c in chunks:
            sink.append(c)
            buf += c
            if len(buf) > hold:
                yield buf[:-hold].replace(CONSENT_MARKER, "")
                buf = buf[-hold:]
        yield buf.replace(CONSENT_MARKER, "").rstrip()


def _friendly_error(e: Exception, provider: str = "groq") -> str:
    name = type(e).__name__
    text = str(e)
    pname = PROVIDER_NAMES.get(provider, "The provider")
    key_name = "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"
    other = "another model" if provider == "openai" else "the smaller model"
    if name == "AuthenticationError" or "401" in text:
        return f"{pname} rejected the API key. Check {key_name} in your secrets."
    if "insufficient_quota" in text or "exceeded your current quota" in text.lower():
        return (f"{pname} says there is no credit left on this key. "
                "Add billing or credit on your OpenAI account, or switch to a Groq model.")
    if name == "RateLimitError" or "429" in text:
        return f"{pname} rate limit reached. Wait a minute, or switch to {other} in the sidebar."
    if name in ("APIConnectionError", "APITimeoutError"):
        return f"Could not reach {pname}. Check your internet connection and try again."
    if "model" in text.lower() and ("not found" in text.lower() or "decommission" in text.lower()
                                    or "does not exist" in text.lower()):
        return f"That model is not available on {pname} (for this key). Pick another model in the sidebar."
    return f"{pname} returned an error ({name}). Please try again. Details: {text[:200]}"
