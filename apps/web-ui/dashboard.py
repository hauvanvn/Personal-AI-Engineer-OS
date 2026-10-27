import json
import uuid
from pathlib import Path
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8080"

UI_DIR = Path(__file__).resolve().parent
BOT_ICON_PATH = str(UI_DIR / "resource" / "bot.png")

st.set_page_config(
    page_title="PaPa - Personal AI Engineer OS",
    page_icon=BOT_ICON_PATH,
    layout="wide",
)

if Path(BOT_ICON_PATH).exists():
    st.logo(BOT_ICON_PATH)

# Initialize session states
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None


def fetch_long_term_memories() -> str:
    try:
        res = requests.get(f"{API_BASE_URL}/memories", timeout=5)
        if res.status_code == 200:
            return res.json().get("user_profile", "No facts stored yet.")
    except Exception as e:
        return f"Backend offline: {str(e)}"
    return "Failed to load memories."


# ==========================================
# LEFT SIDEBAR: MEMORY & SYSTEM STATUS
# ==========================================
with st.sidebar:
    st.title("PaPa - Second Brain OS")
    st.caption("Architecture: LangGraph + MCP + Milvus + Postgres")

    st.divider()

    # Session Management
    col1, col2 = st.columns([3, 2])
    with col1:
        st.text_input("Thread ID", value=st.session_state.thread_id, disabled=True)
    with col2:
        st.write("")
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state.chat_history = []
            st.session_state.pending_interrupt = None
            st.rerun()

    st.divider()

    # Real-time Long-Term User Memory Panel
    st.subheader("Long-Term User Profile")
    st.caption("Auto-extracted into PostgreSQL across sessions:")
    if st.button("Refresh Profile", use_container_width=True):
        st.rerun()

    memory_text = fetch_long_term_memories()
    st.code(memory_text, language="markdown")

    st.divider()

    # Active MCP Subsystems Indicator
    st.subheader("Connected MCP Servers")
    st.markdown(
        """
        - **`second-brain-rag`** *(Milvus Hybrid + Time-Decay)*
        - **`dev-life-workspace`** *(Postgres Tasks + Local Git)*
        """
    )


# ==========================================
# MAIN PANEL: CHAT & MCP TRACE VISUALIZATION
# ==========================================
st.title("PaPa - Personal AI Engineer Assistant")
st.caption(
    "Ask about system architecture notes, inspect local Git status, or manage engineering tasks."
)

# Render conversation history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        # Render collapsible MCP Tool execution traces if available
        if msg.get("tool_traces"):
            for trace in msg["tool_traces"]:
                with st.expander(f"MCP Tool Executed: `{trace['tool_name']}`", expanded=False):
                    st.code(trace["output"], language="text")

        st.markdown(msg["content"])
        if msg.get("latency"):
            st.caption(f"Latency: {msg['latency']}s")


# ==========================================
# HUMAN-IN-THE-LOOP (HITL) APPROVAL CARD
# ==========================================
if st.session_state.pending_interrupt:
    interrupt_data = st.session_state.pending_interrupt
    sensitive_calls = interrupt_data.get("sensitive_calls", [])

    st.warning("**Security Gatekeeper: Human-in-the-Loop Approval Required**")

    for call in sensitive_calls:
        tool_name = call.get("name", "unknown_tool")
        tool_args = call.get("args", {})

        with st.container(border=True):
            st.markdown(f"**Pending Mutate Tool:** `{tool_name}`")
            edited_args_str = st.text_area(
                "Inspect or Edit Tool Arguments (JSON):",
                value=json.dumps(tool_args, indent=2),
                height=130,
                key=f"edit_{call.get('id', tool_name)}",
            )

            c1, c2, c3 = st.columns(3)
            action_taken = None
            payload_args = None

            with c1:
                if st.button("Approve", type="primary", use_container_width=True):
                    action_taken = "approve"
            with c2:
                if st.button("Save Edited & Run", use_container_width=True):
                    try:
                        payload_args = json.loads(edited_args_str)
                        action_taken = "edit"
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format in arguments!")
            with c3:
                if st.button("Reject Action", use_container_width=True):
                    action_taken = "reject"

            if action_taken:
                with st.spinner(f"Resuming graph with decision: {action_taken.upper()}..."):
                    res = requests.post(
                        f"{API_BASE_URL}/chat/resume",
                        json={
                            "thread_id": st.session_state.thread_id,
                            "action": action_taken,
                            "new_args": payload_args,
                        },
                        timeout=30,
                    ).json()

                st.session_state.pending_interrupt = None
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": res.get("response", ""),
                        "tool_traces": res.get("tool_traces", []),
                        "latency": res.get("latency_sec"),
                    }
                )
                st.rerun()


# ==========================================
# CHAT INPUT HANDLER
# ==========================================
prompt = st.chat_input(
    "Type a command (e.g., 'Check my Second Brain for OCR queue tech and create a high-priority task')...",
    disabled=st.session_state.pending_interrupt is not None,
)

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Orchestrating LangGraph & MCP Servers..."):
            try:
                res = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "thread_id": st.session_state.thread_id,
                        "message": prompt,
                    },
                    timeout=30,
                ).json()

                status = res.get("status")
                traces = res.get("tool_traces", [])
                latency = res.get("latency_sec")

                if status == "interrupted_for_approval":
                    # Save intermediate read-only traces if any executed before the interrupt
                    if traces:
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": "*Paused execution waiting for human confirmation on state-mutating tool...*",
                                "tool_traces": traces,
                                "latency": latency,
                            }
                        )
                    st.session_state.pending_interrupt = res.get("pending_approval")
                    st.rerun()
                else:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": res.get("response", ""),
                            "tool_traces": traces,
                            "latency": latency,
                        }
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Error communicating with Orchestrator API: {str(e)}")