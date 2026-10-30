import json
import uuid
from pathlib import Path
from typing import List, Dict, Any
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8080"

UI_DIR = Path(__file__).resolve().parent
BOT_ICON_PATH = str(UI_DIR / "resource" / "bot.png")

st.set_page_config(
    page_title="PaPa - Personal AI Engineer OS",
    page_icon=BOT_ICON_PATH if Path(BOT_ICON_PATH).exists() else None,
    layout="wide",
)

if Path(BOT_ICON_PATH).exists():
    st.logo(BOT_ICON_PATH)

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


def fetch_knowledge_documents() -> List[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE_URL}/knowledge/documents", timeout=5)
        if res.status_code == 200:
            return res.json().get("documents", [])
    except Exception:
        return []
    return []


def get_avatar(role: str):
    if role == "assistant" and Path(BOT_ICON_PATH).exists():
        return BOT_ICON_PATH
    return None


# ==========================================
# LEFT SIDEBAR: MEMORY, UPLOAD & SYSTEM STATUS
# ==========================================
with st.sidebar:
    title_col1, title_col2 = st.columns([1, 4], vertical_alignment="center")
    with title_col1:
        if Path(BOT_ICON_PATH).exists():
            st.image(BOT_ICON_PATH, width=48)
    with title_col2:
        st.title("Second Brain OS")

    st.caption("Architecture: LangGraph + MCP + Milvus + Postgres")

    st.divider()

    # Session Management
    col1, col2 = st.columns([3, 2])
    with col1:
        st.text_input("Thread ID", value=st.session_state.thread_id, disabled=True)
    with col2:
        st.write("")
        if st.button("New Chat", use_container_width=True):
            st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state.chat_history = []
            st.session_state.pending_interrupt = None
            st.rerun()

    st.divider()

    # Knowledge Base Upload & Background Ingestion Panel
    st.subheader("Knowledge Base Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Documents (.pdf, .md, .txt)",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    if st.button(
        "Ingest in Background",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        files_payload = [
            ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
            for f in uploaded_files
        ]
        try:
            res = requests.post(
                f"{API_BASE_URL}/knowledge/upload",
                files=files_payload,
                timeout=15,
            )
            if res.status_code == 202:
                data = res.json()
                st.success(data.get("message", "Queued for background ingestion."))
            else:
                st.error(f"Upload failed: {res.text}")
        except Exception as e:
            st.error(f"Upload error: {str(e)}")

    if st.button("Refresh Document Status", use_container_width=True):
        st.rerun()

    docs = fetch_knowledge_documents()
    if docs:
        with st.expander(f"Indexed Documents ({len(docs)})", expanded=False):
            for d in docs:
                st.markdown(
                    f"**{d['source_file']}**  \n"
                    f"Status: `{d['status']}` | Chunks: `{d['chunk_count']}`"
                )
    else:
        st.caption("No documents indexed yet.")

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
        - **`web-search`** *(Tavily Web Search Fallback)*
        """
    )


# ==========================================
# MAIN PANEL: CHAT, CITATIONS & MCP TRACES
# ==========================================
header_col1, header_col2 = st.columns([1, 18], vertical_alignment="center")
with header_col1:
    if Path(BOT_ICON_PATH).exists():
        st.image(BOT_ICON_PATH, width=52)
with header_col2:
    st.title("Personal AI Engineer Assistant")

st.caption(
    "Ask about system architecture notes, inspect local Git status, or manage engineering tasks."
)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=get_avatar(msg["role"])):
        if msg.get("tool_traces"):
            for trace in msg["tool_traces"]:
                with st.expander(
                    f"[MCP Tool Executed: {trace['tool_name']}]", expanded=False
                ):
                    st.code(trace["output"], language="text")

        st.markdown(msg["content"])

        # Render structured source citations (Knowledge Base + Web Search)
        if msg.get("citations"):
            kb_citations = [c for c in msg["citations"] if c.get("source_file")]
            web_citations = [c for c in msg["citations"] if c.get("source_type") == "web"]

            if kb_citations:
                st.markdown("**Cited Knowledge Sources:**")
                for idx, cit in enumerate(kb_citations, 1):
                    page_label = f" | Page {cit['page']}" if cit.get("page") else ""
                    header = (
                        f"[{idx}] {cit['source_file']} "
                        f"(Type: {cit['doc_type']}{page_label} | Score: {cit['score']})"
                    )
                    with st.expander(header, expanded=False):
                        st.text(cit["snippet"])

            if web_citations:
                st.markdown("**Web Search Results:**")
                for idx, cit in enumerate(web_citations, 1):
                    header = (
                        f"[{idx}] {cit['title']} (Score: {cit['score']:.2f})"
                    )
                    with st.expander(header, expanded=False):
                        st.markdown(f"[{cit['url']}]({cit['url']})")
                        st.text(cit["snippet"])

        if msg.get("latency"):
            st.caption(f"Latency: {msg['latency']}s")


# ==========================================
# HUMAN-IN-THE-LOOP (HITL) APPROVAL CARD
# ==========================================
if st.session_state.pending_interrupt:
    interrupt_data = st.session_state.pending_interrupt
    sensitive_calls = interrupt_data.get("sensitive_calls", [])

    st.warning("Security Gatekeeper: Human-in-the-Loop Approval Required")

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
                        "citations": res.get("citations", []),
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

    with st.chat_message("assistant", avatar=get_avatar("assistant")):
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
                citations = res.get("citations", [])
                latency = res.get("latency_sec")

                if status == "interrupted_for_approval":
                    if traces:
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": "*Paused execution waiting for human confirmation on state-mutating tool...*",
                                "tool_traces": traces,
                                "citations": citations,
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
                            "citations": citations,
                            "latency": latency,
                        }
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Error communicating with Orchestrator API: {str(e)}")