import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import re
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Annotated, TypedDict, List, Literal, Optional, Dict, Any
from dotenv import load_dotenv, dotenv_values

import psycopg
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from memory_store import LongTermMemoryManager, extract_clean_text
from cache_manager import RedisCacheManager
from thread_store import ThreadMetadataStore

# Load load .env file
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Import DocumentIngestionWorker for background file uploads
sys.path.append(str(ROOT_DIR / "apps" / "background-worker"))
from ingest_worker import DocumentIngestionWorker, KNOWLEDGE_DIR  # noqa: E402

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "")
DB_URI = os.getenv(
    "POSTGRES_URI", "postgresql://admin:secretpassword@localhost:5432/PaPa_db"
)

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}

SENSITIVE_MUTATE_TOOLS = {
    "create_personal_task",
    "complete_personal_task",
    "save_memory_note",
}

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

BASE_SYSTEM_PROMPT = """You are a Personal AI Engineer OS Assistant powered by MCP and Agentic RAG.
You have access to three specialized subsystems:
1. Second Brain (Milvus Vector DB): Use `search_knowledge` to retrieve past notes, slides, and architecture decisions, and `save_memory_note` to store new knowledge.
2. Dev & Life Workspace (PostgreSQL & Git): Use `list_pending_tasks`, `create_personal_task`, and `complete_personal_task` to manage tasks, and `list_workspace_git_repos`, `inspect_local_git_status`, `get_local_git_diff`, `get_recent_git_commits`, and `set_workspace_dir` to inspect local git repositories and dynamically change the active workspace directory at runtime without server restart.
3. Web Search (Tavily API): Use `search_web` to find real-time information from the internet when the knowledge base has no relevant results.

=== USER LONG-TERM PROFILE ===
{user_profile}
==============================

KNOWLEDGE-FIRST RETRIEVAL & WEB SEARCH FALLBACK RULES:
1. Priority Lookup: For any informational, technical, architectural, project-related, or factual question (excluding direct task management, direct git inspection commands like git status, git diff, or git log, or direct repository security audit commands), you MUST call `search_knowledge` and include inline citations at the end of relevant sentences in the exact format FIRST before answering. Direct git inspection commands (`list_workspace_git_repos`, `inspect_local_git_status`, `get_local_git_diff`, `get_recent_git_commits`, `set_workspace_dir`) should call these workspace tools directly without calling `search_knowledge` first.
2. Workspace Git Inspection Rules:
   - All git inspection tools (`inspect_local_git_status`, `get_local_git_diff`, `get_recent_git_commits`) accept an optional `repo_path` parameter (either a repository name inside `WORKSPACE_DIR` or a direct filesystem path). If omitted, they default to the current repository.
   - If the user asks about a project or repository's git status, diff, or commits and the exact repo name or path is unclear, call `list_workspace_git_repos` first to discover available repositories.
   - If the user asks to change, switch, or set the workspace directory, use `set_workspace_dir` directly.
3. Grounded Answer (When Found in Knowledge Base): If `search_knowledge` returns relevant context that answers the user's question, prioritize that information and cite the exact source file inline using the format: `` (for example: ``).
4. Web Search Fallback (When Not Found in Knowledge Base): If `search_knowledge` returns "No query matched in Second Brain." or retrieved chunks do not adequately answer the question, you MUST call `search_web` to find real-time information from the internet BEFORE answering from general knowledge. When citing web results, use the format: `[Source Title](URL)` inline.
5. General Knowledge Last Resort: Only if BOTH `search_knowledge` AND `search_web` fail to return useful results, answer using your own general knowledge with a disclaimer: "Note: No results found in knowledge base or web search. Answering from general knowledge."
6. Save Useful Web Findings: If `search_web` returns highly relevant results that the user would benefit from keeping, proactively suggest using `save_memory_note` to store a summary in the knowledge base for future reference. Let the user decide via the approval flow.
7. Keep your responses concise, structured, and technical. Never fabricate source filenames or URLs.
8. Repository Security Audit Rules: For repository security audit requests (local repo name or remote Git URL), use the security audit toolchain in order: prepare_repo_for_security_audit -> analyze_repo_workflow_and_packages -> scan_repo_static_malware -> run_repo_sandbox_probe -> generate_security_audit_report (or call generate_security_audit_report(audit_id, auto_run_missing_stages=True) to complete remaining stages from a checkpoint). Never call search_knowledge for direct repository security audit commands.

Grounding Rules:
- Personalize your technical responses using the User Long-Term Profile above.
"""

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

# ==========================================
# FASTAPI LIFESPAN & GRAPH INITIALIZATION
# ==========================================

def _parse_citations_from_tool_output(output_text: str) -> List[Dict[str, Any]]:
    """Parse structured source citations from search_knowledge tool output."""
    citations = []
    if (
        not output_text
        or "No relevant documents found" in output_text
        or "No query matched in Second Brain." in output_text
    ):
        return citations

    blocks = output_text.split("\n\n---\n\n")
    pattern = re.compile(
        r"\[\d+\]\s*Source:\s*(?P<source>[^\s(]+)\s*\(Type:\s*(?P<doc_type>[^|)]+?)(?:\s*\|\s*|\s+)Score:\s*(?P<score>[\d.]+)\)\s*\n(?:Content|Text):\s*(?P<content>.*)",
        re.DOTALL,
    )

    for block in blocks:
        match = pattern.search(block.strip())
        if match:
            content = match.group("content").strip()
            page_match = re.match(r"^\[Page\s+(\d+)\]\s*(.*)", content, re.DOTALL)
            page_num = int(page_match.group(1)) if page_match else None
            clean_snippet = page_match.group(2).strip() if page_match else content

            citations.append(
                {
                    "source_file": match.group("source").strip(),
                    "doc_type": match.group("doc_type").strip(),
                    "score": float(match.group("score")),
                    "page": page_num,
                    "snippet": clean_snippet,
                }
            )
    return citations


def _parse_web_citations_from_tool_output(output_text: str) -> List[Dict[str, Any]]:
    """Parse structured web citations from search_web tool output."""
    citations = []
    if (
        not output_text
        or "No relevant web results found" in output_text
        or "Web search failed" in output_text
    ):
        return citations

    blocks = output_text.split("\n\n---\n\n")
    pattern = re.compile(
        r"\[\d+\]\s*Web:\s*(?P<title>.+?)\s*\|\s*URL:\s*(?P<url>\S+)\s*"
        r"\(Score:\s*(?P<score>[\d.]+)\)\s*\n"
        r"Snippet:\s*(?P<content>.*)",
        re.DOTALL,
    )

    for block in blocks:
        match = pattern.search(block.strip())
        if match:
            citations.append(
                {
                    "source_type": "web",
                    "title": match.group("title").strip(),
                    "url": match.group("url").strip(),
                    "score": float(match.group("score")),
                    "snippet": match.group("content").strip(),
                }
            )
    return citations

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Maintain persistent MCP sessions and DB connections across the FastAPI lifecycle."""
    rag_server_path = str(ROOT_DIR / "mcp-servers/second-brain-rag/server.py")
    workspace_server_path = str(ROOT_DIR / "mcp-servers/dev-life-workspace/server.py")
    web_search_server_path = str(ROOT_DIR / "mcp-servers/web-search/server.py")

    mcp_config = {
        "second_brain": {
            "command": "python",
            "args": [rag_server_path],
            "transport": "stdio",
            "env": {**os.environ, "GEMINI_API_KEY": GEMINI_API_KEY},
        },
        "dev_workspace": {
            "command": "python",
            "args": [workspace_server_path],
            "transport": "stdio",
            "env": {
                **os.environ,
                "POSTGRES_URI": DB_URI,
                "WORKSPACE_DIR": WORKSPACE_DIR,
            },
        },
    }

    # Only register web_search MCP server if TAVILY_API_KEY is configured
    if TAVILY_API_KEY:
        mcp_config["web_search"] = {
            "command": "python",
            "args": [web_search_server_path],
            "transport": "stdio",
            "env": {**os.environ, "TAVILY_API_KEY": TAVILY_API_KEY},
        }
    else:
        print(
            "[!] TAVILY_API_KEY not set. Web Search MCP server is DISABLED. "
            "Add TAVILY_API_KEY to .env to enable web search fallback."
        )

    mcp_client = MultiServerMCPClient(mcp_config)

    memory_manager = LongTermMemoryManager(db_uri=DB_URI, api_key=GEMINI_API_KEY)
    ingestion_worker = DocumentIngestionWorker()

    async with AsyncExitStack() as stack:
        rag_session = await stack.enter_async_context(mcp_client.session("second_brain"))
        ws_session = await stack.enter_async_context(mcp_client.session("dev_workspace"))

        all_mcp_tools = (await load_mcp_tools(rag_session)) + (
            await load_mcp_tools(ws_session)
        )

        # Load web search tools if TAVILY_API_KEY is available
        web_search_enabled = False
        if TAVILY_API_KEY:
            web_session = await stack.enter_async_context(
                mcp_client.session("web_search")
            )
            web_tools = await load_mcp_tools(web_session)
            all_mcp_tools += web_tools
            web_search_enabled = True

        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(DB_URI)
        )
        await checkpointer.setup()

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=GEMINI_API_KEY,
            # temperature=0.0,
            # transport="rest",
            timeout=15.0,
            max_retries=2,
        ).bind_tools(all_mcp_tools)

        async def planner_node(state: AgentState):
            profile_str = memory_manager.get_user_profile_context()
            sys_msg = SystemMessage(
                content=SYSTEM_PROMPT.format(user_profile=profile_str)
            )
            response = await llm.ainvoke([sys_msg] + state["messages"])
            return {"messages": [response]}

        async def safety_gatekeeper_node(
            state: AgentState,
        ) -> Command[Literal["mcp_tools", "planner"]]:
            last_ai_msg = state["messages"][-1]
            sensitive_calls = [
                tc
                for tc in last_ai_msg.tool_calls
                if tc["name"] in SENSITIVE_MUTATE_TOOLS
            ]

            if not sensitive_calls:
                return Command(goto="mcp_tools")

            decision = interrupt(
                {
                    "warning": "State-mutating tool call requires human approval.",
                    "sensitive_calls": sensitive_calls,
                }
            )

            action = decision.get("action", "reject")
            if action == "approve":
                return Command(goto="mcp_tools")
            elif action == "edit":
                new_args = decision.get("new_args", {})
                for tc in last_ai_msg.tool_calls:
                    if tc["name"] in SENSITIVE_MUTATE_TOOLS:
                        tc["args"].update(new_args)
                return Command(goto="mcp_tools")
            else:
                cancel_messages = [
                    ToolMessage(
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        content="Action was rejected by the user.",
                    )
                    for tc in last_ai_msg.tool_calls
                ]
                return Command(goto="planner", update={"messages": cancel_messages})

        def route_after_planner(
            state: AgentState,
        ) -> Literal["safety_gatekeeper", "__end__"]:
            last_msg = state["messages"][-1]
            if getattr(last_msg, "tool_calls", None):
                return "safety_gatekeeper"
            return END

        builder = StateGraph(AgentState)
        builder.add_node("planner", planner_node)
        builder.add_node("safety_gatekeeper", safety_gatekeeper_node)
        builder.add_node("mcp_tools", ToolNode(all_mcp_tools))

        builder.add_edge(START, "planner")
        builder.add_conditional_edges("planner", route_after_planner)
        builder.add_edge("mcp_tools", "planner")

        app.state.graph = builder.compile(checkpointer=checkpointer)
        app.state.memory_manager = memory_manager
        app.state.ingestion_worker = ingestion_worker
        app.state.cache_manager = RedisCacheManager()
        app.state.thread_store = ThreadMetadataStore(db_uri=DB_URI)
        app.state.web_search_enabled = web_search_enabled
        web_status = "ENABLED" if web_search_enabled else "DISABLED (no TAVILY_API_KEY)"
        print(
            f"[*] FastAPI Orchestrator ready with {len(all_mcp_tools)} MCP tools. "
            f"Web Search: {web_status}"
        )
        yield


app = FastAPI(title="Personal AI Engineer OS API", lifespan=lifespan)


# ==========================================
# API SCHEMAS & ENDPOINTS
# ==========================================


class ChatRequest(BaseModel):
    thread_id: str = Field("default_session", description="Conversation thread ID")
    message: str = Field(..., description="User input message")


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="Conversation thread ID currently interrupted")
    action: Literal["approve", "edit", "reject"] = Field(
        ..., description="Human decision on the pending tool call"
    )
    new_args: Optional[Dict[str, Any]] = Field(
        None, description="Modified tool arguments when action is 'edit'"
    )


FALLBACK_INDICATORS = (
    "not found in your knowledge base",
    "not in your knowledge base",
    "not found in knowledge base",
    "no results found in knowledge base",
    "no relevant information in your knowledge base",
    "no relevant information",
    "was not found in the knowledge base",
    "does not contain information about",
    "not documented in your knowledge base",
    "not documented in",
    "not mentioned in your knowledge base",
    "not found in second brain",
    "answering from general knowledge",
    "general knowledge",
)


def _is_fallback_response(text: str) -> bool:
    """Check if the assistant response indicates fallback to general knowledge."""
    if not text:
        return False
    lower = text.lower()
    if any(p in lower for p in FALLBACK_INDICATORS):
        return True
    if "note:" in lower and "knowledge base" in lower:
        return True
    return False


async def _inspect_graph_response(graph, memory_manager, config: dict) -> dict:
    """Check if the graph completed normally or paused at a Human-in-the-Loop interrupt."""
    state = await graph.aget_state(config)
    messages = state.values.get("messages", [])

    tool_traces = []
    citations = []
    for m in reversed(messages):
        if m.type == "human":
            break
        if m.type == "tool":
            clean_out = extract_clean_text(m.content)
            tool_traces.append(
                {
                    "tool_name": m.name,
                    "output": clean_out,
                }
            )
            if m.name == "search_knowledge":
                citations.extend(_parse_citations_from_tool_output(clean_out))
            elif m.name == "search_web":
                citations.extend(_parse_web_citations_from_tool_output(clean_out))

    tool_traces.reverse()
    citations.reverse()

    final_text = extract_clean_text(messages[-1].content) if messages else ""
    final_lower = final_text.lower()

    has_web_search = any(
        t.get("tool_name") == "search_web" for t in tool_traces
    )

    is_kb_miss = _is_fallback_response(final_text)

    # If fallback indicators are detected, forcefully clear citations
    # to guarantee zero false citations are returned when falling back.
    if is_kb_miss:
        citations = []
    else:
        filtered_citations = []
        for cit in citations:
            if cit.get("source_type") == "web":
                filtered_citations.append(cit)
            elif cit.get("source_file"):
                file_name = Path(cit["source_file"]).name.lower()
                if has_web_search:
                    if file_name in final_lower or f"<{file_name}>" in final_lower:
                        filtered_citations.append(cit)
                else:
                    filtered_citations.append(cit)
        citations = filtered_citations

    if state.next and "safety_gatekeeper" in state.next:
        interrupt_val = state.tasks[0].interrupts[0].value
        return {
            "status": "interrupted_for_approval",
            "thread_id": config["configurable"]["thread_id"],
            "pending_approval": interrupt_val,
            "tool_traces": tool_traces,
            "citations": citations,
        }

    if messages:
        asyncio.create_task(memory_manager.extract_and_store_async(messages))

    return {
        "status": "completed",
        "thread_id": config["configurable"]["thread_id"],
        "response": final_text,
        "tool_traces": tool_traces,
        "citations": citations,
    }


def _handle_potential_rate_limit_checkpoint(error_str: str) -> Optional[Dict[str, Any]]:
    """Check for an in_progress security audit run and pause it due to API rate limiting."""
    try:
        with psycopg.connect(DB_URI) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    """
                    SELECT * FROM security_audit_runs
                    WHERE status = 'in_progress'
                    ORDER BY updated_at DESC
                    LIMIT 1;
                    """
                )
                row = cur.fetchone()
                if not row:
                    return None
                audit_id = row["audit_id"]
                cur.execute(
                    """
                    UPDATE security_audit_runs
                    SET status = 'rate_limited_paused',
                        error_message = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE audit_id = %s
                    RETURNING *;
                    """,
                    (error_str[:500], audit_id),
                )
                updated_row = cur.fetchone()
            conn.commit()
            return dict(updated_row) if updated_row else None
    except Exception as e:
        print(f"[!] Error handling rate limit checkpoint: {e}")
        return None


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    graph = app.state.graph
    cache_mgr: RedisCacheManager = app.state.cache_manager
    config = {"configurable": {"thread_id": req.thread_id}}

    # Skip cache when the thread has a pending HITL interrupt
    state = await graph.aget_state(config)
    has_pending_interrupt = bool(state.next)

    # Try cache first (only for standard queries without pending interrupt)
    if not has_pending_interrupt:
        cached = cache_mgr.get_cached_response(req.message)
        if cached is not None:
            # Guarantee zero citations if cached response indicates fallback
            if _is_fallback_response(cached.get("response", "")):
                cached["citations"] = []
            cached["status"] = "completed"
            cached["thread_id"] = req.thread_id
            cached["latency_sec"] = 0.0
            return cached

    t0 = time.perf_counter()
    try:
        await graph.ainvoke({"messages": [HumanMessage(content=req.message)]}, config)
        result = await _inspect_graph_response(graph, app.state.memory_manager, config)
    except Exception as e:
        err_msg = str(e)
        if any(kw in err_msg.lower() for kw in ["429", "resource_exhausted", "quota"]):
            paused_run = _handle_potential_rate_limit_checkpoint(err_msg)
            audit_id = paused_run.get("audit_id") if paused_run else None
            current_stage = paused_run.get("current_stage") if paused_run else None
            return {
                "status": "completed",
                "thread_id": req.thread_id,
                "response": (
                    f"Note: Gemini API rate limit reached. Security audit progress has been safely "
                    f"saved at checkpoint [audit_id={audit_id}, stage={current_stage}]. "
                    f"You can resume from this checkpoint without losing progress."
                ),
                "checkpoint_paused": True,
                "audit_id": audit_id,
                "current_stage": current_stage,
                "tool_traces": [],
                "citations": [],
                "latency_sec": round(time.perf_counter() - t0, 2),
            }
        raise

    result["latency_sec"] = round(time.perf_counter() - t0, 2)

    # Guarantee zero citations on fallback before caching or returning
    if _is_fallback_response(result.get("response", "")):
        result["citations"] = []

    # Cache completed responses only when no sensitive mutate tools were triggered
    if result.get("status") == "completed":
        triggered_tools = {t.get("tool_name") for t in result.get("tool_traces", [])}
        if not triggered_tools & SENSITIVE_MUTATE_TOOLS:
            cache_mgr.store_response(
                query=req.message,
                response=result.get("response", ""),
                tool_traces=result.get("tool_traces", []),
                citations=result.get("citations", []),
            )

    # Update thread metadata for conversation history listing
    preview = req.message[:120]
    thread_store: ThreadMetadataStore = app.state.thread_store
    thread_store.upsert_thread(req.thread_id, preview)

    return result


@app.get("/threads")
async def list_threads():
    """Return a list of active conversation thread_ids with their latest message preview."""
    thread_store: ThreadMetadataStore = app.state.thread_store
    threads = thread_store.list_threads(limit=50)
    return {"threads": threads}


@app.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    """Read the full conversation history for a given thread from LangGraph state."""
    graph = app.state.graph
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    messages = state.values.get("messages", [])

    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No messages found for thread_id '{thread_id}'.",
        )

    formatted = []
    for m in messages:
        entry: Dict[str, Any] = {
            "role": m.type,
            "content": extract_clean_text(m.content),
        }

        if m.type == "tool":
            clean_out = extract_clean_text(m.content)
            tool_trace = {"tool_name": getattr(m, "name", ""), "output": clean_out}
            citations = []
            if getattr(m, "name", "") == "search_knowledge":
                citations = _parse_citations_from_tool_output(clean_out)
            elif getattr(m, "name", "") == "search_web":
                citations = _parse_web_citations_from_tool_output(clean_out)
            entry["tool_traces"] = [tool_trace]
            entry["citations"] = citations

        formatted.append(entry)

    return {"thread_id": thread_id, "messages": formatted}


@app.post("/chat/resume")
async def resume_endpoint(req: ResumeRequest):
    graph = app.state.graph
    config = {"configurable": {"thread_id": req.thread_id}}

    state = await graph.aget_state(config)
    if not state.next:
        raise HTTPException(
            status_code=400, detail="No pending interrupt found for this thread_id."
        )

    t0 = time.perf_counter()
    resume_cmd = Command(resume={"action": req.action, "new_args": req.new_args or {}})
    await graph.ainvoke(resume_cmd, config)
    result = await _inspect_graph_response(graph, app.state.memory_manager, config)
    result["latency_sec"] = round(time.perf_counter() - t0, 2)
    return result


@app.get("/memories")
async def get_memories():
    return {
        "user_profile": app.state.memory_manager.get_user_profile_context(limit=50),
        "memories": app.state.memory_manager.get_all_memories(limit=50),
    }


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int):
    """Delete a specific long-term memory entry by its ID."""
    deleted = app.state.memory_manager.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory with id={memory_id} not found.")
    return {"status": "deleted", "memory_id": memory_id}


@app.delete("/knowledge/documents/{filename:path}")
async def delete_knowledge_document(filename: str):
    """Delete a knowledge document: remove vectors from Milvus, registry row from PostgreSQL, and physical file."""
    worker: DocumentIngestionWorker = app.state.ingestion_worker

    # (a) Delete vectors in Milvus
    worker.store.delete_by_source(filename)

    # (b) Delete registry record from PostgreSQL
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_registry WHERE source_file = %s",
                (filename,),
            )
            db_deleted = cur.rowcount > 0
        conn.commit()

    # (c) Remove the physical file if it exists
    file_path = KNOWLEDGE_DIR / filename
    file_removed = False
    if file_path.exists():
        file_path.unlink()
        file_removed = True

    if not db_deleted and not file_removed:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found in registry or on disk.",
        )

    # Invalidate chat cache after knowledge base change
    app.state.cache_manager.invalidate_all_chat_cache()

    return {
        "status": "deleted",
        "filename": filename,
        "milvus_cleared": True,
        "db_deleted": db_deleted,
        "file_removed": file_removed,
    }

@app.post("/knowledge/upload", status_code=202)
async def upload_knowledge_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """Save uploaded documents to knowledge_base/ and queue background ingestion."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    worker: DocumentIngestionWorker = app.state.ingestion_worker
    queued_files = []

    for upload in files:
        filename = Path(upload.filename).name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension '{suffix}' for '{filename}'. Allowed: .pdf, .md, .txt",
            )

        dest_path = KNOWLEDGE_DIR / filename
        content = await upload.read()
        dest_path.write_bytes(content)

        doc_type = "pdf" if suffix == ".pdf" else "architecture"
        file_hash = hashlib.sha256(content).hexdigest()
        if worker.is_file_unchanged(filename, file_hash):
            worker.mark_status(
                filename,
                status="skipped_unchanged",
                file_hash=file_hash,
                doc_type=doc_type,
            )
            continue

        worker.mark_status(filename, status="processing", file_hash=file_hash, doc_type=doc_type)
        background_tasks.add_task(worker.ingest_single_file, dest_path)
        queued_files.append(filename)

    # Invalidate chat cache after knowledge base change
    if queued_files:
        app.state.cache_manager.invalidate_all_chat_cache()

    return {
        "status": "accepted",
        "message": (
            f"Queued {len(queued_files)} file(s) for background ingestion."
            if queued_files
            else "Uploaded file(s) are already indexed and unchanged (skipped)."
        ),
        "files": queued_files,
    }


@app.get("/knowledge/documents")
async def list_knowledge_documents():
    """Return all documents in document_registry and their ingestion status."""
    worker: DocumentIngestionWorker = app.state.ingestion_worker
    return {"documents": worker.list_registered_documents()}


class WorkspaceRequest(BaseModel):
    workspace_dir: str = Field(..., description="New workspace directory path")


def _update_env_file(key: str, value: str):
    """Safely update or insert a key in the .env file without modifying other keys."""
    env_path = ROOT_DIR / ".env"
    lines = []
    found = False
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{key}=") or stripped == key:
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.get("/workspace")
async def get_workspace():
    """Return the currently configured workspace directory without restarting."""
    env_path = ROOT_DIR / ".env"
    current = ""
    if env_path.exists():
        vals = dotenv_values(env_path)
        current = vals.get("WORKSPACE_DIR") or ""
    if not current:
        current = os.getenv("WORKSPACE_DIR", "")
    return {"workspace_dir": current or str(ROOT_DIR.parent.resolve())}


@app.post("/workspace")
async def update_workspace(req: WorkspaceRequest):
    """Update the workspace directory configuration at runtime without restarting the server."""
    cleaned = req.workspace_dir.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Workspace directory path cannot be empty.")

    for p in cleaned.split(","):
        target = Path(os.path.expanduser(p.strip())).resolve()
        if not target.is_dir():
            raise HTTPException(status_code=400, detail=f"Directory does not exist: {target}")

    os.environ["WORKSPACE_DIR"] = cleaned
    _update_env_file("WORKSPACE_DIR", cleaned)
    return {"status": "updated", "workspace_dir": cleaned}


# ==========================================
# REPOSITORY SECURITY AUDIT ENDPOINTS
# ==========================================

workspace_pkg_path = str(ROOT_DIR / "mcp-servers" / "dev-life-workspace")
if workspace_pkg_path not in sys.path:
    sys.path.insert(0, workspace_pkg_path)


@app.get("/security/audits")
async def list_security_audits(limit: int = 20):
    """Retrieve recent repository security audit runs ordered by latest update."""
    try:
        with psycopg.connect(DB_URI) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    """
                    SELECT * FROM security_audit_runs
                    ORDER BY updated_at DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()

        audits = []
        for r in rows:
            d = dict(r)
            if "created_at" in d and hasattr(d["created_at"], "isoformat"):
                d["created_at"] = d["created_at"].isoformat()
            if "updated_at" in d and hasattr(d["updated_at"], "isoformat"):
                d["updated_at"] = d["updated_at"].isoformat()
            audits.append(d)

        return {"audits": audits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query audit runs: {str(e)}")


@app.get("/security/audits/{audit_id}")
async def get_security_audit(audit_id: str):
    """Retrieve full details of a specific security audit run by audit_id."""
    try:
        with psycopg.connect(DB_URI) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    "SELECT * FROM security_audit_runs WHERE audit_id = %s;",
                    (audit_id,),
                )
                row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Security audit run '{audit_id}' not found.",
            )

        d = dict(row)
        if "created_at" in d and hasattr(d["created_at"], "isoformat"):
            d["created_at"] = d["created_at"].isoformat()
        if "updated_at" in d and hasattr(d["updated_at"], "isoformat"):
            d["updated_at"] = d["updated_at"].isoformat()
        return d
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch audit: {str(e)}")


@app.get("/security/audits/{audit_id}/report")
async def get_security_audit_report(audit_id: str):
    """Read and return generated Markdown report content from disk for a completed audit."""
    try:
        with psycopg.connect(DB_URI) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    "SELECT audit_id, report_path FROM security_audit_runs WHERE audit_id = %s;",
                    (audit_id,),
                )
                row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Security audit run '{audit_id}' not found.",
            )

        report_path_str = row.get("report_path")
        if not report_path_str:
            raise HTTPException(
                status_code=404,
                detail=f"No report has been generated yet for audit '{audit_id}'.",
            )

        report_file = Path(report_path_str)
        if not report_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Report file not found on disk at '{report_path_str}'.",
            )

        content = report_file.read_text(encoding="utf-8")
        return {
            "audit_id": audit_id,
            "report_path": report_path_str,
            "markdown": content,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read audit report: {str(e)}")


@app.post("/security/audits/{audit_id}/resume")
async def resume_security_audit(audit_id: str):
    """Deterministically complete remaining audit stages and generate final report."""
    try:
        from security_scanner.report_generator import SecurityAuditReportGenerator

        generator = SecurityAuditReportGenerator(db_uri=DB_URI)
        result = generator.generate_report(audit_id=audit_id, auto_run_missing_stages=True)
        record = generator.state_mgr.get_audit_run(audit_id)

        if record:
            if "created_at" in record and hasattr(record["created_at"], "isoformat"):
                record["created_at"] = record["created_at"].isoformat()
            if "updated_at" in record and hasattr(record["updated_at"], "isoformat"):
                record["updated_at"] = record["updated_at"].isoformat()

        return {
            "status": "completed",
            "audit_id": audit_id,
            "report_path": result.get("report_path"),
            "final_verdict": result.get("final_verdict"),
            "audit_record": record,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume security audit '{audit_id}': {str(e)}",
        )


_WS_SERVER_MODULE = None


def _get_workspace_server_module():
    global _WS_SERVER_MODULE
    if _WS_SERVER_MODULE is None:
        ws_server_path = ROOT_DIR / "mcp-servers" / "dev-life-workspace" / "server.py"
        spec = importlib.util.spec_from_file_location("dev_life_workspace_server", str(ws_server_path))
        _WS_SERVER_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_WS_SERVER_MODULE)
    return _WS_SERVER_MODULE


class StartAuditRequest(BaseModel):
    repo_source: str
    force_new: bool = False


@app.post("/security/audits/start")
async def start_security_audit(req: StartAuditRequest):
    """Start a full deterministic security scan and report generation directly."""
    try:
        from security_scanner.report_generator import SecurityAuditReportGenerator

        ws_mod = _get_workspace_server_module()
        prep_raw = ws_mod.prepare_repo_for_security_audit(
            repo_source=req.repo_source,
            force_new=req.force_new,
        )
        prep_data = json.loads(prep_raw)
        if "error" in prep_data:
            raise HTTPException(status_code=400, detail=prep_data["error"])

        audit_id = prep_data["audit_id"]

        generator = SecurityAuditReportGenerator(db_uri=DB_URI)
        report_res = generator.generate_report(audit_id=audit_id, auto_run_missing_stages=True)
        record = generator.state_mgr.get_audit_run(audit_id)

        if record:
            if "created_at" in record and hasattr(record["created_at"], "isoformat"):
                record["created_at"] = record["created_at"].isoformat()
            if "updated_at" in record and hasattr(record["updated_at"], "isoformat"):
                record["updated_at"] = record["updated_at"].isoformat()

        static_verdict = record.get("static_risk_verdict") if record else None
        final_verdict = static_verdict if static_verdict == "CRITICAL_DANGER" else report_res.get("final_verdict", static_verdict)

        return {
            "status": "completed",
            "audit_id": audit_id,
            "report_path": report_res.get("report_path"),
            "final_verdict": final_verdict,
            "static_risk_verdict": static_verdict,
            "audit_record": record,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start security audit for '{req.repo_source}': {str(e)}",
        )


@app.delete("/security/audits/{audit_id}")
async def delete_security_audit(audit_id: str):
    """Delete a security audit run record and its generated Markdown report file from disk."""
    try:
        from security_scanner.audit_state import SecurityAuditStateManager

        state_mgr = SecurityAuditStateManager(db_uri=DB_URI)
        record = state_mgr.get_audit_run(audit_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"Security audit run '{audit_id}' not found.",
            )

        report_path_str = record.get("report_path")
        if report_path_str:
            try:
                rp = Path(report_path_str)
                if rp.exists():
                    rp.unlink()
            except Exception:
                pass

        state_mgr.delete_audit_run(audit_id)
        return {"status": "deleted", "audit_id": audit_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete security audit '{audit_id}': {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)