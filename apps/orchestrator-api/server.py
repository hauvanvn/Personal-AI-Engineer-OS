import asyncio
import os
import sys
import re
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Annotated, TypedDict, List, Literal, Optional, Dict, Any
from dotenv import load_dotenv

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

# Load load .env file
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Import DocumentIngestionWorker for background file uploads
sys.path.append(str(ROOT_DIR / "apps" / "background-worker"))
from ingest_worker import DocumentIngestionWorker, KNOWLEDGE_DIR  # noqa: E402

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
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

SYSTEM_PROMPT = """You are a Personal AI Engineer OS Assistant powered by MCP and Agentic RAG.
You have access to three specialized subsystems:
1. Second Brain (Milvus Vector DB): Use `search_knowledge` to retrieve past notes, slides, and architecture decisions, and `save_memory_note` to store new knowledge.
2. Dev & Life Workspace (PostgreSQL & Git): Use `list_pending_tasks`, `create_personal_task`, and `complete_personal_task` to manage tasks, and `inspect_local_git_status` to inspect local repositories.
3. Web Search (Tavily API): Use `search_web` to find real-time information from the internet when the knowledge base has no relevant results.

=== USER LONG-TERM PROFILE ===
{user_profile}
==============================

KNOWLEDGE-FIRST RETRIEVAL & WEB SEARCH FALLBACK RULES:
1. Priority Lookup: For any informational, technical, architectural, project-related, or factual question (excluding direct task management or local git status commands), you MUST call `search_knowledge` and include inline citations at the end of relevant sentences in the exact format FIRST before answering.
2. Grounded Answer (When Found in Knowledge Base): If `search_knowledge` returns relevant context that answers the user's question, prioritize that information and cite the exact source file inline using the format: `` (for example: ``).
3. Web Search Fallback (When Not Found in Knowledge Base): If `search_knowledge` returns "No query matched in Second Brain." or retrieved chunks do not adequately answer the question, you MUST call `search_web` to find real-time information from the internet BEFORE answering from general knowledge. When citing web results, use the format: `[Source Title](URL)` inline.
4. General Knowledge Last Resort: Only if BOTH `search_knowledge` AND `search_web` fail to return useful results, answer using your own general knowledge with a disclaimer: "Note: No results found in knowledge base or web search. Answering from general knowledge."
5. Save Useful Web Findings: If `search_web` returns highly relevant results that the user would benefit from keeping, proactively suggest using `save_memory_note` to store a summary in the knowledge base for future reference. Let the user decide via the approval flow.
6. Keep your responses concise, structured, and technical. Never fabricate source filenames or URLs.

Grounding Rules:
- Personalize your technical responses using the User Long-Term Profile above.
"""

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
            "env": {**os.environ, "POSTGRES_URI": DB_URI},
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

    kb_miss_phrases = [
        "not found in your knowledge base",
        "not found in knowledge base",
        "no results found in knowledge base",
        "not found in second brain",
        "không tìm thấy trong knowledge base",
        "không có trong knowledge base",
        "không tìm thấy trong cơ sở tri thức",
        "answering from general knowledge",
    ]
    is_kb_miss = any(p in final_lower for p in kb_miss_phrases)

    filtered_citations = []
    for cit in citations:
        if cit.get("source_type") == "web":
            filtered_citations.append(cit)
        elif cit.get("source_file"):
            file_name = Path(cit["source_file"]).name.lower()
            if has_web_search or is_kb_miss:
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


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    graph = app.state.graph
    config = {"configurable": {"thread_id": req.thread_id}}

    t0 = time.perf_counter()
    await graph.ainvoke({"messages": [HumanMessage(content=req.message)]}, config)
    result = await _inspect_graph_response(graph, app.state.memory_manager, config)
    result["latency_sec"] = round(time.perf_counter() - t0, 2)
    return result


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
        "user_profile": app.state.memory_manager.get_user_profile_context(limit=50)
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
        worker.mark_status(filename, status="processing", doc_type=doc_type)
        background_tasks.add_task(worker.ingest_single_file, dest_path)
        queued_files.append(filename)

    return {
        "status": "accepted",
        "message": f"Queued {len(queued_files)} file(s) for background ingestion.",
        "files": queued_files,
    }


@app.get("/knowledge/documents")
async def list_knowledge_documents():
    """Return all documents in document_registry and their ingestion status."""
    worker: DocumentIngestionWorker = app.state.ingestion_worker
    return {"documents": worker.list_registered_documents()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)