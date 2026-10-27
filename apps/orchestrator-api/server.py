import asyncio
import os
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Annotated, TypedDict, List, Literal, Optional, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_URI = os.getenv(
    "POSTGRES_URI", "postgresql://admin:secretpassword@localhost:5432/PaPa_db"
)

SENSITIVE_MUTATE_TOOLS = {
    "create_personal_task",
    "complete_personal_task",
    "save_memory_note",
}

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

SYSTEM_PROMPT = """You are a Personal AI Engineer OS Assistant powered by MCP and Agentic RAG.
You have access to two specialized subsystems:
1. Second Brain (Milvus Vector DB): Use `search_knowledge` to retrieve past notes, slides, and architecture decisions, and `save_memory_note` to store new knowledge.
2. Dev & Life Workspace (PostgreSQL & Git): Use `list_pending_tasks`, `create_personal_task`, and `complete_personal_task` to manage tasks, and `inspect_local_git_status` to inspect local repositories.

=== USER LONG-TERM PROFILE ===
{user_profile}
==============================

Guidelines:
- Personalize your technical responses using the User Long-Term Profile above.
- Always cite `source_file` and `score` when retrieving from the Second Brain.
"""

# ==========================================
# FASTAPI LIFESPAN & GRAPH INITIALIZATION
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Maintain persistent MCP sessions and DB connections across the FastAPI lifecycle."""
    rag_server_path = str(ROOT_DIR / "mcp-servers/second-brain-rag/server.py")
    workspace_server_path = str(ROOT_DIR / "mcp-servers/dev-life-workspace/server.py")

    mcp_client = MultiServerMCPClient(
        {
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
    )

    memory_manager = LongTermMemoryManager(db_uri=DB_URI, api_key=GEMINI_API_KEY)

    async with AsyncExitStack() as stack:
        rag_session = await stack.enter_async_context(mcp_client.session("second_brain"))
        ws_session = await stack.enter_async_context(mcp_client.session("dev_workspace"))

        all_mcp_tools = (await load_mcp_tools(rag_session)) + (
            await load_mcp_tools(ws_session)
        )

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
        print(
            f"[*] FastAPI Orchestrator ready with {len(all_mcp_tools)} MCP tools."
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
    if state.next and "safety_gatekeeper" in state.next:
        interrupt_val = state.tasks[0].interrupts[0].value
        return {
            "status": "interrupted_for_approval",
            "thread_id": config["configurable"]["thread_id"],
            "pending_approval": interrupt_val,
        }

    messages = state.values.get("messages", [])
    final_text = extract_clean_text(messages[-1].content) if messages else ""

    # Pass full message list so the extractor can see HumanMessage prior to MCP ToolCalls
    if messages:
        asyncio.create_task(memory_manager.extract_and_store_async(messages))

    return {
        "status": "completed",
        "thread_id": config["configurable"]["thread_id"],
        "response": final_text,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)