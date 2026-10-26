import asyncio
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Annotated, TypedDict, List, Literal
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# Load load .env file
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Not found GEMINI_API_KEY in .env!")

DB_URI = os.getenv(
    "POSTGRES_URI", "postgresql://admin:secretpassword@localhost:5432/PaPa_db"
)

# Set of state-mutating MCP tools that require explicit Human-in-the-Loop approval
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

Guidelines:
- When a user request spans multiple domains, proactively chain or combine the appropriate tools.
- Be concise, accurate, and always cite the `source_file` and `score` when answering from the Second Brain.
"""

async def run_orchestrator():
    # Absolute path to MCP Server
    rag_server_path = str(ROOT_DIR / "mcp-servers/second-brain-rag/server.py")
    workspace_server_path = str(ROOT_DIR / "mcp-servers/dev-life-workspace/server.py")

    # Initialize MCP Clinet connect to Second Brain Server
    mcp_client = MultiServerMCPClient(
        {
            "second_brain": {
                "command": "python",
                "args": [rag_server_path],
                "transport": "stdio", # user `see` when on Docker
                "env": {
                    **os.environ, # Keep PATH of venv/uv
                    "GEMINI_API_KEY": GEMINI_API_KEY
                }
            },
            "dev_workspace": {
                "command": "python",
                "args": [workspace_server_path],
                "transport": "stdio",
                "env": {**os.environ, "POSTGRES_URI": DB_URI}
            }
        }
    )

    # Maintain persistent sessions for both MCP servers and the PostgreSQL checkpointer
    async with AsyncExitStack() as stack:
        t0 = time.perf_counter()

        rag_session = await stack.enter_async_context(mcp_client.session("second_brain"))
        ws_session = await stack.enter_async_context(mcp_client.session("dev_workspace"))

        rag_tools = await load_mcp_tools(rag_session)
        ws_tools = await load_mcp_tools(ws_session)
        all_mcp_tools = rag_tools + ws_tools

        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(DB_URI)
        )
        await checkpointer.setup()

        print(
            f"[*] Initialized 2 MCP Servers & Postgres Checkpointer in: {time.perf_counter() - t0:.2f}s"
        )
        print(
            f"[*] Loaded {len(all_mcp_tools)} MCP Tools: {[t.name for t in all_mcp_tools]}"
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=GEMINI_API_KEY,
            # temperature=0.0,
            # transport="rest",
            timeout=15.0,
            max_retries=2,
        )
        llm_with_tools = llm.bind_tools(all_mcp_tools)

        # ==========================================
        # LANGGRAPH NODE DEFINITIONS
        # ==========================================

        async def planner_node(state: AgentState):
            step_t0 = time.perf_counter()
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            response = await llm_with_tools.ainvoke(messages)
            print(f"  [LLM Planner] Responded in: {time.perf_counter() - step_t0:.2f}s")
            return {"messages": [response]}

        async def safety_gatekeeper_node(
            state: AgentState,
        ) -> Command[Literal["mcp_tools", "planner"]]:
            """Intercept sensitive mutate tool calls for Human-in-the-Loop confirmation."""
            last_ai_msg = state["messages"][-1]
            sensitive_calls = [
                tc
                for tc in last_ai_msg.tool_calls
                if tc["name"] in SENSITIVE_MUTATE_TOOLS
            ]

            # Auto-execute if all requested tool calls are read-only
            if not sensitive_calls:
                return Command(goto="mcp_tools")

            # Freeze graph execution and request human confirmation
            decision = interrupt(
                {
                    "warning": "State-mutating tool call detected. Human approval required.",
                    "sensitive_calls": sensitive_calls,
                }
            )

            action = decision.get("action", "reject")

            if action == "approve":
                print("  [HITL] User APPROVED tool execution.")
                return Command(goto="mcp_tools")

            elif action == "edit":
                new_args = decision.get("new_args", {})
                for tc in last_ai_msg.tool_calls:
                    if tc["name"] in SENSITIVE_MUTATE_TOOLS:
                        tc["args"].update(new_args)
                print(f"  [HITL] User EDITED tool arguments: {new_args}")
                return Command(goto="mcp_tools")

            else:
                print("  [HITL] User REJECTED tool execution.")
                cancel_messages = [
                    ToolMessage(
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        content="Action was rejected by the user. Do not execute this operation.",
                    )
                    for tc in last_ai_msg.tool_calls
                ]
                return Command(goto="planner", update={"messages": cancel_messages})

        raw_tool_node = ToolNode(all_mcp_tools)

        async def timed_tool_node(state: AgentState):
            step_t0 = time.perf_counter()
            res = await raw_tool_node.ainvoke(state)
            print(f"  [MCP ToolNode] Executed in: {time.perf_counter() - step_t0:.2f}s")
            return res

        def route_after_planner(
            state: AgentState,
        ) -> Literal["safety_gatekeeper", "__end__"]:
            last_msg = state["messages"][-1]
            if getattr(last_msg, "tool_calls", None):
                return "safety_gatekeeper"
            return END

        # Build and compile the StateGraph
        builder = StateGraph(AgentState)
        builder.add_node("planner", planner_node)
        builder.add_node("safety_gatekeeper", safety_gatekeeper_node)
        builder.add_node("mcp_tools", timed_tool_node)

        builder.add_edge(START, "planner")
        builder.add_conditional_edges("planner", route_after_planner)
        builder.add_edge("mcp_tools", "planner")

        graph = builder.compile(checkpointer=checkpointer)

        # ==========================================
        # STAGE 2 END-TO-END TEST (MULTI-SERVER + HITL)
        # ==========================================
        config = {"configurable": {"thread_id": "ben_stage2_en_session_01"}}

        prompt = (
            "Search my Second Brain to see what technologies I use for the video OCR queue, "
            "then create a new high-priority personal task to 'Optimize worker concurrency for that queue tech' "
            "with a due date of 2026-10-30."
        )
        print(f"\n[User]: {prompt}\n")

        # Initial stream: Executes read-only `search_knowledge` automatically,
        # then pauses at `safety_gatekeeper` before executing `create_personal_task`.
        async for event in graph.astream(
            {"messages": [HumanMessage(content=prompt)]}, config, stream_mode="values"
        ):
            last_msg = event["messages"][-1]
            if getattr(last_msg, "tool_calls", None):
                print(
                    f"  [Tool Call Requested]: {[(tc['name'], tc['args']) for tc in last_msg.tool_calls]}"
                )

        # Check if the graph is currently interrupted waiting for human input
        current_state = await graph.aget_state(config)
        if current_state.next and "safety_gatekeeper" in current_state.next:
            interrupt_payload = current_state.tasks[0].interrupts[0].value
            print("\n" + "=" * 65)
            print("[SECURITY GATEKEEPER - HUMAN-IN-THE-LOOP INTERRUPT]")
            print(f"Pending Sensitive Calls: {interrupt_payload['sensitive_calls']}")
            print("=" * 65)

            choice = (
                input(
                    "Enter 'y' to Approve, 'e' to Edit due_date, or 'n' to Reject: "
                )
                .strip()
                .lower()
            )

            if choice == "y":
                resume_cmd = Command(resume={"action": "approve"})
            elif choice == "e":
                new_due = input("Enter new due date (e.g., 2026-11-05): ").strip()
                resume_cmd = Command(
                    resume={"action": "edit", "new_args": {"due_date": new_due}}
                )
            else:
                resume_cmd = Command(resume={"action": "reject"})

            # Resume graph execution from the PostgreSQL checkpoint
            async for event in graph.astream(
                resume_cmd, config, stream_mode="values"
            ):
                last_msg = event["messages"][-1]
                if last_msg.content and not getattr(last_msg, "tool_calls", None):
                    print(f"\n[Assistant]:\n{last_msg.content}")

if __name__ == "__main__":
    asyncio.run(run_orchestrator())