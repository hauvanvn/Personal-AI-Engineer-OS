import asyncio
import os
import time
from pathlib import Path
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# Load load .env file
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Not found GEMINI_API_KEY in .env!")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

SYSTEM_PROMPT = """You are Personal AI Engineer OS. 
When user ask any information relate to project, system architecture, or personal note:
1. MUST call `search_knowledge` tool before anwsers.
2. When answer, alway cite cleary name of source file (`source_file`) found.
3. If user ask to memory or save information, call `save_memory_note`.
"""

async def run_orchestrator():
    # Absolute path to MCP Server
    mcp_server_path = str(ROOT_DIR / "mcp-servers/second-brain-rag/server.py")

    # Initialize MCP Clinet connect to Second Brain Server
    mcp_client = MultiServerMCPClient(
        {
            "second_brain": {
                "command": "python",
                "args": [mcp_server_path],
                "transport": "stdio", # user `see` when on Docker
                "env": {
                    **os.environ, # Keep PATH of venv/uv
                    "GEMINI_API_KEY": GEMINI_API_KEY
                }
            }
        }
    )

    # Auto get MCP Server Tool list and tranform to LangChain/LangGraph
    t0 = time.perf_counter()
    async with mcp_client.session("second_brain") as session:
        mcp_tools = await load_mcp_tools(session=session)
        print(f"[*] Initialize MCP Session & Connected in: {time.perf_counter() - t0:.2f}s")
        print(f"[*] Successfuly get {len(mcp_tools)} MCP Tools: {[t.name for t in mcp_tools]}")

        # Initialize LLM and bind MCP Tools
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite", 
            google_api_key=GEMINI_API_KEY, 
            # temperature=0.1,
            max_retries=5, # Automatically wait and retry (Exponential Backoff) if a 429 error occurs on the Free Tier
            transport="rest",
            timeout=15.0,
            )
        llm_with_tools = llm.bind_tools(mcp_tools)

        # Define main deduce Node
        async def planner_node(state: AgentState):
            step_t0 = time.perf_counter()
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            response = await llm_with_tools.ainvoke(messages)
            print(f"  [LLM Planer] Response in: {time.perf_counter() - step_t0:.2f}s")
            return {"messages": [response]}

        raw_tool_node = ToolNode(mcp_tools)
        async def time_tool_node(state: AgentState):
            step_t0 = time.perf_counter()
            res = await raw_tool_node.ainvoke(state)
            print(f"  [MCP Tool & Milvus] Executed in: {time.perf_counter() - step_t0:.2f}s")
            return res


        # LangGraph Graph Assembly
        builder = StateGraph(AgentState)
        builder.add_node("planner", planner_node)
        builder.add_node("mcp_tools", time_tool_node)

        builder.add_edge(START, "planner")
        builder.add_conditional_edges(
            "planner",
            tools_condition, # Automatically branch to "mcp_tools" if the LLM decides to call a tool
            {"tools": "mcp_tools", END: END}
        )
        builder.add_edge("mcp_tools", "planner") # After obtaining the results from MCP, return to the Planner to consolidate them

        # Mount a Checkpointer to retain conversation context based on thread_id
        memory = MemorySaver() # In the actual API version, this will be replaced by AsyncPostgresSaver
        graph = builder.compile(checkpointer=memory)

        # End-to-End test
        config = {"configurable": {"thread_id": "ben_dev_session_02"}}

        print("\n-- TEST 1: Ask Assistant save architechture into Second Brain ---")
        start_step1 = time.perf_counter()
        msg1 = "Please add the following content to the architecture notes (file: thesis_backend.md, doc_type: architechture): The thesis system uses NestJS as the primary backend, incorporating BullMQ and Redis to handle the video OCR queue."
        async for event in graph.astream({"messages": [HumanMessage(content=msg1)]}, config, stream_mode="values"):
            event["messages"][-1].pretty_print()
        print(f"-> Total time of TEST 1: {time.perf_counter() - start_step1:.2f}s")
        await asyncio.sleep(2)

        print("\n--- TEST 2: Querying knowledge via RAG MCP ---")
        start_step2 = time.perf_counter()
        msg2 = "Check my `Second Brain` to see what technology I'm using to process the video OCR queue?"
        async for event in graph.astream({"messages": [HumanMessage(content=msg2)]}, config, stream_mode="values"):
            event["messages"][-1].pretty_print()
        print(f"-> Total time of TEST 2: {time.perf_counter() - start_step2:.2f}s")

if __name__ == "__main__":
    asyncio.run(run_orchestrator())