import asyncio
import os
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
    mcp_tools = await mcp_client.get_tools()
    print(f"[*] Successfuly get {len(mcp_tools)} MCP Tools: {[t.name for t in mcp_tools]}")

    # Initialize LLM and bind MCP Tools
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash", 
        google_api_key=GEMINI_API_KEY, 
        temperature=0.1,
        max_retries=5, # Automatically wait and retry (Exponential Backoff) if a 429 error occurs on the Free Tier
        )
    llm_with_tools = llm.bind_tools(mcp_tools)

    # Define main deduce Node
    async def planner_node(state: AgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # LangGraph Graph Assembly
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("mcp_tools", ToolNode(mcp_tools))

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
    config = {"configurable": {"thread_id": "ben_dev_session_01"}}

    print("\n-- TEST 1: Ask Assistant save architechture into Second Brain ---")
    msg1 = "Please add the following content to the architecture notes (file: graduation_arch.md): The thesis system uses NestJS as the primary backend, incorporating BullMQ and Redis to handle the video OCR queue."
    async for event in graph.astream({"messages": [HumanMessage(content=msg1)]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    await asyncio.sleep(2)

    print("\n--- TEST 2: Querying knowledge via RAG MCP ---")
    msg2 = "Check my `Second Brain` to see what technology I'm using to process the video OCR queue?"
    async for event in graph.astream({"messages": [HumanMessage(content=msg2)]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()

if __name__ == "__main__":
    asyncio.run(run_orchestrator())