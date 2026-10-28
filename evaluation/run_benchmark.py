import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")
sys.path.append(str(ROOT_DIR / "mcp-servers" / "second-brain-rag"))

from milvus_store import SecondBrainStore  # noqa: E402

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==========================================
# BENCHMARK DATASET (GOLD STANDARD)
# ==========================================
BENCHMARK_CASES: List[Dict[str, Any]] = [
    {
        "id": "eval_01",
        "question": "What technologies are used to handle the video OCR queue in my thesis backend?",
        "expected_tool": "search_knowledge",
        "expected_source": "thesis_backend.md",
        "ground_truth": "NestJS backend using BullMQ and Redis for the video OCR queue.",
    },
    {
        "id": "eval_02",
        "question": "How does the retrieval pipeline weight semantic similarity vs recency?",
        "expected_tool": "search_knowledge",
        "expected_source": "system_design_guidelines.md",
        "ground_truth": "85% semantic similarity and 15% document recency using a 768-dim Matryoshka HNSW index.",
    },
    {
        "id": "eval_03",
        "question": "Show me all my current pending tasks and deadlines.",
        "expected_tool": "list_pending_tasks",
        "expected_source": None,
        "ground_truth": "Lists pending tasks from the PostgreSQL personal_tasks table.",
    },
    {
        "id": "eval_04",
        "question": "Check the current git status and recent commits of my local repository.",
        "expected_tool": "inspect_local_git_status",
        "expected_source": None,
        "ground_truth": "Executes git status and git log on the local workspace.",
    },
]


class JudgeScore(BaseModel):
    faithfulness_score: float = Field(
        ...,
        description="Score from 0.0 to 1.0 indicating if the answer is grounded in the retrieved context without hallucination.",
    )
    relevance_score: float = Field(
        ...,
        description="Score from 0.0 to 1.0 indicating how well the answer addresses the user question.",
    )
    reasoning: str = Field(..., description="Brief 1-sentence justification.")


async def run_evaluation():
    store = SecondBrainStore()
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GEMINI_API_KEY,
        # temperature=0.0,
        # transport="rest",
    ).with_structured_output(JudgeScore)

    import requests

    api_url = "http://localhost:8080/chat"

    retrieval_hits = 0
    retrieval_total = 0
    tool_correct = 0
    faithfulness_scores = []
    latencies = []

    print("=" * 70)
    print("STARTING PERSONAL AI ENGINEER OS BENCHMARK SUITE")
    print("=" * 70)

    for case in BENCHMARK_CASES:
        q = case["question"]
        expected_tool = case["expected_tool"]
        expected_source = case["expected_source"]

        # 1. Evaluate Direct RAG Retrieval Hit Rate @ 3 (if applicable)
        if expected_source:
            retrieval_total += 1
            rag_results = store.search_with_time_decay(query=q, top_k=3)
            retrieved_sources = [r["source_file"] for r in rag_results]
            hit = expected_source in retrieved_sources
            if hit:
                retrieval_hits += 1
        else:
            retrieved_sources = []

        # 2. Evaluate End-to-End Orchestrator Tool Selection & Generation via FastAPI
        t0 = time.perf_counter()
        res = requests.post(
            api_url,
            json={"thread_id": f"bench_{case['id']}", "message": q},
            timeout=30,
        ).json()
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        traces = res.get("tool_traces", [])
        called_tools = [t["tool_name"] for t in traces]
        is_tool_match = expected_tool in called_tools
        if is_tool_match:
            tool_correct += 1

        # 3. Evaluate Faithfulness & Relevance via LLM-as-a-Judge
        context_str = "\n".join([t["output"] for t in traces]) or "No tool context."
        answer_str = res.get("response", "")

        judge_prompt = (
            f"Question: {q}\n"
            f"Retrieved Tool Context:\n{context_str}\n\n"
            f"Assistant Answer:\n{answer_str}\n\n"
            f"Expected Ground Truth:\n{case['ground_truth']}"
        )
        score: JudgeScore = await judge_llm.ainvoke(
            [
                SystemMessage(
                    content="You are an objective AI Evaluation Judge grading RAG and Agent responses."
                ),
                HumanMessage(content=judge_prompt),
            ]
        )
        faithfulness_scores.append(score.faithfulness_score)

        print(
            f"\n[{case['id']}] Q: {q}\n"
            f"  ├─ Tool Called: {called_tools} (Expected: {expected_tool}) -> {'✅ PASS' if is_tool_match else '❌ FAIL'}\n"
            f"  ├─ RAG Sources: {retrieved_sources}\n"
            f"  ├─ Faithfulness: {score.faithfulness_score:.2f} | Relevance: {score.relevance_score:.2f}\n"
            f"  └─ Latency: {elapsed:.2f}s"
        )
        # Brief pause to respect Gemini Free Tier RPM
        await asyncio.sleep(2)

    # ==========================================
    # AGGREGATE METRICS REPORT
    # ==========================================
    hit_rate = (retrieval_hits / retrieval_total * 100) if retrieval_total else 0.0
    tool_acc = (tool_correct / len(BENCHMARK_CASES)) * 100
    avg_faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores) * 100
        if faithfulness_scores
        else 0.0
    )
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    report = {
        "total_test_cases": len(BENCHMARK_CASES),
        "rag_hit_rate_at_3_pct": round(hit_rate, 1),
        "mcp_tool_selection_accuracy_pct": round(tool_acc, 1),
        "avg_faithfulness_score_pct": round(avg_faithfulness, 1),
        "avg_end_to_end_latency_sec": round(avg_latency, 2),
    }

    report_path = ROOT_DIR / "evaluation" / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK SUMMARY (SAVED TO evaluation/benchmark_report.json)")
    print("=" * 70)
    print(f"• RAG Retrieval Hit Rate @3   : {report['rag_hit_rate_at_3_pct']}%")
    print(f"• MCP Tool Selection Accuracy : {report['mcp_tool_selection_accuracy_pct']}%")
    print(f"• Answer Faithfulness Score   : {report['avg_faithfulness_score_pct']}%")
    print(f"• Average E2E Latency         : {report['avg_end_to_end_latency_sec']}s")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_evaluation())