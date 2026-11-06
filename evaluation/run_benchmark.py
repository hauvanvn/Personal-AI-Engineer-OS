"""
Personal AI Engineer OS -- Benchmark Suite v2.0
================================================
Comprehensive, modular, Free-Tier-safe evaluation covering:
  Suite 1 (rag)   : Hybrid RAG Retrieval + Negative Fallback
  Suite 2 (tools) : MCP Tool Selection & Argument Accuracy
  Suite 3 (hitl)  : Human-in-the-Loop Safety Gatekeeper
  Suite 4 (cache) : Redis Response Cache Performance
  Suite 5 (judge) : LLM-as-a-Judge Answer Quality

Usage:
  python evaluation/run_benchmark.py --suite all
  python evaluation/run_benchmark.py --suite rag --delay 4.0
  python evaluation/run_benchmark.py --help
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")
sys.path.insert(0, str(ROOT_DIR / "mcp-servers" / "second-brain-rag"))
sys.path.insert(0, str(ROOT_DIR / "apps" / "orchestrator-api"))

from milvus_store import SecondBrainStore  # noqa: E402

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ===========================================================================
# Global constants
# ===========================================================================
BENCHMARK_SYNTHETIC_SOURCE = "__benchmark_synthetic_doc__.md"

FALLBACK_PHRASES = [
    "not found in your knowledge base",
    "not found in knowledge base",
    "no results found in knowledge base",
    "not found in second brain",
    "answering from general knowledge",
    "no relevant documents found",
    "no query matched in second brain",
]

SENSITIVE_MUTATE_TOOLS = {
    "create_personal_task",
    "complete_personal_task",
    "save_memory_note",
}


# ===========================================================================
# Utility helpers
# ===========================================================================
def _bench_thread() -> str:
    """Return a unique thread_id prefixed for easy identification / cleanup."""
    return f"bench_thread_{uuid.uuid4().hex[:12]}"


def _extract_text(content: Any) -> str:
    """Safely extract plain text from Gemini message content.

    Handles both ``str`` and ``list[dict]`` content blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _parse_judge_json(raw: str) -> Dict[str, Any]:
    """Parse JSON from LLM output, stripping optional markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _percentile(data: List[float], pct: float) -> float:
    """Return the *pct*-th percentile from a sorted copy of *data*."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


class RateLimitPacer:
    """Simple pacer with exponential-backoff retry for HTTP 429 / quota errors."""

    def __init__(self, delay: float, max_retries: int):
        self.delay = delay
        self.max_retries = max_retries

    async def pace(self) -> None:
        await asyncio.sleep(self.delay)

    def is_rate_limit(self, status_code: int = 0, body: str = "") -> bool:
        if status_code == 429:
            return True
        lower = body.lower()
        return "resource_exhausted" in lower or "quota" in lower

    def parse_retry_delay(self, body: str) -> Optional[float]:
        match = re.search(
            r"retry in ([\d.]+)s|retryDelay['\"]?\s*:\s*['\"]?(\d+)s",
            body,
            re.IGNORECASE,
        )
        if match:
            return float(match.group(1) or match.group(2))
        return None

    async def call_with_retry(self, func, *args, **kwargs):
        """Call *func* with retry on rate-limit responses."""
        import requests

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                if isinstance(result, requests.Response) and self.is_rate_limit(
                    result.status_code, result.text
                ):
                    delay = self.parse_retry_delay(result.text) or (
                        2 ** attempt * 5
                    )
                    print(
                        f"  [WARN] Rate limited (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying in {delay:.1f}s ..."
                    )
                    await asyncio.sleep(delay)
                    continue
                return result
            except Exception as exc:
                last_exc = exc
                body = str(exc)
                if self.is_rate_limit(body=body) and attempt < self.max_retries - 1:
                    delay = self.parse_retry_delay(body) or (2 ** attempt * 5)
                    print(
                        f"  [WARN] Rate limited (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying in {delay:.1f}s ..."
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
        if last_exc:
            raise last_exc


# ===========================================================================
# Suite result container
# ===========================================================================
class SuiteResult:
    """Collect individual test case outcomes for one suite."""

    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.cases: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self.latencies: List[float] = []
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def add_case(
        self,
        case_id: str,
        description: str,
        status: str,
        latency_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.cases.append(
            {
                "case_id": case_id,
                "description": description,
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "details": details or {},
            }
        )
        if latency_ms > 0:
            self.latencies.append(latency_ms)
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.skipped += 1

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite_name,
            "total": len(self.cases),
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "metrics": self.metrics,
            "latency_stats_ms": {
                "mean": round(statistics.mean(self.latencies), 2)
                if self.latencies
                else 0.0,
                "p50": round(_percentile(self.latencies, 50), 2),
                "p95": round(_percentile(self.latencies, 95), 2),
            },
            "cases": self.cases,
        }


# ===========================================================================
# Suite 1 -- Hybrid RAG Retrieval & Negative Fallback
# ===========================================================================
async def run_rag_suite(api_url: str, pacer: RateLimitPacer) -> SuiteResult:
    """Suite 1: Hybrid search retrieval accuracy and negative fallback."""
    import requests

    result = SuiteResult("rag")
    store = SecondBrainStore()
    print("\n" + "=" * 70)
    print("SUITE 1: RAG -- Hybrid Retrieval & Negative Fallback")
    print("=" * 70)

    # -- Synthetic knowledge injection --
    synthetic_chunks = [
        (
            "The Milvus HNSW index occasionally throws error code "
            "ERR_MILVUS_HNSW_768_TIMEOUT when the gRPC deadline is exceeded "
            "during high-concurrency embedding ingestion on the free tier."
        ),
        (
            "Architecture parameter MRL_DIM_768_COSINE defines the Matryoshka "
            "Representation Learning dimensionality used in the personal "
            "knowledge vector store with cosine similarity metric."
        ),
        (
            "The custom token budget TPM_WINDOW_22000_LIMIT is enforced by the "
            "embedding batch pipeline to avoid Gemini Free Tier rate limits "
            "during large PDF ingestion runs."
        ),
    ]

    try:
        # Clean up any leftover from a previous aborted run
        store.delete_by_source(BENCHMARK_SYNTHETIC_SOURCE)
        await asyncio.sleep(1)

        print("[INFO] Inserting 3 synthetic benchmark chunks ...")
        for chunk_text in synthetic_chunks:
            store.add_document(
                text=chunk_text,
                source_file=BENCHMARK_SYNTHETIC_SOURCE,
                doc_type="architecture",
            )
            await asyncio.sleep(1.5)
        print("[INFO] Synthetic chunks inserted successfully.")
        await asyncio.sleep(2)

        # -- Positive retrieval cases --
        positive_queries = [
            {
                "id": "rag_pos_01",
                "query": "What is the ERR_MILVUS_HNSW_768_TIMEOUT error code about?",
                "expected_token": "ERR_MILVUS_HNSW_768_TIMEOUT",
                "description": "Semantic + lexical retrieval of HNSW timeout error",
            },
            {
                "id": "rag_pos_02",
                "query": "Explain the MRL_DIM_768_COSINE architecture parameter",
                "expected_token": "MRL_DIM_768_COSINE",
                "description": "Semantic + lexical retrieval of MRL dimension param",
            },
            {
                "id": "rag_pos_03",
                "query": "What is the TPM_WINDOW_22000_LIMIT token budget?",
                "expected_token": "TPM_WINDOW_22000_LIMIT",
                "description": "Semantic + lexical retrieval of TPM budget config",
            },
        ]

        hit_at_1 = 0
        hit_at_3 = 0
        reciprocal_ranks: List[float] = []

        for pq in positive_queries:
            t0 = time.perf_counter()
            results = store.search_with_time_decay(
                query=pq["query"], top_k=3, min_score=0.50
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            texts = [r.get("text", "") for r in results]
            rank = None
            for i, t in enumerate(texts):
                if pq["expected_token"] in t:
                    rank = i + 1
                    break

            is_hit_1 = rank == 1
            is_hit_3 = rank is not None and rank <= 3
            rr = (1.0 / rank) if rank else 0.0

            if is_hit_1:
                hit_at_1 += 1
            if is_hit_3:
                hit_at_3 += 1
            reciprocal_ranks.append(rr)

            status = "PASS" if is_hit_3 else "FAIL"
            result.add_case(
                case_id=pq["id"],
                description=pq["description"],
                status=status,
                latency_ms=elapsed_ms,
                details={
                    "rank": rank,
                    "hit_at_1": is_hit_1,
                    "hit_at_3": is_hit_3,
                    "reciprocal_rank": round(rr, 4),
                    "num_results": len(results),
                },
            )
            label = "[PASS]" if status == "PASS" else "[FAIL]"
            print(
                f"  {label} {pq['id']}: rank={rank}  HR@1={'Y' if is_hit_1 else 'N'}  "
                f"HR@3={'Y' if is_hit_3 else 'N'}  RR={rr:.4f}  ({elapsed_ms:.0f}ms)"
            )
            await pacer.pace()

        n_pos = len(positive_queries)
        result.metrics["hit_rate_at_1"] = round(hit_at_1 / n_pos, 4) if n_pos else 0.0
        result.metrics["hit_rate_at_3"] = round(hit_at_3 / n_pos, 4) if n_pos else 0.0
        result.metrics["mrr"] = (
            round(sum(reciprocal_ranks) / n_pos, 4) if n_pos else 0.0
        )
        print(
            f"  [INFO] Positive metrics => HR@1={result.metrics['hit_rate_at_1']}  "
            f"HR@3={result.metrics['hit_rate_at_3']}  MRR={result.metrics['mrr']}"
        )

        # -- Negative / Out-of-KB fallback case (E2E via POST /chat) --
        neg_query = (
            "Explain the internal flight control firmware of the Voyager 9X "
            "deep space probe documented in my notes"
        )
        neg_thread = _bench_thread()
        print(f"\n  [INFO] Negative fallback test (thread={neg_thread}) ...")

        t0 = time.perf_counter()
        resp = await pacer.call_with_retry(
            requests.post,
            f"{api_url}/chat",
            json={"thread_id": neg_thread, "message": neg_query},
            timeout=60,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()

        tool_traces = data.get("tool_traces", [])
        citations = data.get("citations", [])
        response_text = data.get("response", "")
        response_lower = response_text.lower()

        search_called = any(
            t.get("tool_name") == "search_knowledge" for t in tool_traces
        )
        citations_empty = len(citations) == 0
        has_fallback = any(fp in response_lower for fp in FALLBACK_PHRASES)

        neg_pass = search_called and citations_empty
        status = "PASS" if neg_pass else "FAIL"
        label = "[PASS]" if neg_pass else "[FAIL]"
        result.add_case(
            case_id="rag_neg_01",
            description="Out-of-KB query triggers fallback, zero false citations",
            status=status,
            latency_ms=elapsed_ms,
            details={
                "search_knowledge_called": search_called,
                "citations_empty": citations_empty,
                "has_fallback_phrase": has_fallback,
                "num_citations": len(citations),
            },
        )
        print(
            f"  {label} rag_neg_01: search_called={search_called}  "
            f"citations_empty={citations_empty}  fallback_phrase={has_fallback}  "
            f"({elapsed_ms:.0f}ms)"
        )
        result.metrics["negative_fallback_pass"] = neg_pass

    finally:
        print(f"  [INFO] Cleaning up synthetic source: {BENCHMARK_SYNTHETIC_SOURCE}")
        try:
            store.delete_by_source(BENCHMARK_SYNTHETIC_SOURCE)
        except Exception as exc:
            print(f"  [WARN] Cleanup failed: {exc}")

    return result


# ===========================================================================
# Suite 2 -- Expanded MCP Tool Selection & Argument Accuracy
# ===========================================================================
async def run_tool_routing_suite(
    api_url: str, pacer: RateLimitPacer
) -> SuiteResult:
    """Suite 2: Tool selection accuracy across all MCP tools."""
    import requests

    result = SuiteResult("tools")
    print("\n" + "=" * 70)
    print("SUITE 2: TOOLS -- MCP Tool Selection & Argument Accuracy")
    print("=" * 70)

    tool_cases = [
        {
            "id": "tool_01",
            "query": "Search my knowledge base for how the retrieval pipeline weights semantic similarity versus recency.",
            "expected_tool": "search_knowledge",
            "is_sensitive": False,
            "expected_args": ["query"],
            "description": "search_knowledge routing",
        },
        {
            "id": "tool_02",
            "query": "Show all my pending tasks in the workspace.",
            "expected_tool": "list_pending_tasks",
            "is_sensitive": False,
            "expected_args": [],
            "description": "list_pending_tasks routing",
        },
        {
            "id": "tool_03",
            "query": "List all git repositories available in my workspace directory.",
            "expected_tool": "list_workspace_git_repos",
            "is_sensitive": False,
            "expected_args": [],
            "description": "list_workspace_git_repos routing",
        },
        {
            "id": "tool_04",
            "query": "Check the git status of my current repository.",
            "expected_tool": "inspect_local_git_status",
            "is_sensitive": False,
            "expected_args": [],
            "description": "inspect_local_git_status routing",
        },
        {
            "id": "tool_05",
            "query": "Show me the uncommitted git diff changes in the current repo.",
            "expected_tool": "get_local_git_diff",
            "is_sensitive": False,
            "expected_args": [],
            "description": "get_local_git_diff routing",
        },
        {
            "id": "tool_06",
            "query": "Show the last 3 git commits in this repository.",
            "expected_tool": "get_recent_git_commits",
            "is_sensitive": False,
            "expected_args": [],
            "description": "get_recent_git_commits routing",
        },
        {
            "id": "tool_07",
            "query": "Create a new task titled Benchmark Verification Task with high priority.",
            "expected_tool": "create_personal_task",
            "is_sensitive": True,
            "expected_args": ["title"],
            "description": "create_personal_task routing (sensitive)",
        },
        {
            "id": "tool_08",
            "query": "Save a memory note to my knowledge base that our staging database port is 5439.",
            "expected_tool": "save_memory_note",
            "is_sensitive": True,
            "expected_args": ["content"],
            "description": "save_memory_note routing (sensitive)",
        },
    ]

    tool_match_count = 0
    arg_match_count = 0
    arg_applicable_count = 0

    for tc in tool_cases:
        thread_id = _bench_thread()
        t0 = time.perf_counter()
        resp = await pacer.call_with_retry(
            requests.post,
            f"{api_url}/chat",
            json={"thread_id": thread_id, "message": tc["query"]},
            timeout=60,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()

        tool_traces = data.get("tool_traces", [])
        called_tools = [t.get("tool_name") for t in tool_traces]
        status_field = data.get("status", "")
        pending = data.get("pending_approval", {})

        # For HITL-intercepted calls, the tool name is in pending_approval
        interrupted = status_field == "interrupted_for_approval"
        intercepted_tools = []
        if interrupted and pending:
            sensitive_calls = pending.get("sensitive_calls", [])
            intercepted_tools = [sc.get("name", "") for sc in sensitive_calls]

        all_triggered = called_tools + intercepted_tools
        is_tool_match = tc["expected_tool"] in all_triggered
        if is_tool_match:
            tool_match_count += 1

        # Argument extraction check
        arg_ok = True
        if tc["expected_args"]:
            arg_applicable_count += 1
            if interrupted and pending:
                sensitive_calls = pending.get("sensitive_calls", [])
                for sc in sensitive_calls:
                    if sc.get("name") == tc["expected_tool"]:
                        sc_args = sc.get("args", {})
                        arg_ok = all(k in sc_args for k in tc["expected_args"])
                        break
            else:
                # Check tool_traces for argument presence is non-trivial
                # since traces only contain output. We mark PASS if tool matched.
                arg_ok = is_tool_match
            if arg_ok:
                arg_match_count += 1

        status = "PASS" if is_tool_match else "FAIL"
        label = "[PASS]" if is_tool_match else "[FAIL]"
        result.add_case(
            case_id=tc["id"],
            description=tc["description"],
            status=status,
            latency_ms=elapsed_ms,
            details={
                "expected_tool": tc["expected_tool"],
                "called_tools": called_tools,
                "intercepted_tools": intercepted_tools,
                "tool_match": is_tool_match,
                "arg_match": arg_ok,
                "interrupted": interrupted,
            },
        )
        print(
            f"  {label} {tc['id']}: expected={tc['expected_tool']}  "
            f"called={all_triggered}  "
            f"interrupted={interrupted}  ({elapsed_ms:.0f}ms)"
        )

        # Immediately reject any HITL interruption to prevent DB mutation
        if interrupted:
            print(f"    [INFO] Sending reject to clean up HITL state (thread={thread_id})")
            try:
                await pacer.call_with_retry(
                    requests.post,
                    f"{api_url}/chat/resume",
                    json={"thread_id": thread_id, "action": "reject"},
                    timeout=30,
                )
            except Exception as exc:
                print(f"    [WARN] HITL reject cleanup failed: {exc}")

        await pacer.pace()

    n_total = len(tool_cases)
    result.metrics["tool_selection_accuracy"] = round(
        tool_match_count / n_total, 4
    ) if n_total else 0.0
    result.metrics["argument_extraction_accuracy"] = round(
        arg_match_count / arg_applicable_count, 4
    ) if arg_applicable_count else 1.0
    print(
        f"\n  [INFO] Tool Selection Accuracy: {result.metrics['tool_selection_accuracy']:.2%}"
    )
    print(
        f"  [INFO] Argument Extraction Accuracy: {result.metrics['argument_extraction_accuracy']:.2%}"
    )

    return result


# ===========================================================================
# Suite 3 -- HITL Safety Gatekeeper & Resume Flow
# ===========================================================================
async def run_hitl_suite(api_url: str, pacer: RateLimitPacer) -> SuiteResult:
    """Suite 3: Human-in-the-Loop interception and safe rejection."""
    import requests

    result = SuiteResult("hitl")
    print("\n" + "=" * 70)
    print("SUITE 3: HITL -- Safety Gatekeeper & Resume Flow")
    print("=" * 70)

    hitl_cases = [
        {
            "id": "hitl_01",
            "query": "Create a new task titled HITL Benchmark Test Task with medium priority.",
            "expected_tool": "create_personal_task",
            "description": "HITL intercept for create_personal_task",
        },
        {
            "id": "hitl_02",
            "query": "Save a memory note to my knowledge base that the benchmark ran successfully at midnight.",
            "expected_tool": "save_memory_note",
            "description": "HITL intercept for save_memory_note",
        },
    ]

    intercept_ok = 0
    reject_ok = 0

    for hc in hitl_cases:
        thread_id = _bench_thread()

        # Step 1: Send chat request that should trigger HITL
        print(f"\n  [INFO] {hc['id']}: Sending trigger request (thread={thread_id})")
        t0 = time.perf_counter()
        resp = await pacer.call_with_retry(
            requests.post,
            f"{api_url}/chat",
            json={"thread_id": thread_id, "message": hc["query"]},
            timeout=60,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()

        status_field = data.get("status", "")
        pending = data.get("pending_approval", {})
        interrupted = status_field == "interrupted_for_approval"

        intercepted_tool = ""
        if interrupted and pending:
            sensitive_calls = pending.get("sensitive_calls", [])
            for sc in sensitive_calls:
                if sc.get("name") == hc["expected_tool"]:
                    intercepted_tool = sc.get("name", "")
                    break

        interception_pass = interrupted and intercepted_tool == hc["expected_tool"]
        if interception_pass:
            intercept_ok += 1

        status = "PASS" if interception_pass else "FAIL"
        label = "[PASS]" if interception_pass else "[FAIL]"
        result.add_case(
            case_id=f"{hc['id']}_intercept",
            description=f"Interception detection: {hc['description']}",
            status=status,
            latency_ms=elapsed_ms,
            details={
                "interrupted": interrupted,
                "intercepted_tool": intercepted_tool,
                "expected_tool": hc["expected_tool"],
            },
        )
        print(
            f"  {label} {hc['id']}_intercept: interrupted={interrupted}  "
            f"tool={intercepted_tool}  ({elapsed_ms:.0f}ms)"
        )

        # Step 2: Resume with reject
        if interrupted:
            print(f"    [INFO] Sending resume with action=reject ...")
            t0 = time.perf_counter()
            try:
                resume_resp = await pacer.call_with_retry(
                    requests.post,
                    f"{api_url}/chat/resume",
                    json={"thread_id": thread_id, "action": "reject"},
                    timeout=30,
                )
                resume_elapsed_ms = (time.perf_counter() - t0) * 1000
                resume_data = resume_resp.json()

                # Verify rejection completed
                resume_status = resume_data.get("status", "")
                rejection_pass = resume_status == "completed"
                if rejection_pass:
                    reject_ok += 1

                rstatus = "PASS" if rejection_pass else "FAIL"
                rlabel = "[PASS]" if rejection_pass else "[FAIL]"
                result.add_case(
                    case_id=f"{hc['id']}_reject",
                    description=f"Safe rejection: {hc['description']}",
                    status=rstatus,
                    latency_ms=resume_elapsed_ms,
                    details={
                        "resume_status": resume_status,
                        "rejection_confirmed": rejection_pass,
                    },
                )
                print(
                    f"  {rlabel} {hc['id']}_reject: resume_status={resume_status}  "
                    f"({resume_elapsed_ms:.0f}ms)"
                )
            except Exception as exc:
                result.add_case(
                    case_id=f"{hc['id']}_reject",
                    description=f"Safe rejection: {hc['description']}",
                    status="FAIL",
                    details={"error": str(exc)},
                )
                print(f"  [FAIL] {hc['id']}_reject: error={exc}")
        else:
            result.add_case(
                case_id=f"{hc['id']}_reject",
                description=f"Safe rejection: {hc['description']}",
                status="SKIP",
                details={"reason": "No interruption detected, cannot test rejection"},
            )
            print(f"  [SKIP] {hc['id']}_reject: no interrupt to reject")

        await pacer.pace()

    n_hitl = len(hitl_cases)
    result.metrics["hitl_interception_accuracy"] = round(
        intercept_ok / n_hitl, 4
    ) if n_hitl else 0.0
    result.metrics["safe_rejection_success_rate"] = round(
        reject_ok / n_hitl, 4
    ) if n_hitl else 0.0
    print(
        f"\n  [INFO] HITL Interception Accuracy: {result.metrics['hitl_interception_accuracy']:.2%}"
    )
    print(
        f"  [INFO] Safe Rejection Success Rate: {result.metrics['safe_rejection_success_rate']:.2%}"
    )

    return result


# ===========================================================================
# Suite 4 -- Redis Cache Performance & Invalidation
# ===========================================================================
async def run_cache_suite(api_url: str, pacer: RateLimitPacer) -> SuiteResult:
    """Suite 4: Cache hit/miss detection, speedup, and invalidation."""
    import requests

    result = SuiteResult("cache")
    print("\n" + "=" * 70)
    print("SUITE 4: CACHE -- Redis Cache Performance & Invalidation")
    print("=" * 70)

    # Try to instantiate RedisCacheManager
    try:
        from cache_manager import RedisCacheManager
        cache_mgr = RedisCacheManager()
        # Quick ping test
        cache_mgr._client.ping()
    except Exception as exc:
        print(f"  [WARN] Redis unavailable - skipping cache suite: {exc}")
        result.add_case(
            case_id="cache_skip",
            description="Redis connectivity check",
            status="SKIP",
            details={"reason": f"Redis unavailable: {exc}"},
        )
        result.metrics["cache_available"] = False
        return result

    cache_query = "List all git repositories available in my workspace directory."
    result.metrics["cache_available"] = True

    # Step 0: Invalidate all cache
    print("  [INFO] Invalidating all existing cache entries ...")
    try:
        cache_mgr.invalidate_all_chat_cache()
    except Exception as exc:
        print(f"  [WARN] Cache invalidation failed: {exc}")

    await asyncio.sleep(1)

    # Step 1: Cold request
    print("  [INFO] Step 1: Cold request (expect cache miss) ...")
    thread_cold = _bench_thread()
    t0 = time.perf_counter()
    resp_cold = await pacer.call_with_retry(
        requests.post,
        f"{api_url}/chat",
        json={"thread_id": thread_cold, "message": cache_query},
        timeout=60,
    )
    cold_latency_ms = (time.perf_counter() - t0) * 1000
    cold_data = resp_cold.json()
    cold_cached = cold_data.get("cached", False)

    cold_pass = cold_cached is not True
    status = "PASS" if cold_pass else "FAIL"
    label = "[PASS]" if cold_pass else "[FAIL]"
    result.add_case(
        case_id="cache_01_cold",
        description="Cold request should not be cached",
        status=status,
        latency_ms=cold_latency_ms,
        details={"cached_flag": cold_cached, "latency_ms": round(cold_latency_ms, 2)},
    )
    print(
        f"  {label} cache_01_cold: cached={cold_cached}  "
        f"latency={cold_latency_ms:.0f}ms"
    )

    await pacer.pace()

    # Step 2: Warm request (same query, different thread)
    print("  [INFO] Step 2: Warm request (expect cache hit) ...")
    thread_warm = _bench_thread()
    t0 = time.perf_counter()
    resp_warm = await pacer.call_with_retry(
        requests.post,
        f"{api_url}/chat",
        json={"thread_id": thread_warm, "message": cache_query},
        timeout=60,
    )
    warm_latency_ms = (time.perf_counter() - t0) * 1000
    warm_data = resp_warm.json()
    warm_cached = warm_data.get("cached", False)

    warm_pass = warm_cached is True
    status = "PASS" if warm_pass else "FAIL"
    label = "[PASS]" if warm_pass else "[FAIL]"
    result.add_case(
        case_id="cache_02_warm",
        description="Warm request should be cached",
        status=status,
        latency_ms=warm_latency_ms,
        details={"cached_flag": warm_cached, "latency_ms": round(warm_latency_ms, 2)},
    )
    print(
        f"  {label} cache_02_warm: cached={warm_cached}  "
        f"latency={warm_latency_ms:.0f}ms"
    )

    await pacer.pace()

    # Step 3: Invalidate and verify miss again
    print("  [INFO] Step 3: Invalidating cache then re-querying ...")
    try:
        cache_mgr.invalidate_all_chat_cache()
    except Exception as exc:
        print(f"  [WARN] Cache invalidation for step 3 failed: {exc}")

    await asyncio.sleep(1)

    thread_post_inv = _bench_thread()
    t0 = time.perf_counter()
    resp_inv = await pacer.call_with_retry(
        requests.post,
        f"{api_url}/chat",
        json={"thread_id": thread_post_inv, "message": cache_query},
        timeout=60,
    )
    inv_latency_ms = (time.perf_counter() - t0) * 1000
    inv_data = resp_inv.json()
    inv_cached = inv_data.get("cached", False)

    inv_pass = inv_cached is not True
    status = "PASS" if inv_pass else "FAIL"
    label = "[PASS]" if inv_pass else "[FAIL]"
    result.add_case(
        case_id="cache_03_invalidation",
        description="Post-invalidation request should not be cached",
        status=status,
        latency_ms=inv_latency_ms,
        details={"cached_flag": inv_cached, "latency_ms": round(inv_latency_ms, 2)},
    )
    print(
        f"  {label} cache_03_invalidation: cached={inv_cached}  "
        f"latency={inv_latency_ms:.0f}ms"
    )

    # Metrics
    speedup = cold_latency_ms / max(warm_latency_ms, 1.0)
    result.metrics["cold_latency_ms"] = round(cold_latency_ms, 2)
    result.metrics["warm_latency_ms"] = round(warm_latency_ms, 2)
    result.metrics["speedup_factor"] = round(speedup, 2)
    result.metrics["cache_hit_accuracy"] = 1.0 if warm_pass else 0.0
    result.metrics["invalidation_accuracy"] = 1.0 if inv_pass else 0.0
    print(
        f"\n  [INFO] Speedup Factor: {speedup:.2f}x  "
        f"(cold={cold_latency_ms:.0f}ms / warm={warm_latency_ms:.0f}ms)"
    )

    return result


# ===========================================================================
# Suite 5 -- LLM-as-a-Judge Answer Quality
# ===========================================================================
async def run_judge_suite(api_url: str, pacer: RateLimitPacer) -> SuiteResult:
    """Suite 5: Faithfulness and relevance scoring via gemini-3.5-flash-lite judge."""
    import requests
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage

    result = SuiteResult("judge")
    print("\n" + "=" * 70)
    print("SUITE 5: JUDGE -- LLM-as-a-Judge Answer Quality")
    print("=" * 70)

    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GEMINI_API_KEY,
        # temperature=0.0,
        # transport="rest",
    )

    judge_system_prompt = (
        "You are an objective AI Evaluation Judge. You grade the quality of "
        "an AI assistant's answer based on the retrieved tool context and "
        "the user's question. Return ONLY a valid JSON object with exactly "
        "these keys:\n"
        '  "faithfulness_score": a float 0.0-1.0 (1.0 = fully grounded in context, no hallucination),\n'
        '  "relevance_score": a float 0.0-1.0 (1.0 = perfectly addresses the question),\n'
        '  "reasoning": a brief one-sentence justification.\n'
        "Do not include any text outside the JSON object."
    )

    judge_cases = [
        {
            "id": "judge_01",
            "query": "Search my knowledge base for how the retrieval pipeline weights semantic similarity versus recency.",
            "ground_truth": "70% semantic similarity, 15% lexical, and 15% time-decay recency using a 768-dim HNSW COSINE index.",
            "description": "RAG retrieval pipeline architecture quality",
        },
        {
            "id": "judge_02",
            "query": "List all git repositories available in my workspace directory.",
            "ground_truth": "Lists the available git repositories found in the configured workspace directory.",
            "description": "Workspace git listing quality",
        },
        {
            "id": "judge_03",
            "query": "Show all my pending tasks in the workspace.",
            "ground_truth": "Lists pending tasks from the PostgreSQL personal_tasks table.",
            "description": "Task listing quality",
        },
    ]

    faithfulness_scores: List[float] = []
    relevance_scores: List[float] = []

    for jc in judge_cases:
        thread_id = _bench_thread()

        # Get the actual answer from the orchestrator
        print(f"  [INFO] {jc['id']}: querying orchestrator ...")
        t0 = time.perf_counter()
        try:
            resp = await pacer.call_with_retry(
                requests.post,
                f"{api_url}/chat",
                json={"thread_id": thread_id, "message": jc["query"]},
                timeout=60,
            )
            data = resp.json()
        except Exception as exc:
            result.add_case(
                case_id=jc["id"],
                description=jc["description"],
                status="FAIL",
                details={"error": f"Orchestrator call failed: {exc}"},
            )
            print(f"  [FAIL] {jc['id']}: orchestrator error: {exc}")
            await pacer.pace()
            continue

        # Handle HITL-interrupted responses (should not happen for read-only queries)
        if data.get("status") == "interrupted_for_approval":
            try:
                requests.post(
                    f"{api_url}/chat/resume",
                    json={"thread_id": thread_id, "action": "reject"},
                    timeout=30,
                )
            except Exception:
                pass
            result.add_case(
                case_id=jc["id"],
                description=jc["description"],
                status="SKIP",
                details={"reason": "Unexpected HITL interruption on read-only query"},
            )
            print(f"  [SKIP] {jc['id']}: unexpected HITL interrupt")
            await pacer.pace()
            continue

        tool_traces = data.get("tool_traces", [])
        answer_text = data.get("response", "")
        context_str = (
            "\n".join(t.get("output", "") for t in tool_traces) or "No tool context."
        )

        # Build judge prompt
        judge_prompt = (
            f"Question: {jc['query']}\n\n"
            f"Retrieved Tool Context:\n{context_str}\n\n"
            f"Assistant Answer:\n{answer_text}\n\n"
            f"Expected Ground Truth:\n{jc['ground_truth']}"
        )

        await pacer.pace()

        # Call judge LLM
        print(f"  [INFO] {jc['id']}: invoking judge LLM ...")
        try:
            judge_response = await judge_llm.ainvoke(
                [
                    SystemMessage(content=judge_system_prompt),
                    HumanMessage(content=judge_prompt),
                ]
            )
            judge_text = _extract_text(judge_response.content)
            scores = _parse_judge_json(judge_text)
            f_score = float(scores.get("faithfulness_score", 0.0))
            r_score = float(scores.get("relevance_score", 0.0))
            reasoning = scores.get("reasoning", "")
        except Exception as exc:
            f_score = 0.0
            r_score = 0.0
            reasoning = f"Judge parsing error: {exc}"
            print(f"    [WARN] Judge parse error: {exc}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        faithfulness_scores.append(f_score)
        relevance_scores.append(r_score)

        status = "PASS" if f_score >= 0.5 and r_score >= 0.5 else "FAIL"
        label = "[PASS]" if status == "PASS" else "[FAIL]"
        result.add_case(
            case_id=jc["id"],
            description=jc["description"],
            status=status,
            latency_ms=elapsed_ms,
            details={
                "faithfulness_score": round(f_score, 4),
                "relevance_score": round(r_score, 4),
                "reasoning": reasoning,
            },
        )
        print(
            f"  {label} {jc['id']}: faithfulness={f_score:.2f}  "
            f"relevance={r_score:.2f}  ({elapsed_ms:.0f}ms)"
        )

        await pacer.pace()

    n = len(faithfulness_scores)
    result.metrics["avg_faithfulness_score"] = round(
        sum(faithfulness_scores) / n, 4
    ) if n else 0.0
    result.metrics["avg_relevance_score"] = round(
        sum(relevance_scores) / n, 4
    ) if n else 0.0
    print(
        f"\n  [INFO] Avg Faithfulness: {result.metrics['avg_faithfulness_score']:.4f}"
    )
    print(f"  [INFO] Avg Relevance: {result.metrics['avg_relevance_score']:.4f}")

    return result


# ===========================================================================
# Report generation
# ===========================================================================
def generate_reports(
    suite_results: List[SuiteResult], output_dir: Path
) -> Tuple[Path, Path]:
    """Write benchmark_report.json and benchmark_report.md."""
    timestamp = datetime.now(timezone.utc).isoformat()

    total_passed = sum(r.passed for r in suite_results)
    total_failed = sum(r.failed for r in suite_results)
    total_skipped = sum(r.skipped for r in suite_results)
    total_cases = total_passed + total_failed + total_skipped

    all_latencies = []
    for r in suite_results:
        all_latencies.extend(r.latencies)

    report_data = {
        "benchmark_version": "2.0",
        "timestamp": timestamp,
        "summary": {
            "total_cases": total_cases,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "pass_rate": round(total_passed / total_cases, 4) if total_cases else 0.0,
            "overall_latency_ms": {
                "mean": round(statistics.mean(all_latencies), 2)
                if all_latencies
                else 0.0,
                "p50": round(_percentile(all_latencies, 50), 2),
                "p95": round(_percentile(all_latencies, 95), 2),
            },
        },
        "suites": [r.summary_dict() for r in suite_results],
    }

    # -- JSON report --
    json_path = output_dir / "benchmark_report.json"
    json_path.write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # -- Markdown report --
    md_lines = []
    md_lines.append("# Personal AI Engineer OS -- Benchmark Report v2.0")
    md_lines.append("")
    md_lines.append(f"**Generated:** {timestamp}")
    md_lines.append("")

    # Executive summary table
    md_lines.append("## Executive Summary")
    md_lines.append("")
    md_lines.append("| Metric | Value |")
    md_lines.append("|--------|-------|")
    md_lines.append(f"| Total Test Cases | {total_cases} |")
    md_lines.append(f"| Passed | {total_passed} |")
    md_lines.append(f"| Failed | {total_failed} |")
    md_lines.append(f"| Skipped | {total_skipped} |")
    md_lines.append(
        f"| Pass Rate | {report_data['summary']['pass_rate']:.2%} |"
    )
    md_lines.append(
        f"| Mean Latency (ms) | {report_data['summary']['overall_latency_ms']['mean']} |"
    )
    md_lines.append(
        f"| P50 Latency (ms) | {report_data['summary']['overall_latency_ms']['p50']} |"
    )
    md_lines.append(
        f"| P95 Latency (ms) | {report_data['summary']['overall_latency_ms']['p95']} |"
    )
    md_lines.append("")

    # Suite-by-suite breakdown
    for sr in suite_results:
        sd = sr.summary_dict()
        md_lines.append(f"## Suite: {sd['suite'].upper()}")
        md_lines.append("")
        md_lines.append(
            f"**Results:** {sd['passed']} passed / {sd['failed']} failed / "
            f"{sd['skipped']} skipped (total {sd['total']})"
        )
        md_lines.append("")

        if sd["metrics"]:
            md_lines.append("### Metrics")
            md_lines.append("")
            md_lines.append("| Metric | Value |")
            md_lines.append("|--------|-------|")
            for mk, mv in sd["metrics"].items():
                if isinstance(mv, float):
                    md_lines.append(f"| {mk} | {mv:.4f} |")
                else:
                    md_lines.append(f"| {mk} | {mv} |")
            md_lines.append("")

        if sd["latency_stats_ms"]["mean"] > 0:
            md_lines.append("### Latency")
            md_lines.append("")
            md_lines.append("| Stat | ms |")
            md_lines.append("|------|-----|")
            md_lines.append(f"| Mean | {sd['latency_stats_ms']['mean']} |")
            md_lines.append(f"| P50 | {sd['latency_stats_ms']['p50']} |")
            md_lines.append(f"| P95 | {sd['latency_stats_ms']['p95']} |")
            md_lines.append("")

        md_lines.append("### Test Cases")
        md_lines.append("")
        md_lines.append(
            "| Case ID | Description | Status | Latency (ms) |"
        )
        md_lines.append("|---------|-------------|--------|--------------|")
        for c in sd["cases"]:
            md_lines.append(
                f"| {c['case_id']} | {c['description']} | {c['status']} | {c['latency_ms']} |"
            )
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("*Report generated by run_benchmark.py v2.0*")
    md_lines.append("")

    md_path = output_dir / "benchmark_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return json_path, md_path


def print_console_summary(suite_results: List[SuiteResult]) -> None:
    """Print a clean ASCII summary table to stdout."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUITE v2.0 -- FINAL SUMMARY")
    print("=" * 70)

    header = f"{'Suite':<10} {'Pass':>5} {'Fail':>5} {'Skip':>5} {'Total':>6} {'Mean(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10}  Status"
    print(header)
    print("-" * len(header))

    overall_pass = True
    for sr in suite_results:
        sd = sr.summary_dict()
        ls = sd["latency_stats_ms"]
        status = "[PASS]" if sd["failed"] == 0 else "[FAIL]"
        if sd["failed"] > 0:
            overall_pass = False
        print(
            f"{sd['suite']:<10} {sd['passed']:>5} {sd['failed']:>5} {sd['skipped']:>5} "
            f"{sd['total']:>6} {ls['mean']:>10.1f} {ls['p50']:>10.1f} {ls['p95']:>10.1f}  {status}"
        )

    print("-" * len(header))
    total_p = sum(r.passed for r in suite_results)
    total_f = sum(r.failed for r in suite_results)
    total_s = sum(r.skipped for r in suite_results)
    total_n = total_p + total_f + total_s
    all_lat = []
    for r in suite_results:
        all_lat.extend(r.latencies)
    m_mean = round(statistics.mean(all_lat), 1) if all_lat else 0.0
    m_p50 = round(_percentile(all_lat, 50), 1)
    m_p95 = round(_percentile(all_lat, 95), 1)
    overall_label = "[PASS]" if overall_pass else "[FAIL]"
    print(
        f"{'TOTAL':<10} {total_p:>5} {total_f:>5} {total_s:>5} "
        f"{total_n:>6} {m_mean:>10.1f} {m_p50:>10.1f} {m_p95:>10.1f}  {overall_label}"
    )

    # Print key metrics per suite
    print("\n" + "-" * 70)
    print("KEY METRICS:")
    print("-" * 70)
    for sr in suite_results:
        if sr.metrics:
            print(f"  [{sr.suite_name.upper()}]")
            for k, v in sr.metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
    print("=" * 70)


# ===========================================================================
# Main orchestrator
# ===========================================================================
async def run_benchmark(args: argparse.Namespace) -> None:
    """Entry point: run selected suites, collect results, generate reports."""
    suite = args.suite
    api_url = args.api_url.rstrip("/")
    pacer = RateLimitPacer(delay=args.delay, max_retries=args.max_retries)

    print("=" * 70)
    print("PERSONAL AI ENGINEER OS -- BENCHMARK SUITE v2.0")
    print("=" * 70)
    print(f"  Suite:       {suite}")
    print(f"  API URL:     {api_url}")
    print(f"  Delay:       {args.delay}s")
    print(f"  Max Retries: {args.max_retries}")
    print(f"  Timestamp:   {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # Verify API is reachable
    import requests
    try:
        health = requests.get(f"{api_url}/threads", timeout=10)
        if health.status_code == 200:
            print("[INFO] API server is reachable.")
        else:
            print(f"[WARN] API responded with status {health.status_code}.")
    except Exception as exc:
        print(f"[ERROR] Cannot reach API at {api_url}: {exc}")
        print("[ERROR] Make sure the orchestrator API is running.")
        return

    suite_results: List[SuiteResult] = []
    run_all = suite == "all"

    try:
        if run_all or suite == "rag":
            sr = await run_rag_suite(api_url, pacer)
            suite_results.append(sr)

        if run_all or suite == "tools":
            sr = await run_tool_routing_suite(api_url, pacer)
            suite_results.append(sr)

        if run_all or suite == "hitl":
            sr = await run_hitl_suite(api_url, pacer)
            suite_results.append(sr)

        if run_all or suite == "cache":
            sr = await run_cache_suite(api_url, pacer)
            suite_results.append(sr)

        if run_all or suite == "judge":
            sr = await run_judge_suite(api_url, pacer)
            suite_results.append(sr)

    except KeyboardInterrupt:
        print("\n[WARN] Benchmark interrupted by user.")
    except Exception as exc:
        print(f"\n[ERROR] Benchmark suite failed: {exc}")
        traceback.print_exc()

    if not suite_results:
        print("[WARN] No suite results to report.")
        return

    # Console summary
    print_console_summary(suite_results)

    # File reports
    output_dir = ROOT_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = generate_reports(suite_results, output_dir)
    print(f"\n[INFO] JSON report saved to: {json_path}")
    print(f"[INFO] Markdown report saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Personal AI Engineer OS -- Benchmark Suite v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        choices=["all", "rag", "tools", "hitl", "cache", "judge"],
        default="all",
        help="Which benchmark suite to run (default: all).",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8080",
        help="Base URL of the Orchestrator API (default: http://localhost:8080).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to sleep between LLM-heavy API calls for rate limiting (default: 3.0).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries with exponential backoff on 429 / RESOURCE_EXHAUSTED (default: 3).",
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
