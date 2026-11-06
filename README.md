# PaPa — Personal AI Engineer OS

> **P**ersonal **A**I **P**owered **A**ssistant — An agentic AI operating system that acts as your **Second Brain** and **Dev Workspace Manager**, powered by LangGraph, Model Context Protocol (MCP) servers, Milvus hybrid vector search, Redis response caching, and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Orchestration-orange)
![MCP](https://img.shields.io/badge/MCP-3%20Tool%20Servers-green)
![Milvus](https://img.shields.io/badge/Milvus-v3.0.3-blue)
![Redis](https://img.shields.io/badge/Redis-Response%20Cache-red?logo=redis)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-teal?logo=fastapi)

---

## Overview

PaPa is a **production-grade agentic AI assistant** purpose-built for software engineers:

- **Second Brain Knowledge Base (Hybrid RAG)** — Ingest and retrieve personal notes, system designs, slides, and PDFs using a 3-way hybrid scoring algorithm: **70% semantic similarity + 15% BM25 lexical keyword matching + 15% exponential time-decay recency**, with Gemini Free-Tier TPM rate-limit protection.
- **Web Search Fallback** — Real-time internet search via Tavily MCP server when queries miss the local knowledge base or require current data, with proactive recommendations to persist useful findings into the Second Brain.
- **Dev Workspace & Multi-Repo Git Operations** — Discover local Git repositories up to depth 2, inspect branch status, staged/unstaged diffs, and commit history, with runtime workspace switching without server restarts.
- **Task & Deadline Management** — Track, create, and complete engineering tasks backed by PostgreSQL.
- **Long-Term Memory (Auto-Extracted User Profile)** — Asynchronously extract permanent user preferences, tech stack, and project contexts across conversation turns, with full CRUD support.
- **Sub-Millisecond Redis Response Caching** — Deterministic SHA-256 query caching with automatic non-blocking cache invalidation upon document ingestion or deletion.
- **Multi-Session Conversation Threads** — Persistent chat sessions in PostgreSQL with a session switcher in the Web UI.
- **Human-in-the-Loop (HITL) Safety Gatekeeper** — Intercept state-mutating actions (`create_personal_task`, `complete_personal_task`, `save_memory_note`) using LangGraph checkpoints, enabling user **Approve**, **Edit Arguments & Run**, or **Reject**.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   Streamlit Web UI                                     │
│  • Session Switcher   • Chat & Citations (KB + Web)   • HITL Argument Approval Cards   │
│  • Document Upload & Ingestion Monitor (CRUD)         • Long-Term Profile Manager      │
│  • Dynamic Workspace Directory Switcher               • MCP Tool Execution Traces      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ HTTP (REST)
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                              FastAPI Orchestrator API                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              LangGraph State Machine                             │  │
│  │                                                                                  │  │
│  │           START ───► Planner ───► Safety Gatekeeper ───► END                     │  │
│  │                        ▲                  │                                      │  │
│  │                        │                  ▼                                      │  │
│  │                        └──── MCP Tools ◄─── HITL Approval (Interrupt)            │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                        │
│  ┌─────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐  │
│  │   Redis Cache Manager   │   │  Long-Term Memory (LLM)  │   │ Thread Store (Meta) │  │
│  │ (SHA-256 Query Hashing) │   │ (Auto Fact Extraction)   │   │ (Session Switcher)  │  │
│  └─────────────────────────┘   └──────────────────────────┘   └─────────────────────┘  │
└──────────────────────┬──────────────────────┬───────────────────────────┬──────────────┘
                       │                      │                           │
          ┌────────────┴────────┐   ┌─────────┴─────────┐       ┌─────────┴─────────┐
          │     PostgreSQL      │   │    Redis Cache    │       │  Gemini 3.5 Flash │
          │  • personal_tasks   │   │  • ai_os:chat_    │       │       Lite        │
          │  • user_memories    │   │    cache:*        │       │  (Planning, LLM   │
          │  • document_registry│   │  • Sub-ms latency │       │   Judge & Memory) │
          │  • chat_threads     │   │  • Auto-inval on  │       └───────────────────┘
          │  • LangGraph State  │   │    KB mutations   │
          └─────────────────────┘   └───────────────────┘
                       │
┌──────────────────────┴─────────────────────────────────────────────────────────────────┐
│                                MCP Servers (stdio transport)                           │
│                                                                                        │
│  ┌──────────────────────────────┐  ┌───────────────────────────┐  ┌─────────────────┐  │
│  │       second-brain-rag       │  │    dev-life-workspace     │  │   web-search    │  │
│  │                              │  │                           │  │                 │  │
│  │  • search_knowledge          │  │  • list_pending_tasks     │  │  • search_web   │  │
│  │    (70% Semantic, 15% BM25,  │  │  • create_personal_task   │  │    (Tavily API  │  │
│  │     15% Time-Decay Recency)  │  │  • complete_personal_task │  │     real-time   │  │
│  │  • save_memory_note [MUTATE] │  │  • list_workspace_git_    │  │     fallback)   │  │
│  │                              │  │    repos                  │  │                 │  │
│  │  ┌────────────────────────┐  │  │  • inspect_local_git_     │  └─────────────────┘  │
│  │  │   Milvus v3.0 (HNSW)   │  │  │    status                 │                       │
│  │  │   Cosine + Lexical +   │  │  │  • get_local_git_diff     │                       │
│  │  │   768-dim Embeddings   │  │  │  • get_recent_git_commits │                       │
│  │  └────────────────────────┘  │  │  • set_workspace_dir      │                       │
│  └──────────────────────────────┘  └───────────────────────────┘                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
PersonalAIAssistant/
├── apps/
│   ├── orchestrator-api/
│   │   ├── server.py              # FastAPI server + LangGraph workflow + REST endpoints
│   │   ├── cache_manager.py       # Redis response caching with SHA-256 query hashing
│   │   ├── memory_store.py        # Long-term user memory extraction & CRUD (PostgreSQL)
│   │   └── thread_store.py        # Conversation thread metadata store (PostgreSQL)
│   ├── web-ui/
│   │   ├── dashboard.py           # Streamlit UI (sessions, doc/memory CRUD, HITL cards, traces)
│   │   └── resource/
│   │       └── bot.png            # Assistant avatar icon
│   └── background-worker/
│       └── ingest_worker.py       # Document ingestion worker (PDF/MD/TXT -> Milvus, rate-limited)
├── mcp-servers/
│   ├── second-brain-rag/
│   │   ├── server.py              # FastMCP server: search_knowledge, save_memory_note
│   │   └── milvus_store.py        # Milvus vector store with 3-way hybrid search & TPM throttling
│   ├── dev-life-workspace/
│   │   └── server.py              # FastMCP server: task CRUD, multi-repo Git inspection, dynamic workspace
│   └── web-search/
│       └── server.py              # FastMCP server: real-time internet search fallback (Tavily)
├── evaluation/
│   ├── run_benchmark.py           # Modular Benchmark Suite v2.0 (RAG, Tools, HITL, Cache, Judge)
│   ├── benchmark_report.json      # Benchmark execution metrics (JSON)
│   └── benchmark_report.md        # Benchmark executive summary report (Markdown)
├── knowledge_base/                # Local directory for ingesting PDF, Markdown, and TXT files
├── docker-compose.yaml            # PostgreSQL 16, Redis 7, Milvus v3.0.3 (etcd + MinIO + standalone)
├── pyproject.toml                 # Project dependencies managed via uv
├── .env                           # Environment configuration & API keys
└── main.py                        # Entry point placeholder
```

---

## Key Features

### 1. Hybrid Agentic RAG with Time-Decay Reranking
- **3-Way Hybrid Scoring**:
  $$\text{Final Score} = 0.70 \times \text{Semantic} + 0.15 \times \text{Lexical} + 0.15 \times \text{Recency}$$
  - **70% Semantic Similarity**: Gemini Embedding 001 (768-dim normalized vectors) with Milvus HNSW indexing.
  - **15% Lexical Matching**: BM25-inspired keyword frequency saturation with IDF weighting for technical identifiers and exact terms.
  - **15% Time-Decay Recency**: Exponential decay formula exp(-0.01 x age\_hours).
- **Oversampling & Thresholding**: Queries retrieve an expanded candidate pool (5 x top\_k or minimum 25) before reranking, filtering out candidates below `min_score` (default 0.50).
- **Inline Citations**: Grounded responses display exact source file citations with page numbers and relevance scores.

### 2. Production-Grade Rate-Limit Protection (Free Tier Resilient)
- **Token-Aware Sub-Batching**: Documents are split into sub-batches bounded by item count (max 10) and estimated tokens (max 6,000 tokens).
- **Rolling TPM Budget Throttle**: Continuous 60-second window tracker staying safely under the 30,000 TPM limit.
- **Exponential Backoff**: Automatic retry handling for HTTP 429 / `RESOURCE_EXHAUSTED` errors with regex parsing for suggested retry delays (`retry in Xs`).
- **Idempotent Ingestion**: SHA-256 hash checks prevent duplicate embeddings; existing chunks are purged via `delete_by_source` prior to re-indexing.

### 3. Sub-Millisecond Redis Response Caching
- **Deterministic Key Hashing**: Standard queries are indexed in Redis using normalized SHA-256 hashes (`ai_os:chat_cache:<hash>`) with a 1-hour configurable TTL.
- **Zero-Latency Retrieval**: Repeated queries return instantaneously ($0.0\,\text{s}$ latency) with complete tool traces and citation metadata.
- **Non-Blocking Cache Invalidation**: SCAN-based invalidation cleans cached queries whenever documents are uploaded or deleted.
- **Safety Bypass**: State-mutating queries and pending HITL interrupts automatically bypass caching to prevent stale approvals.

### 4. Real-Time Web Search Fallback (Tavily MCP)
- **Knowledge-First Routing Hierarchy**:
  1. Priority lookup in the local Second Brain knowledge base (`search_knowledge`).
  2. If the local search returns no results or lacks sufficient detail, the orchestrator automatically calls `search_web` via Tavily.
  3. General knowledge is used only as a last resort when both sources fail.
- **Memory Persistence Suggestion**: When high-value web information is found, the assistant proactively offers to save the summary into Second Brain via `save_memory_note`.

### 5. Multi-Repo Git Inspection & Dynamic Workspace
- **Multi-Repository Discovery**: Automatically scans `WORKSPACE_DIR` up to depth 2, identifying all active Git repositories and their current branches (`list_workspace_git_repos`).
- **Deep Git Inspection**:
  - `inspect_local_git_status`: Short status and latest 3 commits.
  - `get_local_git_diff`: Unstaged or staged (`staged_only=True`) diffs with character capping.
  - `get_recent_git_commits`: Detailed commit history with `--stat` summaries.
- **Runtime Workspace Switching**: Change the active workspace path on the fly (`set_workspace_dir` or via Web UI) without restarting the API or MCP servers; updates persist to `.env`.

### 6. Human-in-the-Loop (HITL) Safety Gatekeeper
- State-mutating tools (`create_personal_task`, `complete_personal_task`, `save_memory_note`) trigger a LangGraph `interrupt()`.
- The Web UI displays an approval card allowing the engineer to:
  - **Approve**: Execute tool call as planned.
  - **Save Edited & Run**: Modify tool JSON arguments (e.g., adjust priority or filename) before execution.
  - **Reject Action**: Cancel execution and notify the planner to take an alternate path.

### 7. Persistent Conversation Threads & Session Switcher
- **PostgreSQL Session Storage**: Thread metadata, latest message previews, and timestamps are recorded in the `chat_threads` table.
- **Session Switcher UI**: Quickly create new conversation threads or reload past discussions with full chat history, tool traces, and citations.

### 8. Full CRUD for Knowledge Base & Long-Term Memories
- **Knowledge Base CRUD**: Upload `.pdf`, `.md`, and `.txt` files directly in the UI with background ingestion, inspect indexing status, or delete documents (purging Milvus vectors, PostgreSQL registry, and physical disk files).
- **Long-Term Memory CRUD**: View auto-extracted user facts in the UI and delete outdated or incorrect facts with a single click.

---

## MCP Tool Servers

The orchestrator integrates three specialized MCP servers running over standard I/O (`stdio`):

| Server | Tool Name | Type | Description |
|---|---|:---:|---|
| **`second-brain-rag`** | `search_knowledge` | Read | Hybrid search (70% semantic, 15% BM25 lexical, 15% time-decay recency) |
| **`second-brain-rag`** | `save_memory_note` | **Mutate (HITL)** | Store a note or architecture decision into Milvus vector store |
| **`dev-life-workspace`** | `list_pending_tasks` | Read | List pending engineering tasks from PostgreSQL |
| **`dev-life-workspace`** | `create_personal_task` | **Mutate (HITL)** | Create a new task with title, priority, and deadline |
| **`dev-life-workspace`** | `complete_personal_task` | **Mutate (HITL)** | Mark an existing task as completed |
| **`dev-life-workspace`** | `list_workspace_git_repos` | Read | Discover Git repositories up to depth 2 in workspace |
| **`dev-life-workspace`** | `inspect_local_git_status` | Read | Check branch status, modified files, and 3 recent commits |
| **`dev-life-workspace`** | `get_local_git_diff` | Read | Inspect unstaged or staged diff with character truncation |
| **`dev-life-workspace`** | `get_recent_git_commits` | Read | View commit history with `--stat` summaries |
| **`dev-life-workspace`** | `set_workspace_dir` | Read | Dynamically update active workspace directory at runtime |
| **`web-search`** | `search_web` | Read | Real-time internet search fallback powered by Tavily API |

---

## Orchestrator REST API Reference

The FastAPI server provides endpoints for chat orchestration, thread management, document ingestion, and configuration:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send message to AI assistant (checks Redis cache, invokes LangGraph, records thread preview) |
| `POST` | `/chat/resume` | Resume interrupted graph with human decision (`approve`, `edit`, `reject`) |
| `GET` | `/threads` | List active conversation threads with latest message preview and timestamps |
| `GET` | `/threads/{thread_id}/messages` | Retrieve complete message history, tool traces, and citations for a thread |
| `GET` | `/memories` | Retrieve formatted user profile context and structured memory list |
| `DELETE` | `/memories/{memory_id}` | Delete a specific long-term memory fact by its ID |
| `POST` | `/knowledge/upload` | Upload `.pdf`, `.md`, or `.txt` files for background ingestion (invalidates cache) |
| `GET` | `/knowledge/documents` | List registered documents with chunk count and ingestion status |
| `DELETE` | `/knowledge/documents/{filename}` | Delete document vectors from Milvus, registry from PostgreSQL, and file from disk |
| `GET` | `/workspace` | Retrieve current active workspace directory path |
| `POST` | `/workspace` | Update workspace directory dynamically at runtime |

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (Fast Python package manager)
- **Docker & Docker Compose**
- **Google Gemini API Key** ([Get key here](https://aistudio.google.com/apikey))
- **Tavily Search API Key** *(Optional, enables Web Search)* ([Get key here](https://tavily.com))

---

### 1. Clone & Install Dependencies

```bash
git clone <repository-url>
cd PersonalAIAssistant

# Install dependencies using uv
uv sync
```

---

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Required: Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Tavily API Key for real-time web search fallback
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Workspace Directory (defaults to parent directory)
WORKSPACE_DIR=/path/to/your/projects

# Optional: Database URIs (default settings match docker-compose.yaml)
POSTGRES_URI=postgresql://admin:secretpassword@localhost:5432/PaPa_db
REDIS_URL=redis://localhost:6379/0
```

---

### 3. Start Infrastructure Services

```bash
docker compose up -d
```

This spins up:
- **PostgreSQL 16** (`localhost:5432`) — Tasks, long-term memories, document registry, thread metadata, and LangGraph checkpoints.
- **Redis 7** (`localhost:6379`) — Response caching with TTL expiration.
- **Milvus v3.0.3 Standalone** (`localhost:19530`) — Vector database with etcd and MinIO storage backends.

---

### 4. Start the Orchestrator API

```bash
uv run python apps/orchestrator-api/server.py
```

The FastAPI server starts at `http://localhost:8080`.

---

### 5. Ingest Documents into Second Brain

You can ingest documents in two ways:

**Option A: Via Web UI**
Open the Streamlit dashboard, upload files in the **Knowledge Base Ingestion** sidebar, and click **Ingest in Background**.

**Option B: Via Background CLI Worker**
Place documents in `knowledge_base/` and run:

```bash
uv run python apps/background-worker/ingest_worker.py
```

---

### 6. Launch the Web UI

```bash
uv run streamlit run apps/web-ui/dashboard.py
```

Open `http://localhost:8501` to start chatting with PaPa.

---

## Evaluation & Benchmarks (Suite v2.0)

The project includes an automated benchmark suite (`evaluation/run_benchmark.py`) covering five evaluation suites:

1. **Suite 1 (`rag`)** — Hybrid retrieval accuracy (Hit Rate @1, Hit Rate @3, MRR) and negative out-of-KB fallback handling.
2. **Suite 2 (`tools`)** — MCP tool selection accuracy and argument verification.
3. **Suite 3 (`hitl`)** — Safety gatekeeper interrupt, approve, edit, and reject workflows.
4. **Suite 4 (`cache`)** — Redis response cache hit latency and cache invalidation.
5. **Suite 5 (`judge`)** — LLM-as-a-Judge answer faithfulness and grounding.

### Running Benchmarks

```bash
# Run all benchmark suites
uv run python evaluation/run_benchmark.py --suite all

# Run specific suite with custom pacing delay (Free-Tier safe)
uv run python evaluation/run_benchmark.py --suite rag --delay 4.0

# View all options
uv run python evaluation/run_benchmark.py --help
```

### Latest Benchmark Results

Results from the latest benchmark run (`evaluation/benchmark_report.md`):

| Metric | Target | Latest Result | Status |
|---|:---:|:---:|:---:|
| **RAG Hit Rate @1** | $> 80\%$ | **100.0%** | PASS |
| **RAG Hit Rate @3** | $> 90\%$ | **100.0%** | PASS |
| **Mean Reciprocal Rank (MRR)** | $> 0.85$ | **1.0000** | PASS |
| **MCP Tool Selection Accuracy** | $> 95\%$ | **100.0%** | PASS |
| **Mean Latency (Retrieval)** | $< 3.0\text{s}$ | **2.35s** | PASS |
| **P50 Latency** | $< 1.0\text{s}$ | **0.48s** | PASS |
| **Redis Cache Latency** | $< 0.05\text{s}$ | **< 0.01s** | PASS |

Detailed reports are generated in `evaluation/benchmark_report.json` and `evaluation/benchmark_report.md`.

---

## Tech Stack

| Layer | Technology | Details |
|---|---|---|
| **LLM** | Google Gemini 3.5 Flash Lite | Planning, reasoning, memory extraction, evaluation |
| **Embeddings** | Gemini Embedding 001 | 768-dim normalized embeddings |
| **Agent Framework** | LangGraph | StateGraph with PostgreSQL checkpoints and `interrupt()` |
| **Tool Protocol** | Model Context Protocol (MCP) | `FastMCP` servers communicating via `stdio` |
| **Vector Database** | Milvus v3.0.3 | HNSW indexing, Cosine similarity, 3-way hybrid scoring |
| **Response Cache** | Redis 7 | Normalized SHA-256 query caching, SCAN invalidation |
| **Relational Database** | PostgreSQL 16 | Tasks, user memories, thread metadata, registry |
| **Web Search** | Tavily Search API | Real-time internet search fallback |
| **Document Parser** | PyMuPDF (fitz) | Fast PDF text and layout extraction |
| **Web UI** | Streamlit | Chat interface, session switcher, approval cards, CRUD panels |
| **API Server** | FastAPI + Uvicorn | Asynchronous REST endpoints and background tasks |
| **Package Manager** | uv | Ultra-fast Python package management |
| **Infrastructure** | Docker Compose | Multi-container orchestration |

---

## License

This is a personal project. Feel free to use it as a reference or starting point for your own AI assistant.
