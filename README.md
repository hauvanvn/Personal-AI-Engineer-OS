# PaPa — Personal AI Engineer OS

> **P**ersonal **A**I **P**owered **A**ssistant — An agentic AI system that acts as your **Second Brain** and **Dev Workspace Manager**, powered by LangGraph, MCP Servers, Milvus vector search, and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Orchestration-orange)
![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-green)
![Milvus](https://img.shields.io/badge/Milvus-v3.0.3-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red?logo=streamlit)

---

## Overview

PaPa is a **full-stack agentic AI assistant** designed for software engineers to:

- **Search & store personal knowledge** — past notes, architecture decisions, slides, and PDFs — in a vector database (Milvus) using Agentic RAG with time-decay reranking.
- **Manage tasks and deadlines** — create, list, and complete personal engineering tasks stored in PostgreSQL.
- **Inspect local Git repositories** — check branch status, modified files, and recent commits.
- **Learn user preferences over time** — automatically extract long-term user facts (tech stack, preferences, projects) from conversations and personalize future responses.

All state-mutating actions pass through a **Human-in-the-Loop (HITL) Safety Gatekeeper** that pauses execution for user approval before writing to databases.

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                       │
│              (Dashboard + HITL Approval Cards)            │
└──────────────────────┬────────────────────────────────────┘
                       │ HTTP (REST)
┌──────────────────────▼────────────────────────────────────┐
│               FastAPI Orchestrator API                    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              LangGraph State Machine                │  │
│  │                                                     │  │
│  │   START ──► Planner ──► Safety Gatekeeper ──► END   │  │
│  │               ▲              │                      │  │
│  │               │              ▼                      │  │
│  │               └──── MCP Tool Node ◄─── HITL ───┘    │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─────────────────┐       ┌──────────────────────────┐   │
│  │ Long-Term Memory│       │  Gemini 3.5 Flash Lite   │   │
│  │   Manager (LLM) │       │  (Planning + Extraction) │   │
│  └────────┬────────┘       └──────────────────────────┘   │
└───────────┼───────────────────────────────────────────────┘
            │
     ┌──────┴──────┐
     │  PostgreSQL │──── user_memories, personal_tasks,
     │  (Postgres) │     document_registry, LangGraph checkpoints
     └─────────────┘
            │
┌───────────┴─────────────────────────────────────────────────┐
│                     MCP Servers (stdio)                     │
│                                                             │
│  ┌─────────────────────┐    ┌────────────────────────────┐  │
│  │  second-brain-rag   │    │   dev-life-workspace       │  │
│  │                     │    │                            │  │
│  │  • search_knowledge │    │  • list_pending_tasks      │  │
│  │  • save_memory_note │    │  • create_personal_task    │  │
│  │                     │    │  • complete_personal_task  │  │
│  │  ┌───────────────┐  │    │  • inspect_local_git_status│  │
│  │  │  Milvus v3.0  │  │    │                            │  │
│  │  │  HNSW + Cosine│  │    └────────────────────────────┘  │
│  │  │  768-dim      │  │                                    │
│  │  └───────────────┘  │                                    │
│  └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
PersonalAIAssistant/
├── apps/
│   ├── orchestrator-api/
│   │   ├── server.py            # FastAPI + LangGraph orchestrator
│   │   └── memory_store.py      # Long-term user memory extraction (PostgreSQL)
│   ├── web-ui/
│   │   ├── dashboard.py         # Streamlit chat UI with HITL approval cards
│   │   └── resource/bot.png     # Bot avatar icon
│   └── background-worker/
│       └── ingest_worker.py     # Document ingestion pipeline (PDF/MD/TXT → Milvus)
├── mcp-servers/
│   ├── second-brain-rag/
│   │   ├── server.py            # MCP server: search_knowledge, save_memory_note
│   │   └── milvus_store.py      # Milvus vector store with time-decay reranking
│   └── dev-life-workspace/
│       └── server.py            # MCP server: task CRUD + Git inspection
├── evaluation/
│   ├── run_benchmark.py         # Automated benchmark suite (RAG + tool selection + LLM-as-Judge)
│   └── benchmark_report.json    # Latest benchmark results
├── knowledge_base/              # Drop PDF/MD/TXT files here for ingestion
├── docker-compose.yaml          # PostgreSQL, Redis, Milvus (etcd + MinIO + standalone)
├── pyproject.toml               # Project dependencies (uv)
├── .env                         # API keys (GEMINI_API_KEY)
└── main.py                      # Entry point placeholder
```

---

## Key Features

### Agentic RAG with Time-Decay Reranking
- Documents are embedded using **Gemini Embedding 001** (768-dim Matryoshka vectors).
- Stored in **Milvus** with HNSW indexing (M=16, efConstruction=200, Cosine similarity).
- Retrieval scoring: **85% semantic similarity + 15% recency decay** (`1 / (1 + 0.01 × age_days)`).

### Human-in-the-Loop Safety Gatekeeper
- State-mutating tools (`create_personal_task`, `complete_personal_task`, `save_memory_note`) require explicit user approval.
- The LangGraph `interrupt()` mechanism pauses the graph, and the Streamlit UI presents an approval card with **Approve / Edit & Run / Reject** options.

### Long-Term Memory (Auto-Extracted User Profile)
- After each conversation turn, a background LLM call extracts **permanent user facts** (tech stack, preferences, workflows) and stores them in PostgreSQL.
- These facts are injected into the system prompt to **personalize every future interaction**.

### Document Ingestion Pipeline
- The background worker scans `knowledge_base/` for PDF, Markdown, and TXT files.
- **SHA-256 idempotent indexing** — files are only re-indexed if their content has changed.
- Structure-aware chunking with paragraph boundaries and sliding window overlap (1200 chars, 200 overlap).

### MCP Tool Servers
| Server | Tools | Purpose |
|---|---|---|
| `second-brain-rag` | `search_knowledge`, `save_memory_note` | RAG search & note storage in Milvus |
| `dev-life-workspace` | `list_pending_tasks`, `create_personal_task`, `complete_personal_task`, `inspect_local_git_status` | Task management (PostgreSQL) & Git inspection |

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (Python package manager)
- **Docker & Docker Compose**
- **Google Gemini API Key** ([Get one here](https://aistudio.google.com/apikey))

### 1. Clone & Install Dependencies

```bash
git clone <repository-url>
cd PersonalAIAssistant

# Install Python dependencies with uv
uv sync
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Start Infrastructure Services

```bash
docker compose up -d
```

This starts:
- **PostgreSQL** (port `5432`) — tasks, user memories, LangGraph checkpoints
- **Redis** (port `6379`) — available for future use
- **Milvus Standalone** (port `19530`) — vector database (with etcd + MinIO backends)

### 4. Ingest Documents into the Second Brain

Place your PDF, Markdown, or TXT files in the `knowledge_base/` directory, then run:

```bash
uv run python apps/background-worker/ingest_worker.py
```

### 5. Start the Orchestrator API

```bash
uv run python apps/orchestrator-api/server.py
```

The FastAPI server starts at `http://localhost:8080` with the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send a message to the AI assistant |
| `POST` | `/chat/resume` | Resume an interrupted graph (approve/edit/reject) |
| `GET` | `/memories` | View extracted long-term user profile facts |

### 6. Launch the Web UI

```bash
uv run streamlit run apps/web-ui/dashboard.py
```

Open the Streamlit dashboard (default: `http://localhost:8501`) to start chatting.

---

## Evaluation & Benchmarks

The project includes an automated benchmark suite that evaluates:

| Metric | Description | Latest Result |
|---|---|---|
| **RAG Hit Rate @3** | Does the correct source appear in the top-3 retrieval results? | 50.0% |
| **MCP Tool Selection Accuracy** | Does the orchestrator invoke the correct MCP tool? | 100.0% |
| **Answer Faithfulness** | Is the response grounded in retrieved context? (LLM-as-Judge) | 75.0% |
| **Average E2E Latency** | End-to-end response time through the full pipeline | 3.66s |

Run the benchmark (requires the orchestrator API to be running):

```bash
uv run python evaluation/run_benchmark.py
```

Results are saved to `evaluation/benchmark_report.json`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini 3.5 Flash Lite |
| **Embeddings** | Gemini Embedding 001 (768-dim) |
| **Agent Framework** | LangGraph (StateGraph with checkpointing) |
| **Tool Protocol** | Model Context Protocol (MCP) via `langchain-mcp-adapters` |
| **Vector Database** | Milvus v3.0.3 (HNSW, Cosine) |
| **Relational Database** | PostgreSQL 16 |
| **Web UI** | Streamlit |
| **API** | FastAPI + Uvicorn |
| **Package Manager** | uv |
| **Infrastructure** | Docker Compose |

---

## License

This is a personal project. Feel free to use it as a reference or starting point for your own AI assistant.
