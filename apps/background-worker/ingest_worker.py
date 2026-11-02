import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import pymupdf
import psycopg
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

sys.path.append(str(ROOT_DIR / "mcp-servers" / "second-brain-rag"))
from milvus_store import SecondBrainStore  # noqa: E402

DB_URI = os.getenv(
    "POSTGRES_URI", "postgresql://admin:secretpassword@localhost:5432/PaPa_db"
)
KNOWLEDGE_DIR = ROOT_DIR / "knowledge_base"


class DocumentIngestionWorker:
    def __init__(self):
        self.db_uri = DB_URI
        self.store = SecondBrainStore()
        self._init_registry_table()

    def _init_registry_table(self):
        """Create PostgreSQL table and ensure status column exists for background tracking."""
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_registry (
                        source_file VARCHAR(255) PRIMARY KEY,
                        file_hash VARCHAR(64) NOT NULL,
                        chunk_count INT NOT NULL,
                        doc_type VARCHAR(50) NOT NULL,
                        status VARCHAR(30) DEFAULT 'indexed',
                        indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE document_registry
                    ADD COLUMN IF NOT EXISTS status VARCHAR(30) DEFAULT 'indexed';
                    """
                )
            conn.commit()

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_file_unchanged(self, source_file: str, current_hash: str) -> bool:
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT file_hash, status, chunk_count FROM document_registry WHERE source_file = %s",
                    (source_file,),
                )
                row = cur.fetchone()
                if not row or row[0] != current_hash:
                    return False

                # If status is indexed or skipped_unchanged, verify chunks exist in Milvus
                if row[1] in ("indexed", "skipped_unchanged"):
                    milvus_chunks = self.store.count_by_source(source_file)
                    # If Milvus has chunks matching the recorded count, file is unchanged
                    if milvus_chunks > 0 and (row[2] == 0 or milvus_chunks >= row[2]):
                        return True
                    # If Milvus has 0 chunks (e.g. wiped or lost), it must re-ingest
                    return False

                # If status was left in 'processing' from an interrupted run, check Milvus
                if row[1] == "processing" and row[2] and row[2] > 0:
                    milvus_chunks = self.store.count_by_source(source_file)
                    if milvus_chunks >= row[2]:
                        return True

                return False

    def mark_status(
        self,
        source_file: str,
        status: str,
        file_hash: str = "",
        chunk_count: int = 0,
        doc_type: str = "note",
    ):
        """Update document ingestion status in PostgreSQL."""
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_registry (source_file, file_hash, chunk_count, doc_type, status, indexed_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (source_file) DO UPDATE
                    SET file_hash = CASE WHEN EXCLUDED.file_hash != '' THEN EXCLUDED.file_hash ELSE document_registry.file_hash END,
                        chunk_count = CASE WHEN EXCLUDED.chunk_count > 0 THEN EXCLUDED.chunk_count ELSE document_registry.chunk_count END,
                        doc_type = EXCLUDED.doc_type,
                        status = EXCLUDED.status,
                        indexed_at = CURRENT_TIMESTAMP;
                    """,
                    (source_file, file_hash, chunk_count, doc_type, status),
                )
            conn.commit()

    def list_registered_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all registered documents and their current ingestion status."""
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source_file, doc_type, chunk_count, status,
                           TO_CHAR(indexed_at, 'YYYY-MM-DD HH24:MI:SS')
                    FROM document_registry
                    ORDER BY indexed_at DESC;
                    """
                )
                rows = cur.fetchall()
        return [
            {
                "source_file": r[0],
                "doc_type": r[1],
                "chunk_count": r[2],
                "status": r[3],
                "indexed_at": r[4],
            }
            for r in rows
        ]

    @staticmethod
    def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
        """Structure-aware sliding window chunker preserving paragraph boundaries."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                tail = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = f"{tail}\n{para}".strip()

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def parse_file(self, file_path: Path) -> List[str]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            doc = pymupdf.open(file_path)
            chunks = []
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text").strip()
                if page_text:
                    page_chunks = self.chunk_text(page_text)
                    for c in page_chunks:
                        chunks.append(f"[Page {page_num}] {c}")
            doc.close()
            return chunks
        elif suffix in (".md", ".txt"):
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            return self.chunk_text(content)
        return []

    def _get_previous_status(self, source_file: str) -> str:
        """Return the current registry status for a file, or empty string if not found."""
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status FROM document_registry WHERE source_file = %s",
                    (source_file,),
                )
                row = cur.fetchone()
                return row[0] if row else ""

    def ingest_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Ingest a single document with SHA-256 deduplication and status tracking.

        If the file was previously in 'failed' or 'processing' state, any
        leftover partial chunks are deleted from Milvus before re-ingesting
        to prevent duplicates.
        """
        rel_name = file_path.name
        doc_type = "pdf" if file_path.suffix.lower() == ".pdf" else "architecture"

        try:
            file_hash = self.compute_sha256(file_path)
            if self.is_file_unchanged(rel_name, file_hash):
                self.mark_status(rel_name, status="skipped_unchanged", file_hash=file_hash, doc_type=doc_type)
                print(f"  [SKIP] '{rel_name}' is unchanged (SHA-256 verified).", flush=True)
                return {"source_file": rel_name, "status": "skipped_unchanged"}

            # Clean up partial leftovers from a previous interrupted or failed run
            prev_status = self._get_previous_status(rel_name)
            if prev_status in ("failed", "processing"):
                logger.info(
                    "Previous status for '%s' was '%s'. "
                    "Deleting stale chunks from Milvus before re-ingesting.",
                    rel_name, prev_status,
                )
                self.store.delete_by_source(rel_name)

            self.mark_status(
                rel_name, status="processing", file_hash=file_hash, doc_type=doc_type
            )
            logger.info("Starting ingestion of '%s' (type: %s)...", rel_name, doc_type)
            t0 = time.perf_counter()

            logger.info("Parsing '%s'...", rel_name)
            chunks = self.parse_file(file_path)
            logger.info(
                "Parsed %d chunk(s) from '%s'. Starting batch embedding and indexing into Milvus...",
                len(chunks), rel_name
            )
            self.store.delete_by_source(rel_name)
            inserted = self.store.add_documents_batch(
                chunks=chunks, source_file=rel_name, doc_type=doc_type
            )

            self.mark_status(
                rel_name,
                status="indexed",
                file_hash=file_hash,
                chunk_count=inserted,
                doc_type=doc_type,
            )
            elapsed = round(time.perf_counter() - t0, 2)
            logger.info("Successfully indexed '%s' -> %d chunk(s) in %.2fs", rel_name, inserted, elapsed)
            print(
                f"  [INDEXED] '{rel_name}' -> {inserted} chunk(s) in {elapsed}s",
                flush=True,
            )
            return {
                "source_file": rel_name,
                "status": "indexed",
                "chunks": inserted,
                "elapsed_sec": elapsed,
            }
        except Exception as e:
            error_msg = str(e)
            self.mark_status(rel_name, status="failed", doc_type=doc_type)
            logger.error("Failed to ingest '%s': %s", rel_name, error_msg)
            print(f"  [ERROR] Failed to ingest '{rel_name}': {error_msg}", flush=True)
            return {"source_file": rel_name, "status": "failed", "error": error_msg}

    def sync_directory(self, directory: Path = KNOWLEDGE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        files = [
            f
            for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in (".pdf", ".md", ".txt")
        ]

        if not files:
            print(f"[*] No documents found in '{directory}'. Creating a sample architecture doc...", flush=True)
            sample_file = directory / "system_design_guidelines.md"
            sample_file.write_text(
                "# Personal AI OS System Design\n\n"
                "The retrieval pipeline uses Milvus HNSW index with 768-dim Matryoshka embeddings "
                "generated by gemini-embedding-001. Time-decay reranking weights semantic similarity "
                "at 85% and document recency at 15%.\n\n"
                "## Safety & Concurrency\n\n"
                "All state-mutating MCP tool calls are intercepted by a LangGraph Safety Gatekeeper node "
                "using PostgreSQL checkpoints for Human-in-the-Loop approval.",
                encoding="utf-8",
            )
            files = [sample_file]

        print(f"[*] Scanning {len(files)} file(s) in '{directory.name}/'...", flush=True)
        for file_path in files:
            self.ingest_single_file(file_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    worker = DocumentIngestionWorker()
    worker.sync_directory()