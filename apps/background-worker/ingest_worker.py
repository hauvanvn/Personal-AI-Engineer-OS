import hashlib
import os
import sys
import time
from pathlib import Path
from typing import List
import pymupdf  # PyMuPDF
import psycopg
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Import SecondBrainStore from the MCP server package
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
        """Create PostgreSQL table to track file SHA-256 hashes for idempotent ingestion."""
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_registry (
                        source_file VARCHAR(255) PRIMARY KEY,
                        file_hash VARCHAR(64) NOT NULL,
                        chunk_count INT NOT NULL,
                        doc_type VARCHAR(50) NOT NULL,
                        indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
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
                    "SELECT file_hash FROM document_registry WHERE source_file = %s",
                    (source_file,),
                )
                row = cur.fetchone()
                return bool(row and row[0] == current_hash)

    def update_registry(
        self, source_file: str, file_hash: str, chunk_count: int, doc_type: str
    ):
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_registry (source_file, file_hash, chunk_count, doc_type, indexed_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (source_file) DO UPDATE
                    SET file_hash = EXCLUDED.file_hash,
                        chunk_count = EXCLUDED.chunk_count,
                        indexed_at = CURRENT_TIMESTAMP;
                    """,
                    (source_file, file_hash, chunk_count, doc_type),
                )
            conn.commit()

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
                # Keep trailing overlap from previous chunk for context continuity
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

    def sync_directory(self, directory: Path = KNOWLEDGE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        files = [
            f
            for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in (".pdf", ".md", ".txt")
        ]

        if not files:
            print(f"[*] No documents found in '{directory}'. Creating a sample architecture doc...")
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

        print(f"[*] Scanning {len(files)} file(s) in '{directory.name}/'...")
        for file_path in files:
            rel_name = file_path.name
            file_hash = self.compute_sha256(file_path)

            if self.is_file_unchanged(rel_name, file_hash):
                print(f"  [SKIP] '{rel_name}' is unchanged (SHA-256 verified).")
                continue

            t0 = time.perf_counter()
            doc_type = "pdf" if file_path.suffix.lower() == ".pdf" else "architecture"
            chunks = self.parse_file(file_path)

            # Remove outdated vectors before inserting updated chunks
            self.store.delete_by_source(rel_name)
            inserted = self.store.add_documents_batch(
                chunks=chunks, source_file=rel_name, doc_type=doc_type
            )
            self.update_registry(rel_name, file_hash, inserted, doc_type)
            print(
                f"  [INDEXED] '{rel_name}' -> {inserted} chunk(s) in {time.perf_counter() - t0:.2f}s"
            )


if __name__ == "__main__":
    worker = DocumentIngestionWorker()
    worker.sync_directory()