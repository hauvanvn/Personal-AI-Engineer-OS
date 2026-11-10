import os
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dotenv import dotenv_values, load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor, Json

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


def _get_postgres_uri() -> str:
    """Read POSTGRES_URI from .env file or environment variable with default fallback."""
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        try:
            env_vals = dotenv_values(env_path)
            uri = env_vals.get("POSTGRES_URI")
            if uri:
                return uri.strip()
        except Exception:
            pass
    return os.getenv("POSTGRES_URI", "postgresql://admin:secretpassword@localhost:5432/PaPa_db")


class SecurityAuditStateManager:
    """Manages persistent state and stage checkpoints for repository security audits in PostgreSQL."""

    def __init__(self, db_uri: Optional[str] = None):
        self.db_uri = db_uri or _get_postgres_uri()
        self.ensure_table_exists()

    def ensure_table_exists(self) -> None:
        """Create the security_audit_runs table and indexes if they do not exist."""
        with psycopg2.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS security_audit_runs (
                        audit_id VARCHAR(64) PRIMARY KEY,
                        repo_source TEXT NOT NULL,
                        source_type VARCHAR(32) NOT NULL,
                        resolved_local_path TEXT NOT NULL,
                        commit_hash VARCHAR(64),
                        current_stage VARCHAR(64) NOT NULL DEFAULT 'initialized',
                        status VARCHAR(32) NOT NULL DEFAULT 'in_progress',
                        static_risk_score INTEGER DEFAULT 0,
                        static_risk_verdict VARCHAR(32) DEFAULT 'PENDING',
                        stage_results JSONB NOT NULL DEFAULT '{}'::jsonb,
                        report_path TEXT,
                        error_message TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_security_audit_repo_source ON security_audit_runs(repo_source);
                    CREATE INDEX IF NOT EXISTS idx_security_audit_status ON security_audit_runs(status);
                    """
                )
            conn.commit()

    def create_audit_run(
        self,
        repo_source: str,
        source_type: str,
        resolved_local_path: str,
        commit_hash: str = "",
    ) -> Dict[str, Any]:
        """Create a new security audit run with current_stage='initialized' and status='in_progress'."""
        audit_id = f"audit_{uuid.uuid4().hex[:12]}"
        initial_stage_results = {
            "preparation": {
                "repo_source": repo_source,
                "source_type": source_type,
                "resolved_local_path": resolved_local_path,
                "commit_hash": commit_hash,
            }
        }
        with psycopg2.connect(self.db_uri) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO security_audit_runs (
                        audit_id,
                        repo_source,
                        source_type,
                        resolved_local_path,
                        commit_hash,
                        current_stage,
                        status,
                        static_risk_score,
                        static_risk_verdict,
                        stage_results
                    ) VALUES (
                        %s, %s, %s, %s, %s, 'initialized', 'in_progress', 0, 'PENDING', %s
                    ) RETURNING *;
                    """,
                    (
                        audit_id,
                        repo_source,
                        source_type,
                        resolved_local_path,
                        commit_hash,
                        Json(initial_stage_results),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return dict(row) if row else {}

    def get_audit_run(self, audit_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific audit run record by its audit_id."""
        with psycopg2.connect(self.db_uri) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM security_audit_runs
                    WHERE audit_id = %s;
                    """,
                    (audit_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def find_resumable_audit(self, repo_source: str) -> Optional[Dict[str, Any]]:
        """Find the most recent resumable audit run for a given repository source."""
        with psycopg2.connect(self.db_uri) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM security_audit_runs
                    WHERE repo_source = %s AND status IN ('in_progress', 'rate_limited_paused')
                    ORDER BY updated_at DESC
                    LIMIT 1;
                    """,
                    (repo_source,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def get_latest_audit_for_repo(self, repo_source: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest audit run record for a repository source regardless of status."""
        with psycopg2.connect(self.db_uri) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM security_audit_runs
                    WHERE repo_source = %s
                    ORDER BY updated_at DESC
                    LIMIT 1;
                    """,
                    (repo_source,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def save_stage_checkpoint(
        self,
        audit_id: str,
        stage_name: str,
        stage_key: str,
        stage_data: dict,
        status: str = "in_progress",
        static_risk_score: Optional[int] = None,
        static_risk_verdict: Optional[str] = None,
        report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge stage_data under stage_key into stage_results, update stage, status, and optional metrics."""
        new_stage_payload = {stage_key: stage_data}
        with psycopg2.connect(self.db_uri) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE security_audit_runs
                    SET stage_results = COALESCE(stage_results, '{}'::jsonb) || %s::jsonb,
                        current_stage = %s,
                        status = %s,
                        updated_at = CURRENT_TIMESTAMP,
                        static_risk_score = COALESCE(%s, static_risk_score),
                        static_risk_verdict = COALESCE(%s, static_risk_verdict),
                        report_path = COALESCE(%s, report_path)
                    WHERE audit_id = %s
                    RETURNING *;
                    """,
                    (
                        Json(new_stage_payload),
                        stage_name,
                        status,
                        static_risk_score,
                        static_risk_verdict,
                        report_path,
                        audit_id,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        if not row:
            raise ValueError(f"Audit run not found for audit_id: {audit_id}")
        return dict(row)

    def mark_rate_limited(self, audit_id: str, error_message: str) -> Dict[str, Any]:
        """Pause audit run due to rate limit, recording error_message and preserving completed stage_results."""
        with psycopg2.connect(self.db_uri) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE security_audit_runs
                    SET status = 'rate_limited_paused',
                        error_message = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE audit_id = %s
                    RETURNING *;
                    """,
                    (error_message, audit_id),
                )
                row = cur.fetchone()
            conn.commit()
        if not row:
            raise ValueError(f"Audit run not found for audit_id: {audit_id}")
        return dict(row)

    def list_audit_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent audit runs ordered by updated_at descending."""
        with psycopg2.connect(self.db_uri) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM security_audit_runs
                    ORDER BY updated_at DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def delete_audit_run(self, audit_id: str) -> bool:
        """Delete an audit run record by audit_id."""
        with psycopg2.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM security_audit_runs
                    WHERE audit_id = %s
                    RETURNING audit_id;
                    """,
                    (audit_id,),
                )
                deleted = cur.fetchone() is not None
            conn.commit()
        return deleted
