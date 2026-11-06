"""Lightweight metadata store for conversation threads in PostgreSQL.

Tracks thread_id, latest message preview, and timestamps so the UI
can display a list of past conversation sessions without deserializing
full LangGraph checkpoint blobs.
"""

from typing import List, Dict, Any
import psycopg


class ThreadMetadataStore:
    """Manages the chat_threads metadata table in PostgreSQL."""

    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self._init_table()

    def _init_table(self):
        """Create the chat_threads table if it does not exist."""
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_threads (
                        thread_id TEXT PRIMARY KEY,
                        last_message_preview TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
            conn.commit()

    def upsert_thread(self, thread_id: str, last_message_preview: str) -> None:
        """Insert or update a thread record with the latest message preview."""
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_threads (thread_id, last_message_preview, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (thread_id) DO UPDATE
                    SET last_message_preview = EXCLUDED.last_message_preview,
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    (thread_id, last_message_preview),
                )
            conn.commit()

    def list_threads(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return threads ordered by most recently updated, newest first."""
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT thread_id,
                           last_message_preview,
                           TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at,
                           TO_CHAR(updated_at, 'YYYY-MM-DD HH24:MI:SS') AS updated_at
                    FROM chat_threads
                    ORDER BY updated_at DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()

        return [
            {
                "thread_id": r[0],
                "last_message_preview": r[1],
                "created_at": r[2],
                "updated_at": r[3],
            }
            for r in rows
        ]

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread metadata record. Returns True if a row was removed."""
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_threads WHERE thread_id = %s",
                    (thread_id,),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted
