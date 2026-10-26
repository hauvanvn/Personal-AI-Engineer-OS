import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
import psycopg
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

mcp = FastMCP("Dev-Life-Workspace")

# Connect to PostgreSQL in docker-compose
DB_URI = os.getenv("POSTGRES_URI", "postgresql://admin:secretpassword@localhost:5432/PaPa_db")

def init_db():
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS personal_tasks (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    priority VARCHAR(20) DEFAULT 'medium',
                    status VARCHAR(20) DEFAULT 'pending',
                    due_date VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()

init_db()

# ==========================================
# GROUP 1: READ-ONLY TOOLS (automation)
# ==========================================

@mcp.tool()
def list_pending_tasks(
    priority: Optional[str] = Field(None, description="Filtered by prioroty: 'low', 'medium', 'high'")
) -> str:
    """Retrieve a list of the user's pending tasks/deadlines from PostgreSQL."""
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            if priority:
                cur.execute(
                    "SELECT id, title, priority, due_date FROM personal_tasks WHERE status = 'pending' AND priority = %s ORDER BY id DESC",
                    (priority,)
                )
            else:
                cur.execute("SELECT id, title, priority, due_date FROM personal_tasks WHERE status = 'pending' ORDER BY id DESC")
            rows = cur.fetchall()

    if not rows:
        return "There are currently no pending tasks."
    
    return "\n".join([f"- [ID: {r[0]}] {r[1]} (Priority: {r[2]} | Deadline: {r[3] or 'None'})" for r in rows])

@mcp.tool()
def inspect_local_git_status(
    repo_path: str = Field(".", description="Relative or absolute path to the git repository directory")
) -> str:
    """Quickly check the current Git branch, the files being modified, and the 3 most recent commits."""
    target = Path(repo_path).resolve()
    if not target.exists():
        return f"The directory {target} does not exist."
    
    try:
        status = subprocess.check_output(["git", "status", "-s"], cwd=target, text=True, timeout=5)
        log = subprocess.check_output(["git", "log", "-n", "3", "--oneline"], cwd=target, text=True, timeout=5)
        return f"=== Git Status ({target.name}) ===\n{status or 'Working tree clean'}\n=== Recent Commits ===\n{log}"
    except Exception as e:
        return f"Unable to read Git information at {target}: {str(e)}"

# ==========================================
# GROUP 2: MUTATE TOOLS (Need Human Approval)
# ==========================================

@mcp.tool()
def create_personal_task(
    title: str = Field(..., description="Name of the task or deadline to be created"),
    priority: str = Field("high", description="Priority: 'low', 'medium', 'high'"),
    due_date: Optional[str] = Field(None, description="Deadline (YYYY-MM-DD or describe the time)")
) -> str:
    """[MUTATE] Create a new task or deadline in the database PostgreSQL."""
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO personal_tasks (title, priority, due_date) VALUES (%s, %s, %s) RETURNING id",
                (title, priority, due_date)
            )
            task_id = cur.fetchone()[0]
        conn.commit()
    return f"Task #{task_id}: '{title}' (Priority: {priority}, Deadline: {due_date}) created successfully."

@mcp.tool()
def complete_personal_task(
    task_id: int = Field(..., description="ID of the task to be marked as complete")
) -> str:
    """[MUTATE] Mark a task in the list as completed."""
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE personal_tasks SET status = 'completed' WHERE id = %s RETURNING title", (task_id,))
            row = cur.fetchone()
        conn.commit()
    if not row:
        return f"Task ID #{task_id} not found."
    return f"Marked Task #{task_id}: '{row[0]}' as complete."

if __name__ == "__main__":
    mcp.run()