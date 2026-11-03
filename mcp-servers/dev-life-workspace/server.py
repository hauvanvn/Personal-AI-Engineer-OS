import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import psycopg
from dotenv import load_dotenv, dotenv_values
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from pydantic.fields import FieldInfo

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

_DYNAMIC_WORKSPACE_DIR: Optional[str] = None
_LAST_ENV_MTIME: float = 0.0

def _get_workspace_dirs() -> Tuple[Path, List[Path]]:
    """Dynamically resolve current workspace directory from in-memory state, .env, or fallback."""
    global _DYNAMIC_WORKSPACE_DIR, _LAST_ENV_MTIME

    raw_workspace = _DYNAMIC_WORKSPACE_DIR
    env_path = ROOT_DIR / ".env"

    if env_path.exists():
        try:
            mtime = env_path.stat().st_mtime
            if mtime > _LAST_ENV_MTIME:
                _LAST_ENV_MTIME = mtime
                env_vals = dotenv_values(env_path)
                val = env_vals.get("WORKSPACE_DIR")
                if val:
                    raw_workspace = val.strip()
                    _DYNAMIC_WORKSPACE_DIR = raw_workspace
        except Exception:
            pass

    if not raw_workspace:
        raw_workspace = os.getenv("WORKSPACE_DIR", "").strip()

    if raw_workspace:
        dirs = [
            Path(os.path.expanduser(p.strip())).resolve()
            for p in raw_workspace.split(",")
            if p.strip()
        ]
        primary = (
            Path(os.path.expanduser(raw_workspace)).resolve()
            if len(dirs) == 1
            else (dirs[0] if dirs else ROOT_DIR.parent.resolve())
        )
        return primary, (dirs or [primary])

    fallback = ROOT_DIR.parent.resolve()
    return fallback, [fallback]


def _update_env_file(key: str, value: str):
    """Safely update or insert a key in the .env file without modifying other keys."""
    env_path = ROOT_DIR / ".env"
    lines = []
    found = False
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{key}=") or stripped == key:
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


WORKSPACE_DIR = ROOT_DIR.parent.resolve()
IGNORE_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__"}

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

def _is_git_repo(target: Path) -> bool:
    """Check whether a target directory is a valid Git repository."""
    if not target.is_dir():
        return False
    if (target / ".git").exists():
        return True
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=target,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return res.returncode == 0 and res.stdout.strip() == "true"
    except Exception:
        return False


def _get_git_branch(repo_path: Path) -> str:
    """Retrieve active branch name for the given repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        branch = res.stdout.strip()
        if branch:
            return branch
        res2 = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return res2.stdout.strip() or "HEAD"
    except Exception:
        return "unknown"


def _discover_workspace_git_repos() -> List[Dict[str, str]]:
    """Scan WORKSPACE_DIR up to depth 2 for Git repositories."""
    _, workspace_dirs = _get_workspace_dirs()
    found = []
    seen = set()
    for root in workspace_dirs:
        if not root.is_dir():
            continue
        if (root / ".git").exists():
            if root not in seen:
                seen.add(root)
                found.append(root)
            continue
        try:
            for item1 in root.iterdir():
                if not item1.is_dir() or item1.name.startswith(".") or item1.name in IGNORE_DIRS:
                    continue
                if (item1 / ".git").exists():
                    if item1 not in seen:
                        seen.add(item1)
                        found.append(item1)
                    continue
                try:
                    for item2 in item1.iterdir():
                        if not item2.is_dir() or item2.name.startswith(".") or item2.name in IGNORE_DIRS:
                            continue
                        if (item2 / ".git").exists():
                            if item2 not in seen:
                                seen.add(item2)
                                found.append(item2)
                except (PermissionError, OSError):
                    pass
        except (PermissionError, OSError):
            pass
    repos = [{"name": r.name, "branch": _get_git_branch(r), "path": str(r)} for r in found]
    repos.sort(key=lambda x: x["name"].lower())
    return repos


def _resolve_git_repo(repo_path: Optional[str] = None) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve repository path from repo_path, WORKSPACE_DIR, or default to repository root."""
    primary_ws, workspace_dirs = _get_workspace_dirs()
    if repo_path is None or not str(repo_path).strip() or str(repo_path).strip() == ".":
        if _is_git_repo(ROOT_DIR):
            return (ROOT_DIR, None)
        return (None, f"Default repository {ROOT_DIR} is not a valid Git repository.")

    cleaned = str(repo_path).strip()
    direct = Path(os.path.expanduser(cleaned)).resolve()
    if direct.is_dir() and _is_git_repo(direct):
        return (direct, None)

    for ws in workspace_dirs:
        if not ws.is_dir():
            continue
        if ws.name.lower() == cleaned.lower() and _is_git_repo(ws):
            return (ws, None)
        sub = (ws / cleaned).resolve()
        if sub.is_dir() and _is_git_repo(sub):
            return (sub, None)
        try:
            for item1 in ws.iterdir():
                if not item1.is_dir() or item1.name.startswith(".") or item1.name in IGNORE_DIRS:
                    continue
                if item1.name.lower() == cleaned.lower() and _is_git_repo(item1):
                    return (item1.resolve(), None)
                try:
                    for item2 in item1.iterdir():
                        if not item2.is_dir() or item2.name.startswith(".") or item2.name in IGNORE_DIRS:
                            continue
                        if item2.name.lower() == cleaned.lower() and _is_git_repo(item2):
                            return (item2.resolve(), None)
                except (PermissionError, OSError):
                    pass
        except (PermissionError, OSError):
            pass

    available = _discover_workspace_git_repos()
    if available:
        formatted = "\n".join(
            [f"- [{i}] {r['name']} (Branch: {r['branch']}) - Path: {r['path']}" for i, r in enumerate(available, 1)]
        )
        error_msg = f"Repository '{repo_path}' was not found or is not a valid Git repository.\n\nAvailable Git repositories in workspace:\n{formatted}"
    else:
        error_msg = f"Repository '{repo_path}' was not found or is not a valid Git repository. No Git repositories found in workspace ({primary_ws})."
    return (None, error_msg)

# ==========================================
# GROUP 1: READ-ONLY TOOLS (automation)
# ==========================================

@mcp.tool()
def list_workspace_git_repos() -> str:
    """List all Git repositories found inside WORKSPACE_DIR up to depth 2 with their active branches."""
    primary_ws, _ = _get_workspace_dirs()
    repos = _discover_workspace_git_repos()
    if not repos:
        return f"No Git repositories found in workspace ({primary_ws})."

    lines = [f"Available Workspace Git Repositories ({primary_ws}):"]
    for idx, r in enumerate(repos, 1):
        lines.append(f"[{idx}] {r['name']} (Branch: {r['branch']}) - Path: {r['path']}")
    return "\n".join(lines)

@mcp.tool()
def set_workspace_dir(
    new_workspace_path: str = Field(..., description="Path to the new workspace directory (can be comma-separated for multiple directories)")
) -> str:
    """Dynamically change the active workspace directory at runtime without restarting the server."""
    global _DYNAMIC_WORKSPACE_DIR, _LAST_ENV_MTIME
    if isinstance(new_workspace_path, FieldInfo):
        new_workspace_path = str(new_workspace_path.default)

    cleaned = str(new_workspace_path).strip()
    if not cleaned:
        return "Workspace path cannot be empty."

    paths = [p.strip() for p in cleaned.split(",") if p.strip()]
    missing = []
    for p in paths:
        target = Path(os.path.expanduser(p)).resolve()
        if not target.is_dir():
            missing.append(str(target))

    if missing:
        return f"Cannot change workspace. Directory does not exist: {', '.join(missing)}"

    _DYNAMIC_WORKSPACE_DIR = cleaned
    os.environ["WORKSPACE_DIR"] = cleaned

    try:
        _update_env_file("WORKSPACE_DIR", cleaned)
        env_path = ROOT_DIR / ".env"
        if env_path.exists():
            _LAST_ENV_MTIME = env_path.stat().st_mtime
    except Exception:
        pass

    primary_ws, _ = _get_workspace_dirs()
    repos = _discover_workspace_git_repos()
    repo_lines = (
        "\n".join([f"- [{i}] {r['name']} (Branch: {r['branch']}) - Path: {r['path']}" for i, r in enumerate(repos, 1)])
        if repos
        else "No Git repositories found in this workspace."
    )

    return (
        f"Workspace directory successfully updated to: {cleaned}\n"
        f"Primary Workspace: {primary_ws}\n"
        f"Discovered {len(repos)} Git repository(ies):\n{repo_lines}"
    )

    lines = ["Available Workspace Git Repositories:"]
    for idx, r in enumerate(repos, 1):
        lines.append(f"[{idx}] {r['name']} (Branch: {r['branch']}) - Path: {r['path']}")
    return "\n".join(lines)

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
    repo_path: Optional[str] = Field(None, description="Repository folder name inside WORKSPACE_DIR or full path. Defaults to current repository if omitted.")
) -> str:
    """Quickly check the current Git branch, the files being modified, and the 3 most recent commits."""
    if isinstance(repo_path, FieldInfo):
        repo_path = repo_path.default

    target_dir, error_msg = _resolve_git_repo(repo_path)
    if error_msg or not target_dir:
        return error_msg or f"Invalid repository: {repo_path}"

    try:
        status_res = subprocess.run(["git", "status", "-s"], cwd=str(target_dir), capture_output=True, text=True, timeout=10)
        log_res = subprocess.run(["git", "log", "-n", "3", "--oneline"], cwd=str(target_dir), capture_output=True, text=True, timeout=10)
        status = status_res.stdout.strip()
        log = log_res.stdout.strip()
        header = f"Repository: {target_dir.name} ({target_dir})\n"
        return f"{header}=== Git Status ===\n{status or 'Working tree clean'}\n=== Recent Commits ===\n{log or 'No commits yet'}"
    except Exception as e:
        return f"Unable to read Git information at {target_dir}: {str(e)}"

@mcp.tool()
def get_local_git_diff(
    repo_path: Optional[str] = Field(None, description="Repository folder name inside WORKSPACE_DIR or full path. Defaults to current repository if omitted."),
    staged_only: bool = Field(False, description="Whether to show only staged changes (git diff --staged)"),
    max_chars: int = Field(8000, description="Maximum characters of diff output to return before truncating"),
) -> str:
    """Inspect local git diff for unstaged or staged changes."""
    if isinstance(repo_path, FieldInfo):
        repo_path = repo_path.default
    if isinstance(staged_only, FieldInfo):
        staged_only = bool(staged_only.default)
    if isinstance(max_chars, FieldInfo):
        max_chars = int(max_chars.default)

    target_dir, error_msg = _resolve_git_repo(repo_path)
    if error_msg or not target_dir:
        return error_msg or f"Invalid repository: {repo_path}"

    cmd = ["git", "diff", "--staged"] if staged_only else ["git", "diff"]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"Git diff failed: {result.stderr.strip()}"

        output = result.stdout.strip()
        change_type = "staged" if staged_only else "unstaged"
        header = f"Repository: {target_dir.name} ({target_dir})\n"

        if not output:
            return f"{header}No {change_type} changes exist in the repository."

        if len(output) > max_chars:
            output = output[:max_chars] + f"\n\n[Diff truncated: exceeded {max_chars} characters limit.]"

        diff_section = f"=== Git Diff ({'Staged' if staged_only else 'Unstaged'}) ==="
        return f"{header}{diff_section}\n{output}"
    except Exception as e:
        return f"Unable to read git diff: {str(e)}"

@mcp.tool()
def get_recent_git_commits(
    repo_path: Optional[str] = Field(None, description="Repository folder name inside WORKSPACE_DIR or full path. Defaults to current repository if omitted."),
    limit: int = Field(5, description="Number of recent commits to retrieve"),
) -> str:
    """Inspect recent git commit history with summary statistics."""
    if isinstance(repo_path, FieldInfo):
        repo_path = repo_path.default
    if isinstance(limit, FieldInfo):
        limit = int(limit.default)

    target_dir, error_msg = _resolve_git_repo(repo_path)
    if error_msg or not target_dir:
        return error_msg or f"Invalid repository: {repo_path}"

    try:
        result = subprocess.run(
            ["git", "log", f"-n{limit}", "--oneline", "--stat"],
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"Git log failed: {result.stderr.strip()}"

        output = result.stdout.strip()
        header = f"Repository: {target_dir.name} ({target_dir})\n"
        if not output:
            return f"{header}No commit history found in the repository."

        return f"{header}=== Recent Commits (Last {limit}) ===\n{output}"
    except Exception as e:
        return f"Unable to read git commit history: {str(e)}"

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