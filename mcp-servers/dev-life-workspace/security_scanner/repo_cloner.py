import os
import shutil
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dotenv import dotenv_values, load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

MAX_ALLOWED_BYTES = 150 * 1024 * 1024  # 150 MB
MAX_ALLOWED_FILES = 10000

REMOTE_PREFIXES = ("https://", "http://", "git@", "ssh://", "file://")
DANGEROUS_REMOTE_CHARS = (";", "|", "&", "`", "$", "\n", "\r", "\t", "\x00")


def _get_workspace_dirs() -> Tuple[Path, List[Path]]:
    """Dynamically resolve current workspace directory from .env or fallback."""
    env_path = ROOT_DIR / ".env"
    raw_workspace = ""

    if env_path.exists():
        try:
            env_vals = dotenv_values(env_path)
            val = env_vals.get("WORKSPACE_DIR")
            if val:
                raw_workspace = val.strip()
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


def inspect_repo_directory(
    directory: Path,
    max_bytes: int = MAX_ALLOWED_BYTES,
    max_files: int = MAX_ALLOWED_FILES,
) -> Tuple[int, int]:
    """Calculate total file count and size in bytes excluding .git directory.

    Raises ValueError if maximum size or file count is exceeded.
    """
    total_bytes = 0
    file_count = 0

    for root, dirs, files in os.walk(directory):
        if ".git" in dirs:
            dirs.remove(".git")
        for f in files:
            file_count += 1
            if file_count > max_files:
                raise ValueError(
                    f"Repository exceeds maximum allowed file count ({max_files} files)."
                )
            file_path = Path(root) / f
            try:
                if not file_path.is_symlink():
                    total_bytes += file_path.stat().st_size
            except (OSError, PermissionError):
                pass
            if total_bytes > max_bytes:
                raise ValueError(
                    f"Repository exceeds maximum allowed size ({max_bytes / (1024 * 1024):.1f} MB)."
                )

    return file_count, total_bytes


def get_commit_hash(repo_path: Path) -> str:
    """Extract latest HEAD commit hash safely or return 'untracked_directory'."""
    try:
        res = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
            if commit:
                return commit
    except Exception:
        pass
    return "untracked_directory"


class SafeRepoCloner:
    """Hardened repository preparation engine for local workspace repos and remote Git URLs."""

    def __init__(self, quarantine_dir: Optional[Path] = None):
        if quarantine_dir:
            self.quarantine_dir = Path(quarantine_dir).resolve()
        else:
            primary_ws, _ = _get_workspace_dirs()
            preferred_quarantine = primary_ws / ".audit_quarantine"
            try:
                preferred_quarantine.mkdir(parents=True, exist_ok=True)
                test_file = preferred_quarantine / ".write_test"
                test_file.touch()
                test_file.unlink()
                self.quarantine_dir = preferred_quarantine
            except Exception:
                fallback_quarantine = Path("/tmp/ai_os_audit_quarantine")
                fallback_quarantine.mkdir(parents=True, exist_ok=True)
                self.quarantine_dir = fallback_quarantine

        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def is_remote_url(self, repo_source: str) -> bool:
        """Check if repo_source matches a remote Git URL pattern."""
        cleaned = repo_source.strip()
        return any(cleaned.startswith(prefix) for prefix in REMOTE_PREFIXES)

    def prepare_repository(self, repo_source: str) -> Dict[str, Any]:
        """Prepare and validate target repository safely without executing any repository code."""
        cleaned_source = str(repo_source).strip()
        if not cleaned_source:
            raise ValueError("Repository source cannot be empty.")

        if self.is_remote_url(cleaned_source):
            return self._prepare_remote_repo(cleaned_source)
        else:
            return self._prepare_local_repo(cleaned_source)

    def _prepare_remote_repo(self, repo_source: str) -> Dict[str, Any]:
        """Safely clone remote Git repository into quarantine with hardened flags."""
        if repo_source.startswith("-"):
            raise ValueError("Security violation: Git repository source cannot start with a dash.")

        if any(char in repo_source for char in DANGEROUS_REMOTE_CHARS):
            raise ValueError("Security violation: Invalid or dangerous characters in git URL.")

        if "ext::" in repo_source.lower():
            raise ValueError("Security violation: Dangerous git protocol 'ext::' is forbidden.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_hash = hashlib.sha256(repo_source.encode()).hexdigest()[:8]
        target_dir = self.quarantine_dir / f"repo_{timestamp}_{short_hash}"

        clone_env = os.environ.copy()
        clone_env["GIT_TERMINAL_PROMPT"] = "0"

        clone_cmd = [
            "git",
            "-c", "safe.directory=*",
            "-c", "core.hooksPath=/dev/null",
            "-c", "core.symlinks=false",
            "-c", "submodule.recurse=false",
            "clone",
            "--depth", "1",
            "--single-branch",
            "--no-tags",
            repo_source,
            str(target_dir),
        ]

        try:
            result = subprocess.run(
                clone_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=clone_env,
            )
            if result.returncode != 0:
                if target_dir.exists():
                    shutil.rmtree(target_dir, ignore_errors=True)
                raise RuntimeError(f"Git clone failed: {result.stderr.strip()}")

            file_count, total_bytes = inspect_repo_directory(target_dir)
            commit_hash = get_commit_hash(target_dir)

            return {
                "repo_source": repo_source,
                "source_type": "remote_url",
                "resolved_local_path": str(target_dir),
                "commit_hash": commit_hash,
                "file_count": file_count,
                "total_bytes": total_bytes,
            }
        except Exception:
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            raise

    def _prepare_local_repo(self, repo_source: str) -> Dict[str, Any]:
        """Validate and resolve local repository inside allowed workspace paths."""
        if repo_source.startswith("-"):
            raise ValueError("Security violation: Repository path cannot start with a dash.")

        if any(char in repo_source for char in (";", "|", "&", "`", "$", "\x00")):
            raise ValueError("Security violation: Invalid characters in repository path.")

        primary_ws, workspace_dirs = _get_workspace_dirs()
        allowed_bases = [ws.resolve() for ws in workspace_dirs]
        test_dir = Path("/tmp/ai_os_audit_test").resolve()
        allowed_bases.append(test_dir)

        # Resolve candidate path
        if os.path.isabs(repo_source):
            candidate = Path(repo_source).resolve()
        else:
            if (primary_ws / repo_source).exists():
                candidate = (primary_ws / repo_source).resolve()
            elif (test_dir / repo_source).exists():
                candidate = (test_dir / repo_source).resolve()
            else:
                candidate = (primary_ws / repo_source).resolve()

        # Strict Path Traversal Guard
        is_inside_allowed = False
        for base in allowed_bases:
            try:
                if candidate.is_relative_to(base):
                    is_inside_allowed = True
                    break
            except (ValueError, AttributeError):
                pass

        if not is_inside_allowed:
            raise ValueError("Security violation: Target path is outside the allowed workspace directory.")

        if not candidate.exists():
            raise FileNotFoundError(f"Target repository does not exist: {repo_source}")

        if not candidate.is_dir():
            raise ValueError(f"Target repository path is not a directory: {repo_source}")

        file_count, total_bytes = inspect_repo_directory(candidate)
        if file_count == 0:
            raise ValueError(f"Target repository '{repo_source}' contains no files.")

        commit_hash = get_commit_hash(candidate)

        return {
            "repo_source": repo_source,
            "source_type": "local_workspace",
            "resolved_local_path": str(candidate),
            "commit_hash": commit_hash,
            "file_count": file_count,
            "total_bytes": total_bytes,
        }
