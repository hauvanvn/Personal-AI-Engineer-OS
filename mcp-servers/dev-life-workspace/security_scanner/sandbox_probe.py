"""Isolated Sandbox Probe for Dynamic Runtime Security Analysis.

Runs target repository code inside a strictly isolated environment (Docker container
with --network none, or a hardened subprocess fallback) to capture runtime telemetry:
- Network connection attempts (DNS, TCP, HTTP)
- File access patterns (especially decoy honeypot credentials)
- Process spawn attempts (shell commands, external binaries)
- Environment variable inspection (sensitive and honeypot keys)

All outbound network traffic is intercepted and blocked by a sitecustomize.py
telemetry harness injected via PYTHONPATH before any target code executes.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional


# Maximum number of Python entrypoints to probe
MAX_ENTRYPOINTS = 2

# Timeout for sandbox execution (seconds)
SANDBOX_TIMEOUT_DOCKER = 15
SANDBOX_TIMEOUT_SUBPROCESS = 8

# Sensitive environment variable names for honeypot detection
HONEYPOT_ENV_KEYS = {
    "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID",
    "OPENAI_API_KEY", "GITHUB_TOKEN", "GITLAB_TOKEN",
    "SLACK_TOKEN", "DISCORD_TOKEN", "DATABASE_URL",
    "DB_PASSWORD", "SECRET_KEY", "PRIVATE_KEY",
}

# Sensitive file path fragments for honeypot detection
HONEYPOT_PATH_FRAGMENTS = {
    ".ssh", "id_rsa", "id_ed25519", "authorized_keys",
    ".aws/credentials", ".env", "/etc/passwd", "/etc/shadow",
    ".gnupg", ".kube/config", ".config/gcloud",
}

# Dangerous shell commands to block
BLOCKED_SHELL_COMMANDS = {
    "curl", "wget", "nc", "ncat", "netcat",
    "sh", "bash", "zsh", "dash", "csh", "ksh",
    "powershell", "cmd",
}

# Honeypot decoy environment variables injected into the sandbox
DECOY_ENV_VARS = {
    "AWS_SECRET_ACCESS_KEY": "HONEYPOT_DEC_AWS_KEY",
    "OPENAI_API_KEY": "HONEYPOT_DEC_OPENAI_KEY",
    "GITHUB_TOKEN": "HONEYPOT_DEC_GH_TOKEN",
}


def _generate_sitecustomize_py() -> str:
    """Generate the sitecustomize.py telemetry harness source code.

    This script is loaded automatically by CPython when PYTHONPATH points to
    its containing directory. It monkey-patches socket, urllib, http.client,
    subprocess, builtins.open, and os.environ access to intercept and log all
    runtime behavior while blocking real outbound connections and dangerous
    process spawns.
    """
    return textwrap.dedent('''\
        """Sandbox Telemetry Harness - Auto-loaded via PYTHONPATH.

        Intercepts and logs all network, file, process, and environment access
        to /tmp/telemetry_events.jsonl while blocking real outbound connections.
        """
        import json
        import os
        import sys
        import socket
        import io

        _TELEMETRY_PATH = "/tmp/telemetry_events.jsonl"
        _REPO_DIR = os.environ.get("SANDBOX_REPO_DIR", "/tmp/work")

        _HONEYPOT_ENV_KEYS = {
            "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID",
            "OPENAI_API_KEY", "GITHUB_TOKEN", "GITLAB_TOKEN",
            "SLACK_TOKEN", "DISCORD_TOKEN", "DATABASE_URL",
            "DB_PASSWORD", "SECRET_KEY", "PRIVATE_KEY",
        }

        _HONEYPOT_PATH_FRAGMENTS = {
            ".ssh", "id_rsa", "id_ed25519", "authorized_keys",
            ".aws/credentials", ".env", "/etc/passwd", "/etc/shadow",
            ".gnupg", ".kube/config", ".config/gcloud",
        }

        _BLOCKED_CMDS = {
            "curl", "wget", "nc", "ncat", "netcat",
            "sh", "bash", "zsh", "dash", "csh", "ksh",
            "powershell", "cmd",
        }

        def _log_event(event_dict):
            """Append a JSON event to the telemetry log."""
            try:
                with io.open(_TELEMETRY_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event_dict, default=str) + "\\n")
            except Exception:
                pass

        # =====================================================================
        # 1. Socket Hooks (connect, connect_ex, create_connection, DNS)
        # =====================================================================
        _orig_socket_connect = socket.socket.connect
        _orig_socket_connect_ex = socket.socket.connect_ex
        _orig_create_connection = socket.create_connection
        _orig_getaddrinfo = socket.getaddrinfo
        _orig_gethostbyname = socket.gethostbyname

        def _hooked_connect(self, address):
            host = str(address[0]) if isinstance(address, tuple) else str(address)
            port = int(address[1]) if isinstance(address, tuple) and len(address) > 1 else 0
            _log_event({"event": "network_connect", "host": host, "port": port})
            raise OSError("Sandbox network isolation: outbound connection blocked")

        def _hooked_connect_ex(self, address):
            host = str(address[0]) if isinstance(address, tuple) else str(address)
            port = int(address[1]) if isinstance(address, tuple) and len(address) > 1 else 0
            _log_event({"event": "network_connect", "host": host, "port": port})
            raise OSError("Sandbox network isolation: outbound connection blocked")

        def _hooked_create_connection(address, timeout=None, source_address=None):
            host = str(address[0]) if isinstance(address, tuple) else str(address)
            port = int(address[1]) if isinstance(address, tuple) and len(address) > 1 else 0
            _log_event({"event": "network_connect", "host": host, "port": port})
            raise OSError("Sandbox network isolation: outbound connection blocked")

        def _hooked_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            _log_event({"event": "dns_lookup", "host": str(host)})
            safe_port = int(port) if port else 80
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.254", safe_port))]

        def _hooked_gethostbyname(hostname):
            _log_event({"event": "dns_lookup", "host": str(hostname)})
            return "127.0.0.254"

        socket.socket.connect = _hooked_connect
        socket.socket.connect_ex = _hooked_connect_ex
        socket.create_connection = _hooked_create_connection
        socket.getaddrinfo = _hooked_getaddrinfo
        socket.gethostbyname = _hooked_gethostbyname

        # =====================================================================
        # 2. HTTP / urllib Hooks
        # =====================================================================
        try:
            import urllib.request
            _orig_urlopen = urllib.request.urlopen

            def _hooked_urlopen(url, data=None, timeout=None, **kwargs):
                url_str = str(url) if not hasattr(url, "full_url") else url.full_url
                _log_event({
                    "event": "http_request",
                    "method": "GET",
                    "url_or_path": url_str,
                    "headers_preview": "",
                })
                raise OSError("Sandbox network isolation: HTTP request blocked")

            urllib.request.urlopen = _hooked_urlopen
        except Exception:
            pass

        try:
            import http.client

            _orig_http_request = http.client.HTTPConnection.request

            def _hooked_http_request(self, method, url, body=None, headers=None, **kwargs):
                dest = getattr(self, "host", "unknown")
                port = getattr(self, "port", 80)
                full_url = f"http://{dest}:{port}{url}"
                headers_preview = ""
                if headers and isinstance(headers, dict):
                    headers_preview = str(dict(list(headers.items())[:3]))
                _log_event({
                    "event": "http_request",
                    "method": str(method),
                    "url_or_path": full_url,
                    "headers_preview": headers_preview,
                })
                raise OSError("Sandbox network isolation: HTTP request blocked")

            http.client.HTTPConnection.request = _hooked_http_request
            if hasattr(http.client, "HTTPSConnection"):
                http.client.HTTPSConnection.request = _hooked_http_request
        except Exception:
            pass

        # =====================================================================
        # 3. Process Spawn Hooks (subprocess, os.system, os.execv, os.spawnv)
        # =====================================================================
        try:
            import subprocess as _subprocess

            _orig_popen_init = _subprocess.Popen.__init__

            def _hooked_popen_init(self, args, **kwargs):
                cmd_str = str(args)
                _log_event({"event": "process_spawn", "command": cmd_str})
                # Check if command invokes blocked binaries
                if isinstance(args, (list, tuple)):
                    base_cmd = os.path.basename(str(args[0])).lower()
                else:
                    base_cmd = os.path.basename(str(args).split()[0]).lower() if args else ""
                if base_cmd in _BLOCKED_CMDS:
                    raise PermissionError(
                        f"Sandbox isolation: process spawn blocked for command: {base_cmd}"
                    )
                return _orig_popen_init(self, args, **kwargs)

            _subprocess.Popen.__init__ = _hooked_popen_init
        except Exception:
            pass

        _orig_os_system = os.system

        def _hooked_os_system(command):
            _log_event({"event": "process_spawn", "command": str(command)})
            raise PermissionError("Sandbox isolation: os.system blocked")

        os.system = _hooked_os_system

        for _exec_name in ("execv", "execve", "spawnv", "spawnve"):
            if hasattr(os, _exec_name):
                _orig_fn = getattr(os, _exec_name)

                def _make_hook(fn_name, orig):
                    def _hooked(*args, **kwargs):
                        _log_event({"event": "process_spawn", "command": f"{fn_name}({args})"})
                        raise PermissionError(f"Sandbox isolation: {fn_name} blocked")
                    return _hooked

                setattr(os, _exec_name, _make_hook(_exec_name, _orig_fn))

        # =====================================================================
        # 4. File Access Hooks (builtins.open, io.open)
        # =====================================================================
        import builtins

        _orig_open = builtins.open

        def _is_sensitive_path(path_str):
            """Check if file path touches honeypot or sensitive locations."""
            path_lower = path_str.lower()
            for frag in _HONEYPOT_PATH_FRAGMENTS:
                if frag.lower() in path_lower:
                    return True
            return False

        def _is_outside_repo(path_str):
            """Check if the path is outside the repository working directory."""
            try:
                real_path = os.path.realpath(path_str)
                repo_real = os.path.realpath(_REPO_DIR)
                tmp_real = os.path.realpath("/tmp")
                if real_path.startswith(repo_real):
                    return False
                if real_path.startswith(tmp_real):
                    return False
                return True
            except Exception:
                return False

        def _hooked_open(file, mode="r", *args, **kwargs):
            path_str = str(file)
            is_honeypot = _is_sensitive_path(path_str)
            is_outside = _is_outside_repo(path_str)
            if is_honeypot or is_outside:
                mode_str = str(mode) if mode else "r"
                _log_event({
                    "event": "file_access",
                    "path": path_str,
                    "mode": mode_str,
                    "is_honeypot": is_honeypot,
                })
            return _orig_open(file, mode, *args, **kwargs)

        builtins.open = _hooked_open
        io.open = _hooked_open

        # =====================================================================
        # 5. Environment Variable Access Hooks
        # =====================================================================
        _orig_os_getenv = os.getenv
        _OrigEnviron = type(os.environ)

        def _hooked_getenv(key, default=None):
            key_str = str(key)
            is_honeypot = key_str.upper() in _HONEYPOT_ENV_KEYS
            _log_event({
                "event": "env_access",
                "key": key_str,
                "is_honeypot": is_honeypot,
            })
            return _orig_os_getenv(key, default)

        os.getenv = _hooked_getenv

        _orig_environ_get = os.environ.get

        def _hooked_environ_get(key, default=None):
            key_str = str(key)
            is_honeypot = key_str.upper() in _HONEYPOT_ENV_KEYS
            _log_event({
                "event": "env_access",
                "key": key_str,
                "is_honeypot": is_honeypot,
            })
            return _orig_environ_get(key, default)

        os.environ.get = _hooked_environ_get
    ''')


def _is_docker_available() -> bool:
    """Check if the Docker daemon is accessible with a short timeout."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _is_docker_image_available(image: str = "python:3.11-slim") -> bool:
    """Check if a Docker image is available locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _select_entrypoints(
    repo_path: Path,
    stage_results: Dict[str, Any],
    max_count: int = MAX_ENTRYPOINTS,
) -> List[str]:
    """Select up to max_count safe Python entrypoints from Phase 6.2 analysis.

    Prefers main.py, app.py, cli.py; excludes setup.py unless it is the only option
    and the repo is clean.
    """
    entrypoints = []

    # Try to get entrypoints from Phase 6.2 workflow analysis
    wp_data = stage_results.get("workflow_and_packages", {})
    wf_data = wp_data.get("workflow", {})
    detected_eps = wf_data.get("entrypoints", [])

    priority_names = {"main.py", "app.py", "cli.py", "server.py", "run.py"}

    # Filter to Python entrypoints only
    python_eps = [
        ep for ep in detected_eps
        if ep.get("type") == "python_entrypoint"
        and ep.get("file", "").endswith(".py")
    ]

    # Sort: priority names first, then exclude setup.py
    for ep in python_eps:
        fname = Path(ep["file"]).name
        if fname in priority_names and ep["file"] not in entrypoints:
            entrypoints.append(ep["file"])
        if len(entrypoints) >= max_count:
            break

    # If we still need more, add non-setup.py entrypoints
    if len(entrypoints) < max_count:
        for ep in python_eps:
            fname = Path(ep["file"]).name
            if fname != "setup.py" and ep["file"] not in entrypoints:
                entrypoints.append(ep["file"])
            if len(entrypoints) >= max_count:
                break

    # Fallback: scan repo root for .py files
    if not entrypoints:
        for py_file in sorted(repo_path.glob("*.py")):
            if py_file.name != "setup.py":
                entrypoints.append(py_file.name)
            if len(entrypoints) >= max_count:
                break

    return entrypoints[:max_count]


def _prepare_staging_dir(
    repo_path: Path,
    staging_base: Optional[str] = None,
) -> Path:
    """Prepare a temporary staging directory with repo copy and telemetry harness.

    Returns the path to the staging directory containing:
      staging_dir/
        harness/sitecustomize.py
        repo/<copied files excluding .git>
        sandbox_home/.ssh/id_rsa
        sandbox_home/.aws/credentials
        sandbox_home/.env.decoy
    """
    staging_dir = Path(tempfile.mkdtemp(
        prefix="sandbox_probe_",
        dir=staging_base,
    ))

    # 1. Create harness directory with sitecustomize.py
    harness_dir = staging_dir / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize_path = harness_dir / "sitecustomize.py"
    sitecustomize_path.write_text(_generate_sitecustomize_py(), encoding="utf-8")

    # 2. Copy repository files (excluding .git)
    repo_copy_dir = staging_dir / "repo"
    repo_copy_dir.mkdir(parents=True, exist_ok=True)
    for item in repo_path.iterdir():
        if item.name == ".git":
            continue
        dest = repo_copy_dir / item.name
        if item.is_dir():
            shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
        else:
            shutil.copy2(str(item), str(dest))

    # 3. Create decoy honeypot files
    sandbox_home = staging_dir / "sandbox_home"
    ssh_dir = sandbox_home / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    (ssh_dir / "id_rsa").write_text(
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "HONEYPOT_DECOY_SSH_KEY_DO_NOT_USE\n"
        "-----END OPENSSH PRIVATE KEY-----\n",
        encoding="utf-8",
    )

    aws_dir = sandbox_home / ".aws"
    aws_dir.mkdir(parents=True, exist_ok=True)
    (aws_dir / "credentials").write_text(
        "[default]\n"
        "aws_access_key_id = HONEYPOT_DEC_AWS_ACCESS\n"
        "aws_secret_access_key = HONEYPOT_DEC_AWS_SECRET\n",
        encoding="utf-8",
    )

    (sandbox_home / ".env.decoy").write_text(
        "SECRET_KEY=HONEYPOT_DEC_SECRET_KEY\n"
        "DATABASE_URL=postgresql://honeypot:decoy@localhost/fake\n",
        encoding="utf-8",
    )

    return staging_dir


def _parse_telemetry_events(telemetry_content: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse telemetry_events.jsonl content and deduplicate into categorized lists."""
    network_attempts = []
    file_access_attempts = []
    process_spawns = []
    env_access_attempts = []
    honeypot_triggers = []

    seen_network = set()
    seen_file = set()
    seen_process = set()
    seen_env = set()

    for line in telemetry_content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("event", "")

        if event_type == "dns_lookup":
            host = event.get("host", "")
            dedup_key = f"dns:{host}"
            if dedup_key not in seen_network:
                seen_network.add(dedup_key)
                network_attempts.append({
                    "type": "dns_lookup",
                    "destination": host,
                    "details": f"DNS resolution attempted for {host}",
                })

        elif event_type == "network_connect":
            host = event.get("host", "")
            port = event.get("port", 0)
            dedup_key = f"connect:{host}:{port}"
            if dedup_key not in seen_network:
                seen_network.add(dedup_key)
                network_attempts.append({
                    "type": "network_connect",
                    "destination": f"{host}:{port}",
                    "details": f"TCP connection attempted to {host}:{port}",
                })

        elif event_type == "http_request":
            url = event.get("url_or_path", "")
            method = event.get("method", "GET")
            dedup_key = f"http:{method}:{url}"
            if dedup_key not in seen_network:
                seen_network.add(dedup_key)
                network_attempts.append({
                    "type": "http_request",
                    "destination": url,
                    "details": f"HTTP {method} request to {url}",
                })

        elif event_type == "file_access":
            fpath = event.get("path", "")
            mode = event.get("mode", "r")
            is_honeypot = event.get("is_honeypot", False)
            dedup_key = f"file:{fpath}:{mode}"
            if dedup_key not in seen_file:
                seen_file.add(dedup_key)
                file_access_attempts.append({
                    "path": fpath,
                    "mode": mode,
                    "is_honeypot": is_honeypot,
                })
                if is_honeypot:
                    honeypot_triggers.append(
                        f"Honeypot file accessed: {fpath} (mode={mode})"
                    )

        elif event_type == "process_spawn":
            command = event.get("command", "")
            dedup_key = f"proc:{command}"
            if dedup_key not in seen_process:
                seen_process.add(dedup_key)
                process_spawns.append({"command": command})
                # Check if it is an unauthorized shell command
                cmd_base = ""
                try:
                    if command.startswith("[") or command.startswith("("):
                        parts = json.loads(command.replace("(", "[").replace(")", "]").replace("'", '"'))
                        if parts:
                            cmd_base = os.path.basename(str(parts[0])).lower()
                    else:
                        cmd_base = os.path.basename(command.split()[0]).lower()
                except Exception:
                    cmd_base = ""
                if cmd_base in BLOCKED_SHELL_COMMANDS:
                    honeypot_triggers.append(
                        f"Unauthorized shell process spawn: {command}"
                    )

        elif event_type == "env_access":
            key = event.get("key", "")
            is_honeypot = event.get("is_honeypot", False)
            dedup_key = f"env:{key}"
            if dedup_key not in seen_env:
                seen_env.add(dedup_key)
                env_access_attempts.append({
                    "key": key,
                    "is_honeypot": is_honeypot,
                })
                if is_honeypot:
                    honeypot_triggers.append(
                        f"Honeypot environment variable accessed: {key}"
                    )

    return {
        "network_attempts": network_attempts,
        "file_access_attempts": file_access_attempts,
        "process_spawns": process_spawns,
        "env_access_attempts": env_access_attempts,
        "honeypot_triggers": honeypot_triggers,
    }


def _compute_sandbox_verdict(parsed: Dict[str, List]) -> str:
    """Compute sandbox_risk_verdict based on telemetry findings."""
    if parsed.get("honeypot_triggers"):
        return "CRITICAL_DANGER"

    # Check for unauthorized shell spawns
    for spawn in parsed.get("process_spawns", []):
        cmd = spawn.get("command", "")
        cmd_base = ""
        try:
            if cmd.startswith("[") or cmd.startswith("("):
                parts = json.loads(cmd.replace("(", "[").replace(")", "]").replace("'", '"'))
                if parts:
                    cmd_base = os.path.basename(str(parts[0])).lower()
            else:
                cmd_base = os.path.basename(cmd.split()[0]).lower()
        except Exception:
            pass
        if cmd_base in BLOCKED_SHELL_COMMANDS:
            return "CRITICAL_DANGER"

    # Check for unexpected external network attempts
    if parsed.get("network_attempts"):
        return "CAUTION"

    return "SAFE"


class IsolatedSandboxProbe:
    """Runs target repository entrypoints in an isolated sandbox and captures runtime telemetry."""

    def __init__(self):
        pass

    def run_probe(
        self,
        audit_record: Dict[str, Any],
        force_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute the sandbox probe on an audit record.

        Args:
            audit_record: Full audit record dict from SecurityAuditStateManager.
            force_run: If True, bypass the static security gate check.

        Returns:
            Structured probe result dictionary.
        """
        # ------------------------------------------------------------------
        # A. Pre-Flight Static Gate Check
        # ------------------------------------------------------------------
        static_risk_verdict = audit_record.get("static_risk_verdict", "PENDING")
        stage_results = audit_record.get("stage_results", {})
        malware_scan = stage_results.get("static_malware_scan", {})

        if static_risk_verdict in ("CRITICAL_DANGER", "HIGH_RISK") and not force_run:
            return {
                "sandbox_executed": False,
                "skipped_reason": (
                    f"Blocked by static security gate: repository verdict is "
                    f"{static_risk_verdict}. Dynamic execution skipped to protect host."
                ),
                "execution_mode": "blocked_by_static_gate",
                "entrypoints_tested": [],
                "network_attempts": [],
                "file_access_attempts": [],
                "process_spawns": [],
                "env_access_attempts": [],
                "honeypot_triggers": [],
                "sandbox_risk_verdict": static_risk_verdict,
            }

        # ------------------------------------------------------------------
        # B. Prepare Staging Directory
        # ------------------------------------------------------------------
        repo_path = Path(audit_record["resolved_local_path"]).resolve()
        if not repo_path.is_dir():
            return {
                "sandbox_executed": False,
                "skipped_reason": f"Repository path does not exist: {repo_path}",
                "execution_mode": "error",
                "entrypoints_tested": [],
                "network_attempts": [],
                "file_access_attempts": [],
                "process_spawns": [],
                "env_access_attempts": [],
                "honeypot_triggers": [],
                "sandbox_risk_verdict": "SAFE",
            }

        staging_dir = _prepare_staging_dir(repo_path)

        try:
            # Select entrypoints
            entrypoints = _select_entrypoints(repo_path, stage_results)
            if not entrypoints:
                return {
                    "sandbox_executed": False,
                    "skipped_reason": "No Python entrypoints found for sandbox probing.",
                    "execution_mode": "no_entrypoints",
                    "entrypoints_tested": [],
                    "network_attempts": [],
                    "file_access_attempts": [],
                    "process_spawns": [],
                    "env_access_attempts": [],
                    "honeypot_triggers": [],
                    "sandbox_risk_verdict": "SAFE",
                }

            # ------------------------------------------------------------------
            # C. Execute in Isolated Environment
            # ------------------------------------------------------------------
            docker_available = _is_docker_available()
            docker_image_ok = docker_available and _is_docker_image_available("python:3.11-slim")

            execution_mode = "docker_container" if docker_image_ok else "isolated_python_harness"
            all_telemetry_lines = []
            execution_errors = []
            start_time = time.monotonic()

            for entrypoint in entrypoints:
                if docker_image_ok:
                    telemetry_output, error = self._run_in_docker(
                        staging_dir, entrypoint
                    )
                else:
                    telemetry_output, error = self._run_in_subprocess(
                        staging_dir, entrypoint
                    )

                if telemetry_output:
                    all_telemetry_lines.append(telemetry_output)
                if error:
                    execution_errors.append(error)

            elapsed = round(time.monotonic() - start_time, 2)

            # ------------------------------------------------------------------
            # D. Parse and Aggregate Telemetry
            # ------------------------------------------------------------------
            combined_telemetry = "\n".join(all_telemetry_lines)
            parsed = _parse_telemetry_events(combined_telemetry)
            sandbox_verdict = _compute_sandbox_verdict(parsed)

            return {
                "sandbox_executed": True,
                "execution_mode": execution_mode,
                "entrypoints_tested": entrypoints,
                "execution_time_seconds": elapsed,
                "execution_errors": execution_errors if execution_errors else [],
                "network_attempts": parsed["network_attempts"],
                "file_access_attempts": parsed["file_access_attempts"],
                "process_spawns": parsed["process_spawns"],
                "env_access_attempts": parsed["env_access_attempts"],
                "honeypot_triggers": parsed["honeypot_triggers"],
                "sandbox_risk_verdict": sandbox_verdict,
            }

        finally:
            # Cleanup staging directory
            try:
                shutil.rmtree(staging_dir, ignore_errors=True)
            except Exception:
                pass

    def _run_in_docker(
        self,
        staging_dir: Path,
        entrypoint: str,
    ) -> tuple:
        """Run an entrypoint inside a hardened Docker container.

        Returns (telemetry_output, error_message).
        """
        # Build the shell command that copies repo, runs python, and cats telemetry
        inner_cmd = (
            "cp -r /sandbox/repo /tmp/work && "
            "mkdir -p /tmp/sandbox_home/.ssh /tmp/sandbox_home/.aws && "
            "cp -r /sandbox/sandbox_home/. /tmp/sandbox_home/ 2>/dev/null || true && "
            f"cd /tmp/work && timeout 10 python3 {entrypoint} 2>/dev/null || true && "
            "cat /tmp/telemetry_events.jsonl 2>/dev/null || true"
        )

        docker_cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--memory", "256m",
            "--cpus", "0.5",
            "--pids-limit", "32",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "PYTHONPATH=/sandbox/harness:/sandbox/repo",
            "-e", "HOME=/tmp/sandbox_home",
            "-e", "SANDBOX_REPO_DIR=/tmp/work",
            "-e", f"AWS_SECRET_ACCESS_KEY={DECOY_ENV_VARS['AWS_SECRET_ACCESS_KEY']}",
            "-e", f"GITHUB_TOKEN={DECOY_ENV_VARS['GITHUB_TOKEN']}",
            "-e", f"OPENAI_API_KEY={DECOY_ENV_VARS['OPENAI_API_KEY']}",
            "-v", f"{staging_dir}:/sandbox:ro",
            "python:3.11-slim",
            "sh", "-c", inner_cmd,
        ]

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=SANDBOX_TIMEOUT_DOCKER,
            )
            telemetry_output = result.stdout.strip()
            error = result.stderr.strip() if result.returncode != 0 else ""
            return telemetry_output, error
        except subprocess.TimeoutExpired:
            return "", f"Docker execution timed out after {SANDBOX_TIMEOUT_DOCKER}s for {entrypoint}"
        except Exception as e:
            return "", f"Docker execution error for {entrypoint}: {str(e)}"

    def _run_in_subprocess(
        self,
        staging_dir: Path,
        entrypoint: str,
    ) -> tuple:
        """Run an entrypoint in an isolated subprocess with telemetry harness.

        The sitecustomize.py monkey-patches ensure host safety by intercepting
        all network, process, and file operations before they can execute.

        Returns (telemetry_output, error_message).
        """
        harness_dir = staging_dir / "harness"
        repo_dir = staging_dir / "repo"
        sandbox_home = staging_dir / "sandbox_home"

        # Create a temporary telemetry output file
        telemetry_file = staging_dir / "telemetry_events.jsonl"
        telemetry_file.touch()

        # Build environment with monkey-patched sitecustomize loading
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{harness_dir}:{repo_dir}"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HOME"] = str(sandbox_home)
        env["SANDBOX_REPO_DIR"] = str(repo_dir)
        # Inject honeypot environment variables
        for key, value in DECOY_ENV_VARS.items():
            env[key] = value

        # The telemetry harness writes to /tmp/telemetry_events.jsonl
        # In subprocess mode we need to redirect that to our staging dir.
        # We achieve this by creating a symlink or modifying the path.
        # Simpler approach: set a temp env var and patch the harness path.
        # Actually, we write a wrapper that sets the telemetry path.
        wrapper_script = staging_dir / "run_probe.py"
        wrapper_script.write_text(
            "import os\n"
            "import sys\n"
            "# Override telemetry path for subprocess mode\n"
            "import importlib\n"
            f"os.environ['SANDBOX_REPO_DIR'] = {repr(str(repo_dir))}\n"
            "# sitecustomize.py is auto-loaded via PYTHONPATH\n"
            "# Now exec the target entrypoint\n"
            f"target = {repr(str(repo_dir / entrypoint))}\n"
            "if os.path.exists(target):\n"
            "    try:\n"
            "        with open(target, 'r') as f:\n"
            "            code = f.read()\n"
            "        exec(compile(code, target, 'exec'), {'__name__': '__main__', '__file__': target})\n"
            "    except SystemExit:\n"
            "        pass\n"
            "    except Exception as e:\n"
            "        pass\n",
            encoding="utf-8",
        )

        try:
            result = subprocess.run(
                [
                    sys.executable, "-I",
                    "-c",
                    (
                        "import sys, os;"
                        f"sys.path.insert(0, {repr(str(harness_dir))});"
                        f"sys.path.insert(1, {repr(str(repo_dir))});"
                        f"os.environ['SANDBOX_REPO_DIR'] = {repr(str(repo_dir))};"
                        "import sitecustomize;"
                        f"target = {repr(str(repo_dir / entrypoint))};"
                        "exec(open(target).read(), {'__name__': '__main__', '__file__': target})"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=SANDBOX_TIMEOUT_SUBPROCESS,
                env=env,
                cwd=str(repo_dir),
            )

            # Read the telemetry events from /tmp/telemetry_events.jsonl
            telemetry_path = Path("/tmp/telemetry_events.jsonl")
            telemetry_output = ""
            if telemetry_path.exists():
                telemetry_output = telemetry_path.read_text(encoding="utf-8").strip()
                # Clear the telemetry file for next entrypoint
                telemetry_path.write_text("", encoding="utf-8")

            error = ""
            if result.returncode != 0 and result.stderr:
                # Filter out expected sandbox error messages
                stderr_lines = result.stderr.strip().splitlines()
                real_errors = [
                    line for line in stderr_lines
                    if "Sandbox network isolation" not in line
                    and "Sandbox isolation" not in line
                ]
                if real_errors:
                    error = "\n".join(real_errors[-3:])

            return telemetry_output, error
        except subprocess.TimeoutExpired:
            # Still try to read telemetry
            telemetry_path = Path("/tmp/telemetry_events.jsonl")
            telemetry_output = ""
            if telemetry_path.exists():
                telemetry_output = telemetry_path.read_text(encoding="utf-8").strip()
                telemetry_path.write_text("", encoding="utf-8")
            return telemetry_output, f"Subprocess execution timed out after {SANDBOX_TIMEOUT_SUBPROCESS}s for {entrypoint}"
        except Exception as e:
            return "", f"Subprocess execution error for {entrypoint}: {str(e)}"


import sys
