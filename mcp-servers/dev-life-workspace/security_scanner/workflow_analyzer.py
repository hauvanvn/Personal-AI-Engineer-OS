import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

try:
    import yaml
except ImportError:
    yaml = None


IGNORE_DIRS: Set[str] = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".audit_quarantine",
    ".egg-info",
}

EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".mts": "TypeScript",
    ".cts": "TypeScript",
    ".jsx": "JavaScript (React)",
    ".tsx": "TypeScript (React)",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".go": "Go",
    ".rs": "Rust",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".hpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".json": "JSON",
    ".toml": "TOML",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "CSS",
    ".sass": "CSS",
    ".sql": "SQL",
}


class RepoWorkflowAnalyzer:
    """Statically analyzes repository directory structure, entrypoints, CI/CD, and execution flow."""

    def __init__(self, max_depth: int = 3, max_tree_entries: int = 100):
        self.max_depth = max_depth
        self.max_tree_entries = max_tree_entries

    def analyze(self, repo_path: Path) -> Dict[str, Any]:
        """Perform full static analysis on the target repository."""
        repo_path = repo_path.resolve()
        if not repo_path.is_dir():
            raise ValueError(f"Repository path is not a valid directory: {repo_path}")

        directory_tree = self._build_directory_tree(repo_path)
        language_breakdown = self._analyze_languages(repo_path)
        entrypoints = self._detect_entrypoints(repo_path)
        container_and_ci_configs = self._detect_container_and_ci(repo_path)
        module_dependency_graph = self._build_dependency_graph(repo_path, entrypoints)
        workflow_summary = self._generate_workflow_summary(
            language_breakdown=language_breakdown,
            entrypoints=entrypoints,
            container_and_ci_configs=container_and_ci_configs,
            module_dependency_graph=module_dependency_graph,
        )

        return {
            "directory_tree": directory_tree,
            "language_breakdown": language_breakdown,
            "entrypoints": entrypoints,
            "container_and_ci_configs": container_and_ci_configs,
            "module_dependency_graph": module_dependency_graph,
            "workflow_summary": workflow_summary,
        }

    def _should_skip_dir(self, dir_name: str) -> bool:
        """Check if a directory should be excluded from analysis."""
        return dir_name in IGNORE_DIRS or dir_name.endswith(".egg-info")

    def _build_directory_tree(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Build bounded directory tree summary up to max_depth and max_tree_entries."""
        tree: List[Dict[str, Any]] = []

        def _traverse(current_dir: Path, depth: int) -> None:
            if depth > self.max_depth or len(tree) >= self.max_tree_entries:
                return

            try:
                entries = sorted(current_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except (OSError, PermissionError):
                return

            for item in entries:
                if len(tree) >= self.max_tree_entries:
                    break
                if item.name.startswith(".") and item.name not in {".vscode", ".github", ".env"}:
                    if item.name == ".git":
                        continue
                if item.is_dir() and self._should_skip_dir(item.name):
                    continue

                rel_path = str(item.relative_to(repo_path))
                is_dir = item.is_dir()
                size = 0
                if not is_dir:
                    try:
                        size = item.stat().st_size
                    except (OSError, PermissionError):
                        size = 0

                tree.append({
                    "path": rel_path,
                    "type": "directory" if is_dir else "file",
                    "depth": depth,
                    "size": size,
                })

                if is_dir:
                    _traverse(item, depth + 1)

        _traverse(repo_path, 1)
        return tree

    def _analyze_languages(self, repo_path: Path) -> Dict[str, Dict[str, int]]:
        """Count files and lines of code by programming or configuration language."""
        breakdown: Dict[str, Dict[str, int]] = {}

        for root, dirs, files in os.walk(str(repo_path)):
            dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]
            for file_name in files:
                file_path = Path(root) / file_name
                lang = self._identify_language(file_name)
                if not lang:
                    continue

                line_count = 0
                try:
                    if file_path.stat().st_size < 2 * 1024 * 1024:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        line_count = len(content.splitlines())
                except (OSError, PermissionError):
                    line_count = 0

                if lang not in breakdown:
                    breakdown[lang] = {"files": 0, "lines": 0}
                breakdown[lang]["files"] += 1
                breakdown[lang]["lines"] += line_count

        return dict(sorted(breakdown.items(), key=lambda x: x[1]["lines"], reverse=True))

    def _identify_language(self, file_name: str) -> Optional[str]:
        """Map filename or extension to a canonical language name."""
        lower_name = file_name.lower()
        if lower_name == "dockerfile" or lower_name.startswith("dockerfile.") or lower_name.endswith(".dockerfile"):
            return "Dockerfile"
        if lower_name in {"makefile", "gnumakefile"} or lower_name.endswith(".mk"):
            return "Makefile"

        suffix = Path(file_name).suffix.lower()
        return EXTENSION_LANGUAGE_MAP.get(suffix)

    def _detect_entrypoints(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Identify primary application entrypoints for Python and Node.js."""
        entrypoints: List[Dict[str, Any]] = []

        # 1. Inspect package.json (Node.js entrypoint)
        package_json = repo_path / "package.json"
        if package_json.exists() and package_json.is_file():
            try:
                pkg_data = json.loads(package_json.read_text(encoding="utf-8", errors="ignore"))
                scripts = pkg_data.get("scripts", {})
                main_file = pkg_data.get("main", "")
                bin_spec = pkg_data.get("bin", "")
                entrypoints.append({
                    "file": "package.json",
                    "path": "package.json",
                    "type": "nodejs_manifest",
                    "framework": "nodejs",
                    "main": main_file,
                    "bin": bin_spec,
                    "scripts": scripts,
                    "details": f"Node.js package manifest defining {len(scripts)} scripts.",
                })
            except Exception:
                entrypoints.append({
                    "file": "package.json",
                    "path": "package.json",
                    "type": "nodejs_manifest",
                    "framework": "nodejs",
                    "details": "Node.js package manifest (unparseable JSON).",
                })

        # 2. Inspect Python files
        for root, dirs, files in os.walk(str(repo_path)):
            dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]
            for file_name in files:
                if not file_name.endswith(".py"):
                    continue

                full_path = Path(root) / file_name
                rel_path = str(full_path.relative_to(repo_path))

                if file_name == "setup.py":
                    entrypoints.append({
                        "file": rel_path,
                        "path": rel_path,
                        "type": "python_entrypoint",
                        "framework": "setuptools",
                        "details": "Python package setup script (setup.py).",
                    })
                    continue

                is_known_entry_name = file_name in {
                    "main.py",
                    "app.py",
                    "server.py",
                    "cli.py",
                    "manage.py",
                    "wsgi.py",
                    "asgi.py",
                }

                has_main_block = False
                detected_framework = None

                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    if 'if __name__ == "__main__":' in content or "if __name__ == '__main__':":
                        has_main_block = True

                    tree = ast.parse(content, filename=rel_path)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func_name = ""
                            if isinstance(node.func, ast.Name):
                                func_name = node.func.id
                            elif isinstance(node.func, ast.Attribute):
                                func_name = node.func.attr

                            if func_name in {"FastAPI"}:
                                detected_framework = "fastapi"
                            elif func_name in {"Flask"}:
                                detected_framework = "flask"
                            elif func_name in {"Streamlit", "set_page_config"}:
                                detected_framework = "streamlit"
                except Exception:
                    pass

                if is_known_entry_name or has_main_block or detected_framework:
                    framework_label = detected_framework or ("python_main" if has_main_block else "generic_script")
                    entrypoints.append({
                        "file": rel_path,
                        "path": rel_path,
                        "type": "python_entrypoint",
                        "framework": framework_label,
                        "has_main_block": has_main_block,
                        "details": f"Python entrypoint ({framework_label}) with main block: {has_main_block}.",
                    })

        return entrypoints

    def _detect_container_and_ci(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Identify container orchestration, CI/CD pipelines, and IDE automation triggers."""
        configs: List[Dict[str, Any]] = []

        # 1. Dockerfiles
        for root, dirs, files in os.walk(str(repo_path)):
            dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]
            for file_name in files:
                if self._identify_language(file_name) == "Dockerfile":
                    full_path = Path(root) / file_name
                    rel_path = str(full_path.relative_to(repo_path))
                    instructions = self._extract_dockerfile_instructions(full_path)
                    configs.append({
                        "file": rel_path,
                        "path": rel_path,
                        "type": "dockerfile",
                        "details": instructions,
                    })

        # 2. Docker Compose
        compose_candidates = [
            "docker-compose.yml",
            "docker-compose.yaml",
            "compose.yml",
            "compose.yaml",
        ]
        for candidate in compose_candidates:
            compose_file = repo_path / candidate
            if compose_file.exists() and compose_file.is_file():
                rel_path = str(compose_file.relative_to(repo_path))
                details = self._extract_compose_details(compose_file)
                configs.append({
                    "file": rel_path,
                    "path": rel_path,
                    "type": "docker_compose",
                    "details": details,
                })

        # 3. Makefile
        makefile = repo_path / "Makefile"
        if makefile.exists() and makefile.is_file():
            targets = self._extract_makefile_targets(makefile)
            configs.append({
                "file": "Makefile",
                "path": "Makefile",
                "type": "makefile",
                "details": {"targets": targets},
            })

        # 4. CI/CD Workflows (.github/workflows)
        gh_workflows = repo_path / ".github" / "workflows"
        if gh_workflows.exists() and gh_workflows.is_dir():
            for item in gh_workflows.iterdir():
                if item.is_file() and item.suffix.lower() in {".yml", ".yaml"}:
                    rel_path = str(item.relative_to(repo_path))
                    wf_details = self._extract_github_workflow_details(item)
                    configs.append({
                        "file": rel_path,
                        "path": rel_path,
                        "type": "github_actions",
                        "details": wf_details,
                    })

        # 5. GitLab CI (.gitlab-ci.yml)
        gitlab_ci = repo_path / ".gitlab-ci.yml"
        if gitlab_ci.exists() and gitlab_ci.is_file():
            configs.append({
                "file": ".gitlab-ci.yml",
                "path": ".gitlab-ci.yml",
                "type": "gitlab_ci",
                "details": {"file": ".gitlab-ci.yml"},
            })

        # 6. VS Code Automation (.vscode/tasks.json, .vscode/launch.json)
        vscode_dir = repo_path / ".vscode"
        if vscode_dir.exists() and vscode_dir.is_dir():
            tasks_json = vscode_dir / "tasks.json"
            if tasks_json.exists() and tasks_json.is_file():
                tasks_info = self._extract_vscode_tasks(tasks_json)
                configs.append({
                    "file": ".vscode/tasks.json",
                    "path": ".vscode/tasks.json",
                    "type": "vscode_tasks",
                    "details": tasks_info,
                })

            launch_json = vscode_dir / "launch.json"
            if launch_json.exists() and launch_json.is_file():
                launch_info = self._extract_vscode_launch(launch_json)
                configs.append({
                    "file": ".vscode/launch.json",
                    "path": ".vscode/launch.json",
                    "type": "vscode_launch",
                    "details": launch_info,
                })

        return configs

    def _extract_dockerfile_instructions(self, file_path: Path) -> Dict[str, Any]:
        """Extract ENTRYPOINT, CMD, and RUN steps from Dockerfile."""
        entrypoints: List[str] = []
        cmds: List[str] = []
        run_count = 0
        try:
            for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.strip()
                if stripped.startswith("ENTRYPOINT"):
                    entrypoints.append(stripped)
                elif stripped.startswith("CMD"):
                    cmds.append(stripped)
                elif stripped.startswith("RUN"):
                    run_count += 1
        except Exception:
            pass
        return {
            "entrypoint": entrypoints,
            "cmd": cmds,
            "run_steps_count": run_count,
        }

    def _extract_compose_details(self, file_path: Path) -> Dict[str, Any]:
        """Extract services, ports, and privileged flags from docker-compose."""
        content = ""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return {}

        if yaml:
            try:
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    services = data.get("services", {})
                    service_names = list(services.keys()) if isinstance(services, dict) else []
                    privileged = []
                    for s_name, s_conf in (services.items() if isinstance(services, dict) else []):
                        if isinstance(s_conf, dict) and s_conf.get("privileged") is True:
                            privileged.append(s_name)
                    return {
                        "services": service_names,
                        "privileged_services": privileged,
                    }
            except Exception:
                pass

        # Fallback simple line regex
        services = re.findall(r"^\s{2}([a-zA-Z0-9_-]+):\s*$", content, re.MULTILINE)
        return {"services": services, "privileged_services": []}

    def _extract_makefile_targets(self, file_path: Path) -> List[str]:
        """Extract defined targets from Makefile."""
        targets: List[str] = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for line in content.splitlines():
                match = re.match(r"^([a-zA-Z0-9_\-\.]+):", line)
                if match:
                    targets.append(match.group(1))
        except Exception:
            pass
        return targets

    def _extract_github_workflow_details(self, file_path: Path) -> Dict[str, Any]:
        """Extract triggers and job names from GitHub Actions workflow."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if yaml:
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    triggers = data.get("on", [])
                    if isinstance(triggers, dict):
                        triggers = list(triggers.keys())
                    elif isinstance(triggers, str):
                        triggers = [triggers]
                    jobs = list(data.get("jobs", {}).keys()) if isinstance(data.get("jobs"), dict) else []
                    return {
                        "name": data.get("name", file_path.stem),
                        "triggers": triggers,
                        "jobs": jobs,
                    }
        except Exception:
            pass
        return {"name": file_path.stem, "triggers": [], "jobs": []}

    def _extract_vscode_tasks(self, file_path: Path) -> Dict[str, Any]:
        """Extract task definitions and auto-run settings from tasks.json."""
        try:
            data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
            tasks = data.get("tasks", [])
            extracted = []
            auto_run_tasks = []
            for t in tasks:
                if isinstance(t, dict):
                    label = t.get("label", "unlabeled")
                    t_type = t.get("type", "unknown")
                    cmd = t.get("command", "")
                    run_options = t.get("runOptions", {})
                    extracted.append({
                        "label": label,
                        "type": t_type,
                        "command": cmd,
                        "run_options": run_options,
                    })
                    if isinstance(run_options, dict) and run_options.get("runOn") == "folderOpen":
                        auto_run_tasks.append(label)
            return {
                "tasks_count": len(extracted),
                "tasks": extracted,
                "auto_run_on_folder_open": auto_run_tasks,
            }
        except Exception:
            return {"tasks_count": 0, "tasks": [], "auto_run_on_folder_open": []}

    def _extract_vscode_launch(self, file_path: Path) -> Dict[str, Any]:
        """Extract debug launch configurations from launch.json."""
        try:
            data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
            configs = data.get("configurations", [])
            extracted = []
            for c in configs:
                if isinstance(c, dict):
                    extracted.append({
                        "name": c.get("name", "unnamed"),
                        "type": c.get("type", "unknown"),
                        "program": c.get("program", ""),
                    })
            return {"configurations": extracted}
        except Exception:
            return {"configurations": []}

    def _build_dependency_graph(
        self, repo_path: Path, entrypoints: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Map static module dependencies and script call invocations."""
        graph: List[Dict[str, Any]] = []

        # Map local module names to relative file paths
        local_modules: Dict[str, str] = {}
        for root, dirs, files in os.walk(str(repo_path)):
            dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]
            for file_name in files:
                if file_name.endswith(".py"):
                    full_p = Path(root) / file_name
                    rel_p = str(full_p.relative_to(repo_path))
                    mod_name = file_name[:-3]
                    local_modules[mod_name] = rel_p

        # Extract Python import relationships
        for root, dirs, files in os.walk(str(repo_path)):
            dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]
            for file_name in files:
                if not file_name.endswith(".py"):
                    continue
                full_p = Path(root) / file_name
                rel_p = str(full_p.relative_to(repo_path))

                try:
                    content = full_p.read_text(encoding="utf-8", errors="ignore")
                    tree = ast.parse(content, filename=rel_p)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                base_mod = alias.name.split(".")[0]
                                if base_mod in local_modules and local_modules[base_mod] != rel_p:
                                    graph.append({
                                        "source": rel_p,
                                        "target": local_modules[base_mod],
                                        "imported_names": [alias.name],
                                        "type": "python_import",
                                    })
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                base_mod = node.module.split(".")[0]
                                if base_mod in local_modules and local_modules[base_mod] != rel_p:
                                    imported = [a.name for a in node.names]
                                    graph.append({
                                        "source": rel_p,
                                        "target": local_modules[base_mod],
                                        "imported_names": imported,
                                        "type": "python_import",
                                    })
                except Exception:
                    pass

        # Extract Node.js script invocations from package.json
        package_json = repo_path / "package.json"
        if package_json.exists() and package_json.is_file():
            try:
                pkg_data = json.loads(package_json.read_text(encoding="utf-8", errors="ignore"))
                scripts = pkg_data.get("scripts", {})
                for script_name, cmd in scripts.items():
                    graph.append({
                        "source": "package.json",
                        "target": script_name,
                        "command": cmd,
                        "type": "npm_script",
                    })
            except Exception:
                pass

        return graph

    def _generate_workflow_summary(
        self,
        language_breakdown: Dict[str, Dict[str, int]],
        entrypoints: List[Dict[str, Any]],
        container_and_ci_configs: List[Dict[str, Any]],
        module_dependency_graph: List[Dict[str, Any]],
    ) -> str:
        """Produce a concise, human-readable summary of repository startup and operation."""
        parts: List[str] = []

        # Languages
        if language_breakdown:
            top_langs = [f"{lang} ({data['files']} files, {data['lines']} lines)" for lang, data in list(language_breakdown.items())[:3]]
            parts.append(f"Repository Primary Languages: {', '.join(top_langs)}.")
        else:
            parts.append("Repository contains no recognized source code files.")

        # Entrypoints
        if entrypoints:
            ep_names = [f"{e.get('file')} ({e.get('type')})" for e in entrypoints[:5]]
            parts.append(f"Identified Entrypoints: {', '.join(ep_names)}.")
        else:
            parts.append("No explicit application entrypoints detected.")

        # Containers & CI
        if container_and_ci_configs:
            cfg_types = [f"{c.get('file')} ({c.get('type')})" for c in container_and_ci_configs]
            parts.append(f"Orchestration & Automation: {', '.join(cfg_types)}.")
        else:
            parts.append("No container or CI/CD configuration files found.")

        # Internal relationships
        if module_dependency_graph:
            parts.append(f"Internal Module Dependency Graph: {len(module_dependency_graph)} execution relationships traced.")

        return " ".join(parts)
