import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore


# Curated baseline lists of popular packages for typosquatting checks
PYPI_POPULAR_PACKAGES: Set[str] = {
    "requests", "urllib3", "cryptography", "pyyaml", "numpy", "pandas",
    "scipy", "matplotlib", "torch", "tensorflow", "scikit-learn", "fastapi",
    "flask", "django", "pydantic", "sqlalchemy", "psycopg2", "psycopg2-binary",
    "redis", "celery", "boto3", "openai", "anthropic", "langchain",
    "pillow", "beautifulsoup4", "selenium", "pytest", "httpx", "aiohttp",
    "paramiko", "pyjwt", "python-dotenv", "colorama", "tqdm", "click", "rich",
}

NPM_POPULAR_PACKAGES: Set[str] = {
    "react", "react-dom", "express", "axios", "lodash", "chalk",
    "commander", "dotenv", "moment", "typescript", "webpack", "next",
    "vue", "socket.io", "jsonwebtoken", "bcrypt", "mongoose", "prisma",
    "ethers", "web3", "cross-env", "nodemon", "eslint", "prettier",
}

# Known Python standard library or ubiquitous benign modules
KNOWN_BENIGN_PACKAGES: Set[str] = {
    "setuptools", "wheel", "pip", "build", "twine", "flake8", "mypy", "black",
    "isort", "coverage", "tox", "virtualenv", "packaging", "certifi", "idna",
    "charset-normalizer", "six", "typing-extensions",
}

DANGEROUS_HOOK_COMMAND_PATTERNS = [
    r"\bcurl\b",
    r"\bwget\b",
    r"\bnc\b",
    r"\bnetcat\b",
    r"invoke-webrequest",
    r"http://",
    r"https://",
    r"\|\s*sh\b",
    r"\|\s*bash\b",
    r"\bpowershell\b",
    r"\bcmd(?:\.exe)?\b",
    r"\bbase64\b",
    r"\bb64decode\b",
    r"\beval\(",
    r"\bexec\(",
    r"\bnode\s+-e\b",
    r"\bpython\s+-c\b",
]

NPM_LIFECYCLE_HOOKS: Set[str] = {
    "preinstall",
    "install",
    "postinstall",
    "preuninstall",
    "postuninstall",
    "prepare",
}


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Damerau-Levenshtein distance between two strings."""
    d: Dict[Tuple[int, int], int] = {}
    len1, len2 = len(s1), len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,        # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost  # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + cost)  # transposition

    return d[(len1 - 1, len2 - 1)]


class SuspiciousPackageScanner:
    """Statically audits dependency manifests and package lifecycle hooks for malicious patterns."""

    def __init__(self):
        self.pypi_baseline = {pkg.lower().replace("_", "-") for pkg in PYPI_POPULAR_PACKAGES}
        self.npm_baseline = {pkg.lower().replace("_", "-") for pkg in NPM_POPULAR_PACKAGES}

    def scan(self, repo_path: Path) -> Dict[str, Any]:
        """Audit all dependency manifests in repo_path."""
        repo_path = repo_path.resolve()
        if not repo_path.is_dir():
            raise ValueError(f"Repository path is not a valid directory: {repo_path}")

        manifests_found: List[str] = []
        dependencies_by_ecosystem: Dict[str, List[Dict[str, Any]]] = {
            "pypi": [],
            "npm": [],
            "cargo": [],
            "go": [],
        }
        suspicious_packages: List[Dict[str, Any]] = []
        install_hooks_detected: List[Dict[str, Any]] = []

        # 1. requirements.txt files
        for item in sorted(repo_path.glob("**/requirements*.txt")):
            if self._in_ignored_dir(item, repo_path):
                continue
            rel_file = str(item.relative_to(repo_path))
            manifests_found.append(rel_file)
            self._scan_requirements_txt(item, rel_file, dependencies_by_ecosystem, suspicious_packages)

        # 2. package.json files
        for item in sorted(repo_path.glob("**/package.json")):
            if self._in_ignored_dir(item, repo_path):
                continue
            rel_file = str(item.relative_to(repo_path))
            manifests_found.append(rel_file)
            self._scan_package_json(item, rel_file, dependencies_by_ecosystem, suspicious_packages, install_hooks_detected)

        # 3. setup.py files
        for item in sorted(repo_path.glob("**/setup.py")):
            if self._in_ignored_dir(item, repo_path):
                continue
            rel_file = str(item.relative_to(repo_path))
            manifests_found.append(rel_file)
            self._scan_setup_py(item, rel_file, dependencies_by_ecosystem, suspicious_packages, install_hooks_detected)

        # 4. pyproject.toml files
        for item in sorted(repo_path.glob("**/pyproject.toml")):
            if self._in_ignored_dir(item, repo_path):
                continue
            rel_file = str(item.relative_to(repo_path))
            manifests_found.append(rel_file)
            self._scan_pyproject_toml(item, rel_file, dependencies_by_ecosystem, suspicious_packages)

        # 5. Pipfile files
        for item in sorted(repo_path.glob("**/Pipfile")):
            if self._in_ignored_dir(item, repo_path):
                continue
            rel_file = str(item.relative_to(repo_path))
            manifests_found.append(rel_file)
            self._scan_pipfile(item, rel_file, dependencies_by_ecosystem, suspicious_packages)

        # 6. Cargo.toml files
        for item in sorted(repo_path.glob("**/Cargo.toml")):
            if self._in_ignored_dir(item, repo_path):
                continue
            rel_file = str(item.relative_to(repo_path))
            manifests_found.append(rel_file)
            self._scan_cargo_toml(item, rel_file, dependencies_by_ecosystem, suspicious_packages)

        # 7. go.mod files
        for item in sorted(repo_path.glob("**/go.mod")):
            if self._in_ignored_dir(item, repo_path):
                continue
            rel_file = str(item.relative_to(repo_path))
            manifests_found.append(rel_file)
            self._scan_go_mod(item, rel_file, dependencies_by_ecosystem, suspicious_packages)

        total_dependencies_count = sum(len(deps) for deps in dependencies_by_ecosystem.values())
        package_risk_score = self._calculate_risk_score(suspicious_packages, install_hooks_detected)
        summary = self._generate_summary(
            manifests_found=manifests_found,
            total_dependencies=total_dependencies_count,
            suspicious_packages=suspicious_packages,
            install_hooks=install_hooks_detected,
            risk_score=package_risk_score,
        )

        return {
            "manifests_found": manifests_found,
            "total_dependencies_count": total_dependencies_count,
            "dependencies_by_ecosystem": dependencies_by_ecosystem,
            "suspicious_packages": suspicious_packages,
            "install_hooks_detected": install_hooks_detected,
            "package_risk_score": package_risk_score,
            "summary": summary,
        }

    def _in_ignored_dir(self, path: Path, base_dir: Path) -> bool:
        """Check if path is inside an ignored directory."""
        ignored_parts = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".audit_quarantine"}
        rel_parts = path.relative_to(base_dir).parts
        return any(p in ignored_parts or p.endswith(".egg-info") for p in rel_parts)

    def _check_typosquatting(
        self,
        package_name: str,
        ecosystem: str,
        manifest_file: str,
        line_number: int,
        evidence: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if package_name is an edit distance of 1 or 2 from a popular package baseline."""
        normalized = package_name.strip().lower().replace("_", "-")
        if not normalized:
            return None

        # Determine target baseline
        baseline = self.pypi_baseline if ecosystem == "pypi" else self.npm_baseline
        all_baselines = self.pypi_baseline | self.npm_baseline | KNOWN_BENIGN_PACKAGES

        # If package is an exact match to a known package, it is safe
        if normalized in all_baselines:
            return None

        # Check against target ecosystem baseline
        for target in sorted(baseline):
            # Guard against short strings generating spurious matches
            if len(target) <= 3 and len(normalized) != len(target):
                continue

            dist = damerau_levenshtein_distance(normalized, target)
            if dist in (1, 2):
                return {
                    "package_name": package_name,
                    "ecosystem": ecosystem,
                    "manifest_file": manifest_file,
                    "line_number": line_number,
                    "issue_type": "typosquatting",
                    "severity": "HIGH",
                    "evidence": evidence,
                    "description": f"Potential typosquatting targeting popular package '{target}' (distance: {dist}).",
                }

        # Check cross-ecosystem baseline as fallback
        other_baseline = self.npm_baseline if ecosystem == "pypi" else self.pypi_baseline
        for target in sorted(other_baseline):
            if len(target) <= 3 and len(normalized) != len(target):
                continue

            dist = damerau_levenshtein_distance(normalized, target)
            if dist in (1, 2):
                return {
                    "package_name": package_name,
                    "ecosystem": ecosystem,
                    "manifest_file": manifest_file,
                    "line_number": line_number,
                    "issue_type": "typosquatting",
                    "severity": "HIGH",
                    "evidence": evidence,
                    "description": f"Potential typosquatting targeting popular package '{target}' (distance: {dist}).",
                }

        return None

    def _scan_requirements_txt(
        self,
        file_path: Path,
        manifest_file: str,
        dependencies: Dict[str, List[Dict[str, Any]]],
        suspicious: List[Dict[str, Any]],
    ) -> None:
        """Scan a Python requirements.txt file."""
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return

        for idx, line in enumerate(lines, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            # Check direct URL / tarball / git source
            if re.match(r"^(?:https?|git\+|git|ssh|file):\/\/", raw) or raw.endswith((".tar.gz", ".whl", ".zip", ".tgz")):
                is_unencrypted = raw.startswith("http://")
                is_file = raw.startswith("file://")
                severity = "HIGH" if (is_unencrypted or is_file or "tar.gz" in raw or "whl" in raw) else "MEDIUM"
                suspicious.append({
                    "package_name": raw,
                    "ecosystem": "pypi",
                    "manifest_file": manifest_file,
                    "line_number": idx,
                    "issue_type": "direct_url_dependency",
                    "severity": severity,
                    "evidence": raw,
                    "description": f"Direct untrusted package source or tarball dependency: {raw}",
                })
                dependencies["pypi"].append({
                    "name": raw,
                    "spec": "direct_url",
                    "manifest_file": manifest_file,
                    "line_number": idx,
                })
                continue

            # Standard package requirement
            match = re.match(r"^([a-zA-Z0-9_\-\.]+)(?:([=<>!~]+.*))?$", raw)
            if match:
                pkg_name = match.group(1)
                spec = match.group(2) or "unpinned"
                dependencies["pypi"].append({
                    "name": pkg_name,
                    "spec": spec,
                    "manifest_file": manifest_file,
                    "line_number": idx,
                })

                typo_finding = self._check_typosquatting(pkg_name, "pypi", manifest_file, idx, raw)
                if typo_finding:
                    suspicious.append(typo_finding)

    def _scan_package_json(
        self,
        file_path: Path,
        manifest_file: str,
        dependencies: Dict[str, List[Dict[str, Any]]],
        suspicious: List[Dict[str, Any]],
        install_hooks: List[Dict[str, Any]],
    ) -> None:
        """Scan a Node.js package.json file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            data = json.loads(content)
        except Exception:
            return

        def _find_line(text_snippet: str) -> int:
            for i, l in enumerate(lines, start=1):
                if text_snippet in l:
                    return i
            return 1

        # 1. Audit lifecycle hooks in scripts
        scripts = data.get("scripts", {})
        if isinstance(scripts, dict):
            for hook_name, cmd in scripts.items():
                if hook_name in NPM_LIFECYCLE_HOOKS:
                    cmd_str = str(cmd)
                    line_no = _find_line(f'"{hook_name}"')
                    has_dangerous_pattern = any(
                        re.search(pat, cmd_str, re.IGNORECASE) for pat in DANGEROUS_HOOK_COMMAND_PATTERNS
                    )
                    severity = "CRITICAL" if has_dangerous_pattern else "HIGH"
                    desc = (
                        f"Lifecycle hook '{hook_name}' executes external command: {cmd_str}"
                    )

                    hook_entry = {
                        "file": manifest_file,
                        "hook_name": hook_name,
                        "command": cmd_str,
                        "severity": severity,
                        "description": desc,
                    }
                    install_hooks.append(hook_entry)

                    suspicious.append({
                        "package_name": hook_name,
                        "ecosystem": "npm",
                        "manifest_file": manifest_file,
                        "line_number": line_no,
                        "issue_type": "malicious_install_hook",
                        "severity": severity,
                        "evidence": f'"{hook_name}": "{cmd_str}"',
                        "description": desc,
                    })

        # 2. Audit dependencies
        dep_sections = ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"]
        for section in dep_sections:
            deps_dict = data.get(section, {})
            if isinstance(deps_dict, dict):
                for pkg_name, spec in deps_dict.items():
                    spec_str = str(spec)
                    line_no = _find_line(f'"{pkg_name}"')

                    dependencies["npm"].append({
                        "name": pkg_name,
                        "spec": spec_str,
                        "manifest_file": manifest_file,
                        "section": section,
                        "line_number": line_no,
                    })

                    # Direct URL / Tarball dependency check
                    if re.match(r"^(?:https?|git|ssh|file):\/\/", spec_str) or spec_str.startswith(("git+", "github:", "file:", "../")):
                        is_unencrypted = spec_str.startswith("http://")
                        severity = "HIGH" if (is_unencrypted or spec_str.startswith(("file:", "../"))) else "MEDIUM"
                        suspicious.append({
                            "package_name": pkg_name,
                            "ecosystem": "npm",
                            "manifest_file": manifest_file,
                            "line_number": line_no,
                            "issue_type": "direct_url_dependency",
                            "severity": severity,
                            "evidence": f'"{pkg_name}": "{spec_str}"',
                            "description": f"Direct untrusted source or tarball dependency: {spec_str}",
                        })
                        continue

                    # Typosquatting check
                    typo_finding = self._check_typosquatting(
                        pkg_name, "npm", manifest_file, line_no, f'"{pkg_name}": "{spec_str}"'
                    )
                    if typo_finding:
                        suspicious.append(typo_finding)

    def _scan_setup_py(
        self,
        file_path: Path,
        manifest_file: str,
        dependencies: Dict[str, List[Dict[str, Any]]],
        suspicious: List[Dict[str, Any]],
        install_hooks: List[Dict[str, Any]],
    ) -> None:
        """Statically inspect setup.py using AST for malicious top-level code and cmdclass overrides."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            tree = ast.parse(content, filename=manifest_file)
        except Exception:
            return

        # Check top-level statements for dangerous execution outside functions/classes
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                self._check_ast_call_for_malice(call, lines, manifest_file, suspicious, install_hooks)
            elif isinstance(node, ast.Assign):
                # Check assigns that call dangerous functions
                if isinstance(node.value, ast.Call):
                    self._check_ast_call_for_malice(node.value, lines, manifest_file, suspicious, install_hooks)

        # Inspect setup() call for install_requires and cmdclass
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                if func_name == "setup":
                    for kw in node.keywords:
                        if kw.arg == "cmdclass":
                            evidence_line = lines[kw.lineno - 1] if kw.lineno <= len(lines) else "cmdclass=..."
                            install_hooks.append({
                                "file": manifest_file,
                                "hook_name": "cmdclass_override",
                                "command": evidence_line.strip(),
                                "severity": "CRITICAL",
                                "description": "Custom cmdclass install command override detected in setup.py",
                            })
                            suspicious.append({
                                "package_name": "setup.py:cmdclass",
                                "ecosystem": "pypi",
                                "manifest_file": manifest_file,
                                "line_number": kw.lineno,
                                "issue_type": "malicious_install_hook",
                                "severity": "CRITICAL",
                                "evidence": evidence_line.strip(),
                                "description": "Custom cmdclass install command override detected in setup.py",
                            })
                        elif kw.arg == "install_requires" and isinstance(kw.value, ast.List):
                            for elt in kw.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    req_str = elt.value
                                    dependencies["pypi"].append({
                                        "name": req_str,
                                        "spec": "setup.py",
                                        "manifest_file": manifest_file,
                                        "line_number": elt.lineno,
                                    })
                                    match = re.match(r"^([a-zA-Z0-9_\-\.]+)", req_str)
                                    if match:
                                        typo = self._check_typosquatting(
                                            match.group(1), "pypi", manifest_file, elt.lineno, req_str
                                        )
                                        if typo:
                                            suspicious.append(typo)

    def _check_ast_call_for_malice(
        self,
        call: ast.Call,
        lines: List[str],
        manifest_file: str,
        suspicious: List[Dict[str, Any]],
        install_hooks: List[Dict[str, Any]],
    ) -> None:
        """Inspect a single AST Call node in setup.py for dangerous execution primitives."""
        call_name = ""
        if isinstance(call.func, ast.Name):
            call_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            val_id = call.func.value.id if isinstance(call.func.value, ast.Name) else ""
            call_name = f"{val_id}.{call.func.attr}" if val_id else call.func.attr

        dangerous_names = {
            "exec", "eval", "os.system", "os.popen", "subprocess.Popen",
            "subprocess.run", "subprocess.call", "subprocess.check_call",
            "subprocess.check_output", "urllib.request.urlopen", "socket.socket",
        }

        # Check if call is dangerous
        if call_name in dangerous_names or call_name.startswith("subprocess."):
            line_no = call.lineno
            evidence = lines[line_no - 1].strip() if line_no <= len(lines) else call_name
            desc = f"Dangerous arbitrary code execution call detected in setup.py: {call_name}"

            install_hooks.append({
                "file": manifest_file,
                "hook_name": "setup_execution_call",
                "command": evidence,
                "severity": "CRITICAL",
                "description": desc,
            })
            suspicious.append({
                "package_name": "setup.py",
                "ecosystem": "pypi",
                "manifest_file": manifest_file,
                "line_number": line_no,
                "issue_type": "malicious_install_hook",
                "severity": "CRITICAL",
                "evidence": evidence,
                "description": desc,
            })

    def _scan_pyproject_toml(
        self,
        file_path: Path,
        manifest_file: str,
        dependencies: Dict[str, List[Dict[str, Any]]],
        suspicious: List[Dict[str, Any]],
    ) -> None:
        """Scan pyproject.toml dependencies."""
        if not tomllib:
            return
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            data = tomllib.loads(content)
        except Exception:
            return

        deps = []
        # PEP 621 dependencies
        project_deps = data.get("project", {}).get("dependencies", [])
        if isinstance(project_deps, list):
            deps.extend(project_deps)

        # Poetry dependencies
        poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        if isinstance(poetry_deps, dict):
            deps.extend(poetry_deps.keys())

        for d in deps:
            if isinstance(d, str):
                match = re.match(r"^([a-zA-Z0-9_\-\.]+)", d)
                if match:
                    pkg_name = match.group(1)
                    dependencies["pypi"].append({
                        "name": pkg_name,
                        "spec": d,
                        "manifest_file": manifest_file,
                        "line_number": 1,
                    })
                    typo = self._check_typosquatting(pkg_name, "pypi", manifest_file, 1, d)
                    if typo:
                        suspicious.append(typo)

    def _scan_pipfile(
        self,
        file_path: Path,
        manifest_file: str,
        dependencies: Dict[str, List[Dict[str, Any]]],
        suspicious: List[Dict[str, Any]],
    ) -> None:
        """Scan Pipfile dependencies."""
        if not tomllib:
            return
        try:
            data = tomllib.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return

        packages = data.get("packages", {})
        if isinstance(packages, dict):
            for pkg, spec in packages.items():
                dependencies["pypi"].append({
                    "name": pkg,
                    "spec": str(spec),
                    "manifest_file": manifest_file,
                    "line_number": 1,
                })
                typo = self._check_typosquatting(pkg, "pypi", manifest_file, 1, f"{pkg} = {spec}")
                if typo:
                    suspicious.append(typo)

    def _scan_cargo_toml(
        self,
        file_path: Path,
        manifest_file: str,
        dependencies: Dict[str, List[Dict[str, Any]]],
        suspicious: List[Dict[str, Any]],
    ) -> None:
        """Scan Cargo.toml dependencies."""
        if not tomllib:
            return
        try:
            data = tomllib.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return

        deps = data.get("dependencies", {})
        if isinstance(deps, dict):
            for pkg, spec in deps.items():
                dependencies["cargo"].append({
                    "name": pkg,
                    "spec": str(spec),
                    "manifest_file": manifest_file,
                    "line_number": 1,
                })

    def _scan_go_mod(
        self,
        file_path: Path,
        manifest_file: str,
        dependencies: Dict[str, List[Dict[str, Any]]],
        suspicious: List[Dict[str, Any]],
    ) -> None:
        """Scan go.mod dependencies."""
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return

        in_require = False
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("require ("):
                in_require = True
                continue
            if in_require and stripped.startswith(")"):
                in_require = False
                continue
            if in_require or stripped.startswith("require "):
                parts = stripped.replace("require ", "").split()
                if len(parts) >= 2:
                    dependencies["go"].append({
                        "name": parts[0],
                        "spec": parts[1],
                        "manifest_file": manifest_file,
                        "line_number": idx,
                    })

    def _calculate_risk_score(
        self,
        suspicious_packages: List[Dict[str, Any]],
        install_hooks: List[Dict[str, Any]],
    ) -> int:
        """Calculate aggregate risk contribution (0 to 100) from package and install hook findings."""
        has_critical = any(
            f.get("severity") == "CRITICAL" for f in (suspicious_packages + install_hooks)
        )
        has_high = any(
            f.get("severity") == "HIGH" for f in (suspicious_packages + install_hooks)
        )
        has_medium = any(
            f.get("severity") == "MEDIUM" for f in (suspicious_packages + install_hooks)
        )

        if has_critical:
            return 95
        if has_high:
            return 65
        if has_medium:
            return 30
        if suspicious_packages or install_hooks:
            return 10
        return 0

    def _generate_summary(
        self,
        manifests_found: List[str],
        total_dependencies: int,
        suspicious_packages: List[Dict[str, Any]],
        install_hooks: List[Dict[str, Any]],
        risk_score: int,
    ) -> str:
        """Generate human-readable summary of the package and dependency security scan."""
        if not manifests_found:
            return "No dependency manifests discovered in repository."

        parts = [
            f"Found {len(manifests_found)} dependency manifests with {total_dependencies} total dependencies."
        ]
        if suspicious_packages or install_hooks:
            critical_count = sum(1 for f in (suspicious_packages + install_hooks) if f.get("severity") == "CRITICAL")
            high_count = sum(1 for f in (suspicious_packages + install_hooks) if f.get("severity") == "HIGH")
            parts.append(
                f"Detected {len(suspicious_packages)} suspicious package findings and "
                f"{len(install_hooks)} lifecycle install hooks "
                f"({critical_count} CRITICAL, {high_count} HIGH). Package risk score: {risk_score}/100."
            )
        else:
            parts.append("Zero suspicious packages or malicious lifecycle hooks detected. Clean dependencies.")

        return " ".join(parts)
