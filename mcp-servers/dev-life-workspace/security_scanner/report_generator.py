"""Security Audit Markdown Report Generator.

Synthesizes all checkpointed data from Phases 6.1-6.4 into a structured
Markdown report file saved at knowledge_base/security_reports/.
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

from .audit_state import SecurityAuditStateManager

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
REPORT_DIR = ROOT_DIR / "knowledge_base" / "security_reports"


def _sanitize_for_filename(text: str) -> str:
    """Sanitize a string for safe usage in a filename."""
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(text))
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized[:80] if sanitized else "unknown"


class SecurityAuditReportGenerator:
    """Generates structured Markdown security audit reports from checkpoint data."""

    def __init__(self, db_uri: Optional[str] = None):
        self.state_mgr = SecurityAuditStateManager(db_uri=db_uri)

    def ensure_all_stages_completed(
        self, audit_id: str, auto_run_missing_stages: bool = True
    ) -> Dict[str, Any]:
        """Check and optionally execute any missing audit stages in sequence."""
        record = self.state_mgr.get_audit_run(audit_id)
        if not record:
            raise ValueError(f"Audit run not found for audit_id: {audit_id}")

        if not auto_run_missing_stages:
            return record

        stage_results = record.get("stage_results") or {}
        resolved_path = record.get("resolved_local_path", "")
        target_path = Path(resolved_path)

        if not target_path.exists() or not target_path.is_dir():
            raise ValueError(
                f"Repository directory does not exist: {resolved_path}"
            )

        # Stage 1: workflow_and_packages
        if "workflow_and_packages" not in stage_results:
            from .workflow_analyzer import RepoWorkflowAnalyzer
            from .package_scanner import SuspiciousPackageScanner

            workflow_analyzer = RepoWorkflowAnalyzer()
            workflow_result = workflow_analyzer.analyze(target_path)

            package_scanner = SuspiciousPackageScanner()
            package_result = package_scanner.scan(target_path)

            stage_data = {
                "workflow": workflow_result,
                "packages": package_result,
            }

            all_findings = package_result.get("suspicious_packages", []) + \
                           package_result.get("install_hooks_detected", [])
            has_critical = any(f.get("severity") == "CRITICAL" for f in all_findings)
            has_high = any(f.get("severity") == "HIGH" for f in all_findings)
            has_medium = any(f.get("severity") == "MEDIUM" for f in all_findings)

            if has_critical:
                verdict = "CRITICAL_DANGER"
            elif has_high:
                verdict = "HIGH_RISK"
            elif has_medium:
                verdict = "CAUTION"
            else:
                verdict = "SAFE_SO_FAR"

            pkg_score = package_result.get("package_risk_score", 0)

            record = self.state_mgr.save_stage_checkpoint(
                audit_id=audit_id,
                stage_name="workflow_and_packages_scanned",
                stage_key="workflow_and_packages",
                stage_data=stage_data,
                status="in_progress",
                static_risk_score=pkg_score,
                static_risk_verdict=verdict,
            )
            stage_results = record.get("stage_results") or {}

        # Stage 2: static_malware_scan
        if "static_malware_scan" not in stage_results:
            from .malware_scanner import StaticMalwareScanner

            prior_pkg_score = 0
            prior_verdict = "SAFE_SO_FAR"
            wp_data = stage_results.get("workflow_and_packages", {})
            if wp_data:
                pkg_data = wp_data.get("packages", {})
                prior_pkg_score = pkg_data.get("package_risk_score", 0)
                prior_v = record.get("static_risk_verdict", "SAFE_SO_FAR")
                if prior_v and prior_v != "PENDING":
                    prior_verdict = prior_v

            scanner = StaticMalwareScanner()
            scan_result = scanner.scan(
                repo_path=target_path,
                prior_package_score=prior_pkg_score,
                prior_verdict=prior_verdict,
            )

            record = self.state_mgr.save_stage_checkpoint(
                audit_id=audit_id,
                stage_name="static_malware_scanned",
                stage_key="static_malware_scan",
                stage_data=scan_result,
                status="in_progress",
                static_risk_score=scan_result["static_risk_score"],
                static_risk_verdict=scan_result["static_risk_verdict"],
            )
            stage_results = record.get("stage_results") or {}

        # Stage 3: sandbox_probe
        if "sandbox_probe" not in stage_results:
            from .sandbox_probe import IsolatedSandboxProbe

            # Re-fetch latest record state
            record = self.state_mgr.get_audit_run(audit_id)
            probe = IsolatedSandboxProbe()
            probe_result = probe.run_probe(
                audit_record=record,
                force_run=False,
            )

            current_score = record.get("static_risk_score", 0)
            current_verdict = record.get("static_risk_verdict", "PENDING")
            if probe_result.get("sandbox_risk_verdict") == "CRITICAL_DANGER":
                current_verdict = "CRITICAL_DANGER"
                current_score = max(current_score, 90)

            record = self.state_mgr.save_stage_checkpoint(
                audit_id=audit_id,
                stage_name="sandbox_probed",
                stage_key="sandbox_probe",
                stage_data=probe_result,
                status="in_progress",
                static_risk_score=current_score,
                static_risk_verdict=current_verdict,
            )

        return record

    def generate_report(
        self, audit_id: str, auto_run_missing_stages: bool = True
    ) -> Dict[str, Any]:
        """Execute missing stages, generate Markdown report, and save completion checkpoint."""
        record = self.ensure_all_stages_completed(
            audit_id, auto_run_missing_stages=auto_run_missing_stages
        )

        report_md, report_path = self.generate_markdown(record)
        final_verdict = self._compute_final_verdict(record)
        static_risk_score = record.get("static_risk_score", 0)

        # Save stage checkpoint marking completion
        self.state_mgr.save_stage_checkpoint(
            audit_id=audit_id,
            stage_name="report_generated",
            stage_key="report_generation",
            stage_data={
                "report_path": str(report_path),
                "verdict": final_verdict,
            },
            status="completed",
            report_path=str(report_path),
        )

        return {
            "audit_id": audit_id,
            "status": "completed",
            "current_stage": "report_generated",
            "final_verdict": final_verdict,
            "static_risk_score": static_risk_score,
            "report_path": str(report_path),
            "report_markdown_preview": report_md[:2500],
        }

    def generate_markdown(self, record: Dict[str, Any]) -> Tuple[str, str]:
        """Synthesize record data into Markdown content and write to disk."""
        audit_id = record.get("audit_id", "unknown")
        repo_source = record.get("repo_source", "unknown")
        source_type = record.get("source_type", "unknown")
        resolved_path = record.get("resolved_local_path", "unknown")
        commit_hash = record.get("commit_hash", "") or "N/A"
        static_risk_score = record.get("static_risk_score", 0)
        status = record.get("status", "in_progress")
        stage_results = record.get("stage_results") or {}

        final_verdict = self._compute_final_verdict(record)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        sections = []

        # Top-level Title
        sections.append("# Repository Security Audit Report\n")

        # Metadata Table
        sections.append("## Audit Metadata\n")
        sections.append("| Field | Value |")
        sections.append("|---|---|")
        sections.append(f"| Audit ID | `{audit_id}` |")
        sections.append(f"| Repository Source | `{repo_source}` |")
        sections.append(f"| Source Type | {source_type} |")
        sections.append(f"| Resolved Path | `{resolved_path}` |")
        sections.append(f"| Commit Hash | `{commit_hash}` |")
        sections.append(f"| Final Risk Verdict | **{final_verdict}** |")
        sections.append(f"| Static Risk Score (0-100) | **{static_risk_score}** |")
        sections.append(f"| Audit Status | {status} |")
        sections.append(f"| Generated Timestamp | {timestamp} |")
        sections.append("")

        # 1. Repository Workflow and Architecture
        sections.append(self._section_workflow(stage_results))

        # 2. Suspicious Packages and Dependency Analysis
        sections.append(self._section_packages(stage_results))

        # 3. Suspicious Commands and Code Execution Patterns
        sections.append(self._section_commands(stage_results))

        # 4. Abnormal Encoding and Obfuscation Findings
        sections.append(self._section_obfuscation(stage_results))

        # 5. Abnormal Executables and Binary Artifacts
        sections.append(self._section_executables(stage_results))

        # 6. Dynamic Sandbox Behavioral and Network Telemetry
        sections.append(self._section_sandbox(stage_results))

        # 7. Security Recommendations and Remediation
        sections.append(self._section_recommendations(final_verdict))

        report_md = "\n".join(sections)

        # Write to disk
        repo_slug = _sanitize_for_filename(repo_source)
        report_filename = f"SECURITY_AUDIT_{repo_slug}_{audit_id}.md"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = str(REPORT_DIR / report_filename)
        Path(report_path).write_text(report_md, encoding="utf-8")

        return report_md, report_path

    def _compute_final_verdict(self, record: Dict[str, Any]) -> str:
        """Derive actionable verdict string from cumulative risk score and stage results."""
        verdict = record.get("static_risk_verdict", "PENDING")
        score = record.get("static_risk_score", 0)

        stage_results = record.get("stage_results") or {}
        sandbox_data = stage_results.get("sandbox_probe", {})
        sandbox_verdict = sandbox_data.get("sandbox_risk_verdict", "")

        if verdict == "CRITICAL_DANGER" or sandbox_verdict == "CRITICAL_DANGER" or score >= 60:
            return "DO NOT EXECUTE ON HOST - CRITICAL MALWARE RISK"
        elif verdict == "HIGH_RISK" or score >= 35:
            return "PROCEED WITH CAUTION"
        else:
            return "SAFE TO RUN"

    def _section_workflow(self, stage_results: Dict[str, Any]) -> str:
        """Section 1: Repository Workflow and Architecture."""
        lines = ["## 1. Repository Workflow and Architecture\n"]
        wp_data = stage_results.get("workflow_and_packages", {})
        workflow = wp_data.get("workflow", {})

        if not workflow:
            lines.append("No workflow analysis data available.\n")
            return "\n".join(lines)

        lang_breakdown = workflow.get("language_breakdown", {})
        if lang_breakdown:
            lines.append("### Language Breakdown\n")
            lines.append("| Language | Files | Lines |")
            lines.append("|---|---|---|")
            for lang, stats in lang_breakdown.items():
                if isinstance(stats, dict):
                    f_count = stats.get("files", 0)
                    l_count = stats.get("lines", 0)
                    lines.append(f"| {lang} | {f_count} | {l_count} |")
                else:
                    lines.append(f"| {lang} | {stats} | N/A |")
            lines.append("")

        entrypoints = workflow.get("entrypoints", [])
        if entrypoints:
            lines.append("### Entrypoints\n")
            for ep in entrypoints:
                if isinstance(ep, dict):
                    ep_path = ep.get("file", ep.get("path", str(ep)))
                    ep_type = ep.get("type", "")
                    lines.append(f"- `{ep_path}` ({ep_type})" if ep_type else f"- `{ep_path}`")
                else:
                    lines.append(f"- `{ep}`")
            lines.append("")

        ci_configs = workflow.get("container_and_ci_configs", [])
        if ci_configs:
            lines.append("### CI/CD and Container Configurations\n")
            if isinstance(ci_configs, list):
                for cf in ci_configs:
                    if isinstance(cf, dict):
                        cf_type = cf.get("type", "config")
                        cf_file = cf.get("file", cf.get("path", str(cf)))
                        lines.append(f"- **{cf_type}**: `{cf_file}`")
                    else:
                        lines.append(f"- `{cf}`")
            elif isinstance(ci_configs, dict):
                for config_type, config_files in ci_configs.items():
                    if isinstance(config_files, list):
                        for cf in config_files:
                            if isinstance(cf, dict):
                                lines.append(f"- **{config_type}**: `{cf.get('file', cf.get('path', str(cf)))}`")
                            else:
                                lines.append(f"- **{config_type}**: `{cf}`")
                    elif isinstance(config_files, dict):
                        for k, v in config_files.items():
                            lines.append(f"- **{config_type}/{k}**: `{v}`")
                    else:
                        lines.append(f"- **{config_type}**: `{config_files}`")
            lines.append("")

        dep_graph = workflow.get("module_dependency_graph", [])
        if dep_graph:
            lines.append("### Module Dependency Flow\n")
            if isinstance(dep_graph, list):
                for edge in dep_graph:
                    if isinstance(edge, dict):
                        src = edge.get("source", "unknown")
                        tgt = edge.get("target", "unknown")
                        names = edge.get("imported_names", [])
                        names_str = f" ({', '.join(names)})" if names else ""
                        lines.append(f"- `{src}` -> `{tgt}`{names_str}")
                    else:
                        lines.append(f"- `{edge}`")
            elif isinstance(dep_graph, dict):
                for module, deps in dep_graph.items():
                    if isinstance(deps, list) and deps:
                        dep_str = ", ".join(f"`{d}`" for d in deps)
                        lines.append(f"- `{module}` imports: {dep_str}")
                    else:
                        lines.append(f"- `{module}`: {deps}")
            lines.append("")

        summary = workflow.get("workflow_summary", "")
        if summary:
            lines.append("### Workflow Summary\n")
            lines.append(f"{summary}\n")

        return "\n".join(lines)

    def _section_packages(self, stage_results: Dict[str, Any]) -> str:
        """Section 2: Suspicious Packages and Dependency Analysis."""
        lines = ["## 2. Suspicious Packages and Dependency Analysis\n"]
        wp_data = stage_results.get("workflow_and_packages", {})
        packages = wp_data.get("packages", {})

        if not packages:
            lines.append("No package scanning data available.\n")
            return "\n".join(lines)

        manifests = packages.get("manifests_inspected", [])
        total_deps = packages.get("total_dependencies", 0)
        manifest_str = ", ".join(f"`{m}`" for m in manifests) if manifests else "None"
        lines.append(f"**Manifests Inspected**: {manifest_str}\n")
        lines.append(f"**Total Dependencies Scanned**: {total_deps}\n")

        suspicious = packages.get("suspicious_packages", [])
        hooks = packages.get("install_hooks_detected", [])

        if not suspicious and not hooks:
            lines.append("No suspicious packages or malicious install hooks detected.\n")
            return "\n".join(lines)

        if suspicious:
            lines.append("### Suspicious Packages\n")
            lines.append("| Package Name | Severity | Reason | Manifest |")
            lines.append("|---|---|---|---|")
            for pkg in suspicious:
                name = pkg.get("package_name", pkg.get("name", "unknown"))
                severity = pkg.get("severity", "UNKNOWN")
                reason = pkg.get("reason", pkg.get("detail", "N/A"))
                manifest = pkg.get("manifest_file", pkg.get("source", "N/A"))
                lines.append(f"| `{name}` | {severity} | {reason} | `{manifest}` |")
            lines.append("")

        if hooks:
            lines.append("### Malicious Install Hooks\n")
            lines.append("| Hook Type | Command | Manifest | Severity |")
            lines.append("|---|---|---|---|")
            for hook in hooks:
                hook_type = hook.get("hook_name", hook.get("type", "unknown"))
                command = hook.get("command", "N/A")
                manifest = hook.get("manifest_file", hook.get("source", "N/A"))
                severity = hook.get("severity", "UNKNOWN")
                lines.append(f"| {hook_type} | `{command}` | `{manifest}` | {severity} |")
            lines.append("")

        return "\n".join(lines)

    def _section_commands(self, stage_results: Dict[str, Any]) -> str:
        """Section 3: Suspicious Commands and Code Execution Patterns."""
        lines = ["## 3. Suspicious Commands and Code Execution Patterns\n"]
        malware_data = stage_results.get("static_malware_scan", {})
        commands = malware_data.get("suspicious_commands", [])

        if not commands:
            lines.append("No suspicious commands or code execution patterns detected.\n")
            return "\n".join(lines)

        shell_exec = []
        ide_auto = []
        cred_harvest = []
        secrets = []
        other = []

        for cmd in commands:
            category = str(cmd.get("category", cmd.get("pattern_type", ""))).lower()
            detail = str(cmd.get("detail", cmd.get("description", ""))).lower()
            combined = f"{category} {detail}"
            if any(k in combined for k in ["shell", "subprocess", "os.system", "popen", "pipe"]):
                shell_exec.append(cmd)
            elif any(k in combined for k in ["vscode", "tasks.json", "folderopen", "auto-run", "ide"]):
                ide_auto.append(cmd)
            elif any(k in combined for k in ["ssh", "credential", "env", "shadow", "cookie", "token"]):
                cred_harvest.append(cmd)
            elif any(k in combined for k in ["secret", "private key", "hardcoded", "rsa", "openssh"]):
                secrets.append(cmd)
            else:
                other.append(cmd)

        def _format_findings(title: str, findings: List[Dict]) -> None:
            if not findings:
                return
            lines.append(f"### {title}\n")
            for f in findings:
                file_path = f.get("file", f.get("file_path", "unknown"))
                line_num = f.get("line", f.get("line_number", "N/A"))
                severity = f.get("severity", "UNKNOWN")
                detail = f.get("detail", f.get("description", "N/A"))
                lines.append(f"- **{severity}** `{file_path}` (line {line_num}): {detail}")
            lines.append("")

        _format_findings("Shell Execution Patterns", shell_exec)
        _format_findings("IDE Auto-Execution Patterns", ide_auto)
        _format_findings("Credential Harvesting Patterns", cred_harvest)
        _format_findings("Hardcoded Secrets", secrets)
        _format_findings("Other Suspicious Patterns", other)

        return "\n".join(lines)

    def _section_obfuscation(self, stage_results: Dict[str, Any]) -> str:
        """Section 4: Abnormal Encoding and Obfuscation Findings."""
        lines = ["## 4. Abnormal Encoding and Obfuscation Findings\n"]
        malware_data = stage_results.get("static_malware_scan", {})
        obfuscation = malware_data.get("abnormal_obfuscation", [])

        if not obfuscation:
            lines.append("No abnormal encoding or obfuscation patterns detected.\n")
            return "\n".join(lines)

        for finding in obfuscation:
            file_path = finding.get("file", finding.get("file_path", "unknown"))
            line_num = finding.get("line", finding.get("line_number", "N/A"))
            severity = finding.get("severity", "UNKNOWN")
            detail = finding.get("detail", finding.get("description", "N/A"))
            entropy = finding.get("entropy", "")
            pattern_type = finding.get("pattern_type", finding.get("category", ""))

            entry = f"- **{severity}** `{file_path}` (line {line_num}): {detail}"
            if entropy:
                entry += f" [Shannon entropy: {entropy}]"
            if pattern_type:
                entry += f" [{pattern_type}]"
            lines.append(entry)

        lines.append("")
        return "\n".join(lines)

    def _section_executables(self, stage_results: Dict[str, Any]) -> str:
        """Section 5: Abnormal Executables and Binary Artifacts."""
        lines = ["## 5. Abnormal Executables and Binary Artifacts\n"]
        malware_data = stage_results.get("static_malware_scan", {})
        executables = malware_data.get("abnormal_executables", [])

        if not executables:
            lines.append("No abnormal executables or binary artifacts detected.\n")
            return "\n".join(lines)

        for finding in executables:
            file_path = finding.get("file", finding.get("file_path", "unknown"))
            severity = finding.get("severity", "UNKNOWN")
            detail = finding.get("detail", finding.get("description", "N/A"))
            magic_type = finding.get("magic_type", finding.get("binary_type", ""))
            entry = f"- **{severity}** `{file_path}`: {detail}"
            if magic_type:
                entry += f" [Magic: {magic_type}]"
            lines.append(entry)

        lines.append("")
        return "\n".join(lines)

    def _section_sandbox(self, stage_results: Dict[str, Any]) -> str:
        """Section 6: Dynamic Sandbox Behavioral and Network Telemetry."""
        lines = ["## 6. Dynamic Sandbox Behavioral and Network Telemetry\n"]
        sandbox_data = stage_results.get("sandbox_probe", {})

        if not sandbox_data:
            lines.append("No sandbox probe data available.\n")
            return "\n".join(lines)

        sandbox_executed = sandbox_data.get("sandbox_executed", False)
        skipped_reason = sandbox_data.get("skipped_reason", "")
        execution_mode = sandbox_data.get("execution_mode", "unknown")

        if not sandbox_executed:
            lines.append("**Sandbox Execution**: Skipped\n")
            lines.append(f"**Reason**: {skipped_reason}\n")
            lines.append(f"**Execution Mode**: {execution_mode}\n")
            return "\n".join(lines)

        lines.append("**Sandbox Execution**: Completed\n")
        lines.append(f"**Execution Mode**: {execution_mode}\n")

        entrypoints = sandbox_data.get("entrypoints_tested", [])
        if entrypoints:
            lines.append(f"**Entrypoints Tested**: {', '.join(f'`{e}`' for e in entrypoints)}\n")

        exec_time = sandbox_data.get("execution_time_seconds", "N/A")
        lines.append(f"**Execution Time**: {exec_time}s\n")

        sandbox_verdict = sandbox_data.get("sandbox_risk_verdict", "N/A")
        lines.append(f"**Sandbox Risk Verdict**: {sandbox_verdict}\n")

        # Network attempts
        network = sandbox_data.get("network_attempts", [])
        lines.append("### Outbound Network & DNS Attempts\n")
        if network:
            for attempt in network:
                if isinstance(attempt, dict):
                    att_type = attempt.get("type", "network")
                    destination = attempt.get("destination", attempt.get("host", attempt.get("domain", "unknown")))
                    details = attempt.get("details", "")
                    entry = f"- **{att_type}**: `{destination}`"
                    if details:
                        entry += f" ({details})"
                    lines.append(entry)
                else:
                    lines.append(f"- {attempt}")
        else:
            lines.append("No outbound network or DNS attempts detected.\n")
        lines.append("")

        # Sensitive file access attempts
        file_access = sandbox_data.get("file_access_attempts", [])
        lines.append("### Sensitive/Honeypot File Access Attempts\n")
        if file_access:
            for fa in file_access:
                if isinstance(fa, dict):
                    path = fa.get("path", fa.get("file", "unknown"))
                    is_honeypot = fa.get("is_honeypot", False)
                    mode = fa.get("mode", "")
                    entry = f"- `{path}`"
                    if is_honeypot:
                        entry += " [HONEYPOT]"
                    if mode:
                        entry += f" (mode: {mode})"
                    lines.append(entry)
                else:
                    lines.append(f"- {fa}")
        else:
            lines.append("No sensitive or honeypot file access attempts detected.\n")
        lines.append("")

        # Subprocesses
        processes = sandbox_data.get("process_spawns", [])
        lines.append("### Spawned Subprocesses\n")
        if processes:
            for proc in processes:
                if isinstance(proc, dict):
                    cmd = proc.get("command", "unknown")
                    lines.append(f"- `{cmd}`")
                else:
                    lines.append(f"- `{proc}`")
        else:
            lines.append("No subprocess spawns detected.\n")
        lines.append("")

        # Environment variable inspections
        env_access = sandbox_data.get("env_access_attempts", [])
        lines.append("### Environment Variable Inspections\n")
        if env_access:
            for ea in env_access:
                if isinstance(ea, dict):
                    key = ea.get("key", "unknown")
                    is_honeypot = ea.get("is_honeypot", False)
                    entry = f"- `{key}`"
                    if is_honeypot:
                        entry += " [HONEYPOT TRIGGER]"
                    lines.append(entry)
                else:
                    lines.append(f"- `{ea}`")
        else:
            lines.append("No environment variable inspection detected.\n")
        lines.append("")

        # Honeypot triggers
        triggers = sandbox_data.get("honeypot_triggers", [])
        lines.append("### Honeypot Triggers\n")
        if triggers:
            for trigger in triggers:
                lines.append(f"- {trigger}")
        else:
            lines.append("No honeypot triggers activated.\n")
        lines.append("")

        return "\n".join(lines)

    def _section_recommendations(self, final_verdict: str) -> str:
        """Section 7: Security Recommendations and Remediation."""
        lines = ["## 7. Security Recommendations and Remediation\n"]
        lines.append(f"### Final Verdict: {final_verdict}\n")

        if "CRITICAL" in final_verdict or "DO NOT EXECUTE" in final_verdict:
            lines.append("**Action Required: DO NOT EXECUTE ON HOST - CRITICAL MALWARE RISK**\n")
            lines.append("Remediation steps:\n")
            lines.append("- Do not run, install, or execute any code from this repository on a host machine.")
            lines.append("- Quarantine the repository and conduct deeper manual inspection if required.")
            lines.append("- If this repository is a dependency, immediately replace it with a verified alternative.")
            lines.append("- Inspect identified suspicious commands, obfuscated payloads, and lifecycle install hooks.")
            lines.append("- Report malicious findings to the respective package index or repository host.")
        elif "CAUTION" in final_verdict:
            lines.append("**Action Required: PROCEED WITH CAUTION**\n")
            lines.append("Remediation steps:\n")
            lines.append("- Review all flagged suspicious commands and package dependencies before installation.")
            lines.append("- Execute solely within an isolated container or virtual machine.")
            lines.append("- Pin package versions to known immutable hashes.")
            lines.append("- Monitor network egress and filesystem activity during initial run.")
        else:
            lines.append("**Verdict: SAFE TO RUN**\n")
            lines.append("Recommendations:\n")
            lines.append("- This repository passed all static and dynamic security checks.")
            lines.append("- Follow standard software supply-chain hygiene: maintain lock files and dependency scanning.")
            lines.append("- Re-audit periodically or when dependencies or entrypoints change.")

        lines.append("")
        return "\n".join(lines)
