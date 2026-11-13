"""Repository Security & Malware Auditor Package.

This package provides checkpoint persistence, safe repository cloning,
repository workflow and architecture analysis, dependency package scanning,
static malware scanning (suspicious commands, obfuscation, binary inspection),
isolated runtime sandbox probing with telemetry interception,
and mock test fixtures for the Personal AI Engineer OS security scanner.
"""

from .audit_state import SecurityAuditStateManager
from .repo_cloner import SafeRepoCloner, inspect_repo_directory, get_commit_hash
from .mock_fixtures import create_mock_audit_repos
from .workflow_analyzer import RepoWorkflowAnalyzer
from .package_scanner import SuspiciousPackageScanner
from .malware_scanner import StaticMalwareScanner
from .sandbox_probe import IsolatedSandboxProbe

__all__ = [
    "SecurityAuditStateManager",
    "SafeRepoCloner",
    "inspect_repo_directory",
    "get_commit_hash",
    "create_mock_audit_repos",
    "RepoWorkflowAnalyzer",
    "SuspiciousPackageScanner",
    "StaticMalwareScanner",
    "IsolatedSandboxProbe",
]
