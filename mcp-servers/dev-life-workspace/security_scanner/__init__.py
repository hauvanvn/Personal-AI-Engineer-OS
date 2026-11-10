"""Repository Security & Malware Auditor Package.

This package provides checkpoint persistence, safe repository cloning, and
mock test fixtures for the Personal AI Engineer OS security scanner.
"""

from .audit_state import SecurityAuditStateManager
from .repo_cloner import SafeRepoCloner, inspect_repo_directory, get_commit_hash
from .mock_fixtures import create_mock_audit_repos

__all__ = [
    "SecurityAuditStateManager",
    "SafeRepoCloner",
    "inspect_repo_directory",
    "get_commit_hash",
    "create_mock_audit_repos",
]
