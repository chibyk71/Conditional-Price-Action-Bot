"""CPAB storage package: SQLite signal persistence.

Provides the application's sole authoritative signal store (SQLite), database
path resolution, serialization/fingerprinting, and backup/restore operations.
"""

from cpab.storage.backup import MaintenanceGate, backup_database
from cpab.storage.path import ConfigurationError, resolve_database_path
from cpab.storage.restore import (
    RestoreAbortedError,
    RestoreVerificationError,
    restore_database,
)
from cpab.storage.store import SignalStore

__all__ = [
    "ConfigurationError",
    "MaintenanceGate",
    "RestoreAbortedError",
    "RestoreVerificationError",
    "SignalStore",
    "backup_database",
    "restore_database",
    "resolve_database_path",
]
