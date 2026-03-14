"""Rating computation backends for tick-tock orchestration."""

from __future__ import annotations

from loopr.algorithms.backends.log_odds import LogOddsBackend
from loopr.algorithms.backends.row_pr import RowPRBackend

__all__ = [
    "LogOddsBackend",
    "RowPRBackend",
]
