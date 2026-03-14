"""Ranking engines exposed by the standalone LOOPR package."""

from loopr.algorithms.backends import LogOddsBackend, RowPRBackend
from loopr.algorithms.exposure_log_odds import ExposureLogOddsEngine
from loopr.algorithms.tick_tock import TickTockEngine
from loopr.algorithms.ttl_engine import TTLEngine

LOOPREngine = ExposureLogOddsEngine

__all__ = [
    "ExposureLogOddsEngine",
    "LOOPREngine",
    "TTLEngine",
    "TickTockEngine",
    "LogOddsBackend",
    "RowPRBackend",
]
