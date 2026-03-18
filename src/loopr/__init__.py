"""Standalone domain-agnostic LOOPR ranking library."""

from loopr.api import rank_entities
from loopr.algorithms import (
    ExposureLogOddsEngine,
    LOOPREngine,
    TTLEngine,
    TickTockEngine,
)
from loopr.analysis import LOOAnalyzer
from loopr.core import (
    Clock,
    DecayConfig,
    EngineConfig,
    ExposureLogOddsConfig,
    ExposureLogOddsResult,
    PageRankConfig,
    TickTockConfig,
    TickTockResult,
    convert_matches_dataframe,
)
from loopr.schema import (
    NormalizedRankingInputs,
    prepare_rank_inputs,
)

__all__ = [
    "Clock",
    "DecayConfig",
    "EngineConfig",
    "ExposureLogOddsConfig",
    "ExposureLogOddsEngine",
    "ExposureLogOddsResult",
    "LOOAnalyzer",
    "LOOPREngine",
    "NormalizedRankingInputs",
    "PageRankConfig",
    "TTLEngine",
    "TickTockConfig",
    "TickTockEngine",
    "TickTockResult",
    "convert_matches_dataframe",
    "prepare_rank_inputs",
    "rank_entities",
]

__version__ = "0.1.0"
