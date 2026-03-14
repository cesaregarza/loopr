"""Standalone domain-agnostic LOOPR ranking library."""

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
    normalize_appearances_schema,
    normalize_matches_schema,
    normalize_participants_schema,
    normalize_rank_inputs,
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
    "normalize_appearances_schema",
    "normalize_matches_schema",
    "normalize_participants_schema",
    "normalize_rank_inputs",
    "prepare_rank_inputs",
]

__version__ = "0.1.0"
