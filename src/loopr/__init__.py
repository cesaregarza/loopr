"""Standalone domain-agnostic LOOPR ranking library."""

from loopr._version import __version__
from loopr.api import rank_entities
from loopr.algorithms import LOOPREngine
from loopr.core import ExposureLogOddsConfig
from loopr.fit import assess_dataset_fit
from loopr.schema import prepare_rank_inputs

__all__ = [
    "ExposureLogOddsConfig",
    "LOOPREngine",
    "assess_dataset_fit",
    "prepare_rank_inputs",
    "rank_entities",
]
