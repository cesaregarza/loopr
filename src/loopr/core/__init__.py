"""Core utilities shared by LOOPR ranking engines."""

from loopr.core import constants  # noqa: F401
from loopr.core.config import (
    DecayConfig,
    EngineConfig,
    ExposureLogOddsConfig,
    PageRankConfig,
    TickTockConfig,
)
from loopr.core.convert import (
    build_node_mapping,
    convert_matches_dataframe,
    convert_matches_format,
    convert_team_matches,
    factorize_ids,
)
from loopr.core.edges import (
    build_exposure_triplets,
    build_player_edges,
    build_team_edges,
    compute_denominators,
    edges_to_triplets,
    normalize_edges,
)
from loopr.core.influence import (
    aggregate_multi_round_influence,
    compute_retrospective_strength,
    compute_tournament_influence,
    normalize_influence,
)
from loopr.core.pagerank import (
    pagerank_dense,
    pagerank_from_adjacency,
    pagerank_sparse,
)
from loopr.core.protocols import RatingBackend
from loopr.core.results import (
    ExposureLogOddsResult,
    RankResult,
    TickTockResult,
)
from loopr.core.smoothing import (
    AdaptiveSmoothing,
    ConstantSmoothing,
    HybridSmoothing,
    NoSmoothing,
    SmoothingStrategy,
    WinsProportional,
    get_smoothing_strategy,
)
from loopr.core.teleport import (
    ActivePlayersTeleport,
    CustomTeleport,
    TeleportStrategy,
    UniformTeleport,
    VolumeInverseTeleport,
    uniform,
    volume_inverse,
)
from loopr.core.time import (
    Clock,
    add_time_features,
    apply_inactivity_decay,
    compute_decay_factor,
    create_time_windows,
    decay_expr,
    event_ts_expr,
    filter_by_recency,
)

__all__ = [
    "AdaptiveSmoothing",
    "ActivePlayersTeleport",
    "Clock",
    "ConstantSmoothing",
    "CustomTeleport",
    "DecayConfig",
    "EngineConfig",
    "ExposureLogOddsConfig",
    "ExposureLogOddsResult",
    "HybridSmoothing",
    "NoSmoothing",
    "PageRankConfig",
    "RankResult",
    "RatingBackend",
    "SmoothingStrategy",
    "TeleportStrategy",
    "TickTockConfig",
    "TickTockResult",
    "UniformTeleport",
    "VolumeInverseTeleport",
    "WinsProportional",
    "add_time_features",
    "aggregate_multi_round_influence",
    "apply_inactivity_decay",
    "build_exposure_triplets",
    "build_node_mapping",
    "build_player_edges",
    "build_team_edges",
    "compute_decay_factor",
    "compute_denominators",
    "compute_retrospective_strength",
    "compute_tournament_influence",
    "convert_matches_dataframe",
    "convert_matches_format",
    "convert_team_matches",
    "create_time_windows",
    "decay_expr",
    "edges_to_triplets",
    "event_ts_expr",
    "factorize_ids",
    "filter_by_recency",
    "get_smoothing_strategy",
    "normalize_edges",
    "normalize_influence",
    "pagerank_dense",
    "pagerank_from_adjacency",
    "pagerank_sparse",
    "uniform",
    "volume_inverse",
]
