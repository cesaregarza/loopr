"""Shared constants for LOOPR ranking engines."""

from __future__ import annotations

# Time conversion
SECONDS_PER_DAY = 86400.0

# Lambda auto-tuning: target = fraction * median(win_pagerank)
LAMBDA_TARGET_FRACTION = 0.025

# Internal schema columns (post-schema.py normalization)
WINNERS = "winners"
LOSERS = "losers"
SHARE = "share"
WEIGHT = "weight"
MATCH_ID = "match_id"
TOURNAMENT_ID = "tournament_id"
WINNER_USER_ID = "winner_user_id"
LOSER_USER_ID = "loser_user_id"
WEIGHT_SUM = "weight_sum"
NORMALIZED_WEIGHT = "normalized_weight"
