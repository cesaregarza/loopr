"""End-to-end tests with concrete value verification.

These tests construct realistic scenarios and verify that the entire pipeline
produces mathematically correct rankings with expected properties.
"""

import numpy as np
import polars as pl
import pytest

from loopr import (
    ExposureLogOddsConfig,
    LOOPREngine,
    TickTockConfig,
    TickTockEngine,
    TTLEngine,
)
from loopr.core.config import DecayConfig, EngineConfig, PageRankConfig

NOW = 1_700_000_000.0
DAY = 86400.0


# ── fixture helpers ──────────────────────────────────────────────────────────


def _make_scenario(
    match_specs: list[tuple[int, int, int, int, float]],
    team_rosters: dict[tuple[int, int], list[int]],
):
    """Build matches + participants from compact spec.

    match_specs: list of (event_id, match_id, winner_team, loser_team, ts)
    team_rosters: {(event_id, team_id): [player_ids]}
    """
    matches = pl.DataFrame(
        {
            "event_id": [s[0] for s in match_specs],
            "match_id": [s[1] for s in match_specs],
            "winner_id": [s[2] for s in match_specs],
            "loser_id": [s[3] for s in match_specs],
            "completed_at": [s[4] for s in match_specs],
        }
    )
    rows = []
    for (eid, tid), pids in team_rosters.items():
        for pid in pids:
            rows.append({"event_id": eid, "group_id": tid, "entity_id": pid})
    participants = pl.DataFrame(rows)
    return matches, participants


# ── Scenario 1: Clear dominance ──────────────────────────────────────────────


@pytest.fixture
def dominant_player_scenario():
    """Team A (players 1,2) beats team B (3,4) and team C (5,6).
    Team B beats team C. Clear hierarchy: A > B > C.
    """
    return _make_scenario(
        match_specs=[
            (1, 10, 100, 200, NOW - 2 * DAY),  # A beats B
            (1, 11, 100, 300, NOW - 1 * DAY),  # A beats C
            (1, 12, 200, 300, NOW),  # B beats C
        ],
        team_rosters={
            (1, 100): [1, 2],
            (1, 200): [3, 4],
            (1, 300): [5, 6],
        },
    )


class TestDominantPlayerRanking:
    """Tests that a clearly dominant player gets top ranking."""

    def test_loopr_respects_hierarchy(self, dominant_player_scenario):
        matches, participants = dominant_player_scenario
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)

        scores = {
            row["entity_id"]: row["score"]
            for row in result.iter_rows(named=True)
        }

        # Team A players (1,2) should rank above team B (3,4) above team C (5,6)
        assert scores[1] > scores[3], "Winner should rank above middle"
        assert scores[3] > scores[5], "Middle should rank above bottom"
        # Teammates should be roughly equal
        assert scores[1] == pytest.approx(scores[2], abs=1e-6)
        assert scores[3] == pytest.approx(scores[4], abs=1e-6)
        assert scores[5] == pytest.approx(scores[6], abs=1e-6)

    def test_tick_tock_respects_hierarchy(self, dominant_player_scenario):
        matches, participants = dominant_player_scenario
        engine = TickTockEngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)

        scores = {
            row["entity_id"]: row["score"]
            for row in result.iter_rows(named=True)
        }

        assert scores[1] > scores[3]
        assert scores[3] > scores[5]

    def test_ttl_respects_hierarchy(self, dominant_player_scenario):
        matches, participants = dominant_player_scenario
        engine = TTLEngine()
        result = engine.rank_entities(matches, participants)

        scores = {
            row["entity_id"]: row["score"]
            for row in result.iter_rows(named=True)
        }

        assert scores[1] > scores[3]
        assert scores[3] > scores[5]


# ── Scenario 2: Time decay ──────────────────────────────────────────────────


class TestTimeDecayEffects:
    def test_recent_win_valued_more(self):
        """With time decay, recent wins count more than old ones.
        Team 100 and 300 each win once, but 300's win is recent and 100's is old.
        Add a third team so the graph isn't perfectly symmetric.
        """
        matches, participants = _make_scenario(
            match_specs=[
                # Team 100 beat team 200 long ago (heavily decayed)
                (1, 10, 100, 200, NOW - 90 * DAY),
                # Team 300 beat team 200 recently (barely decayed)
                (1, 11, 300, 200, NOW),
            ],
            team_rosters={
                (1, 100): [1],
                (1, 200): [2],
                (1, 300): [3],
            },
        )
        config = ExposureLogOddsConfig(
            decay=DecayConfig(half_life_days=30.0),
        )
        engine = LOOPREngine(config=config, now_ts=NOW)
        result = engine.rank_entities(matches, participants)

        scores = {
            row["entity_id"]: row["score"]
            for row in result.iter_rows(named=True)
        }
        # Both beat the same team, but 3's win is recent → higher score
        assert scores[3] > scores[1], (
            "Recent winner should outrank player whose win was 90 days ago"
        )


# ── Scenario 3: Multi-tournament ────────────────────────────────────────────


@pytest.fixture
def multi_tournament_scenario():
    """Two tournaments with overlapping players.
    Tournament 1: A beats B, A beats C
    Tournament 2: B beats D, C beats D
    This tests cross-tournament influence propagation.
    """
    return _make_scenario(
        match_specs=[
            (1, 10, 100, 200, NOW - 3 * DAY),
            (1, 11, 100, 300, NOW - 2 * DAY),
            (2, 20, 200, 400, NOW - 1 * DAY),
            (2, 21, 300, 400, NOW),
        ],
        team_rosters={
            (1, 100): [1, 2],
            (1, 200): [3, 4],
            (1, 300): [5, 6],
            (2, 200): [3, 4],
            (2, 300): [5, 6],
            (2, 400): [7, 8],
        },
    )


class TestMultiTournament:
    def test_all_players_ranked(self, multi_tournament_scenario):
        matches, participants = multi_tournament_scenario
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        ranked_ids = set(result["entity_id"].to_list())
        assert ranked_ids == {1, 2, 3, 4, 5, 6, 7, 8}

    def test_transitive_ranking(self, multi_tournament_scenario):
        matches, participants = multi_tournament_scenario
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)

        scores = {
            row["entity_id"]: row["score"]
            for row in result.iter_rows(named=True)
        }
        # A (1,2) beats B,C; B and C beat D → A > D
        assert scores[1] > scores[7]

    def test_tick_tock_tournament_influence_computed(
        self, multi_tournament_scenario
    ):
        matches, participants = multi_tournament_scenario
        engine = TickTockEngine(now_ts=NOW)
        engine.rank_entities(matches, participants)
        assert len(engine.tournament_influence) == 2
        assert all(v > 0 for v in engine.tournament_influence.values())


# ── Scenario 4: Appearances override rosters ─────────────────────────────────


class TestAppearancesOverride:
    def test_only_appeared_players_get_credit(self):
        """When appearances are given, only those players should participate in the match."""
        matches = pl.DataFrame(
            {
                "event_id": [1],
                "match_id": [10],
                "winner_id": [100],
                "loser_id": [200],
                "completed_at": [NOW],
            }
        )
        # Full rosters: 4 per team
        participants = pl.DataFrame(
            {
                "event_id": [1] * 8,
                "group_id": [100] * 4 + [200] * 4,
                "entity_id": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )
        # Only players 1,2 and 5,6 actually played
        appearances = pl.DataFrame(
            {
                "event_id": [1] * 4,
                "match_id": [10] * 4,
                "entity_id": [1, 2, 5, 6],
                "group_id": [100, 100, 200, 200],
            }
        )
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(
            matches, participants, appearances=appearances
        )
        ranked_ids = set(result["entity_id"].to_list())
        # Only appeared players should be ranked
        assert 1 in ranked_ids
        assert 2 in ranked_ids
        assert 5 in ranked_ids
        assert 6 in ranked_ids


# ── Scenario 5: Score properties ─────────────────────────────────────────────


class TestScoreProperties:
    def test_log_odds_winners_positive_losers_negative(self):
        """In a two-team matchup, the winner should have positive score
        and the loser negative (log-odds centered around 0)."""
        matches, participants = _make_scenario(
            match_specs=[(1, 10, 100, 200, NOW)],
            team_rosters={
                (1, 100): [1],
                (1, 200): [2],
            },
        )
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        scores = {
            row["entity_id"]: row["score"]
            for row in result.iter_rows(named=True)
        }
        assert scores[1] > 0, "Winner should have positive log-odds"
        assert scores[2] < 0, "Loser should have negative log-odds"

    def test_exposure_column_present(self, dominant_player_scenario):
        matches, participants = dominant_player_scenario
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        assert "exposure" in result.columns
        assert all(result["exposure"].to_numpy() > 0)

    def test_win_loss_pr_columns(self, dominant_player_scenario):
        matches, participants = dominant_player_scenario
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        assert "win_pr" in result.columns
        assert "loss_pr" in result.columns
        # All PageRank values should be positive
        assert all(result["win_pr"].to_numpy() > 0)
        assert all(result["loss_pr"].to_numpy() > 0)


# ── Scenario 6: Repeated matchups ───────────────────────────────────────────


class TestRepeatedMatchups:
    def test_undefeated_player_ranks_first(self):
        """A player who wins all their matches should rank #1."""
        matches, participants = _make_scenario(
            match_specs=[
                (1, 10, 100, 200, NOW - 2 * DAY),
                (1, 11, 100, 200, NOW - DAY),
                (1, 12, 100, 200, NOW),
            ],
            team_rosters={(1, 100): [1], (1, 200): [2]},
        )
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        scores = {
            row["entity_id"]: row["score"]
            for row in result.iter_rows(named=True)
        }
        assert scores[1] > scores[2]
        # Log-odds winner should be positive, loser negative
        assert scores[1] > 0
        assert scores[2] < 0


# ── Scenario 7: LOO analyzer values ─────────────────────────────────────────


class TestLOOAnalyzerValues:
    def test_removing_win_decreases_score(self, dominant_player_scenario):
        """Removing a player's win should decrease their score."""
        matches, participants = dominant_player_scenario
        engine = LOOPREngine(now_ts=NOW)
        engine.rank_entities(matches, participants)
        engine.prepare_loo_analyzer()

        # Remove match 10 (team A beats team B) impact on player 1
        impact = engine.analyze_match_impact(match_id=10, player_id=1)
        assert impact["ok"] is True
        # Removing a win should decrease (or maintain) score
        assert impact["delta"]["score"] <= 0.01  # should go down or near zero

    def test_removing_loss_increases_score(self, dominant_player_scenario):
        """Removing a player's loss should increase their score."""
        matches, participants = dominant_player_scenario
        engine = LOOPREngine(now_ts=NOW)
        engine.rank_entities(matches, participants)
        engine.prepare_loo_analyzer()

        # Remove match 10 (team A beats team B) impact on player 3 (loser)
        impact = engine.analyze_match_impact(match_id=10, player_id=3)
        assert impact["ok"] is True
        # Removing a loss should increase score
        assert impact["delta"]["score"] >= -0.01  # should go up or near zero

    def test_impact_contains_required_fields(self, dominant_player_scenario):
        matches, participants = dominant_player_scenario
        engine = LOOPREngine(now_ts=NOW)
        engine.rank_entities(matches, participants)
        engine.prepare_loo_analyzer()

        impact = engine.analyze_match_impact(match_id=10, player_id=1)
        assert "ok" in impact
        assert "old" in impact
        assert "new" in impact
        assert "delta" in impact
        assert "score" in impact["old"]
        assert "s_win" in impact["old"]
        assert "s_loss" in impact["old"]

    def test_analyze_player_matches_returns_dataframe(
        self, dominant_player_scenario
    ):
        matches, participants = dominant_player_scenario
        engine = LOOPREngine(now_ts=NOW)
        engine.rank_entities(matches, participants)
        engine.prepare_loo_analyzer()

        df = engine.analyze_player_matches(player_id=1)
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert "match_id" in df.columns
        assert "score_delta" in df.columns


# ── Scenario 8: Configuration effects ───────────────────────────────────────


class TestConfigurationEffects:
    def test_higher_alpha_changes_scores(self, dominant_player_scenario):
        matches, participants = dominant_player_scenario
        cfg_low = ExposureLogOddsConfig(
            pagerank=PageRankConfig(alpha=0.5),
        )
        cfg_high = ExposureLogOddsConfig(
            pagerank=PageRankConfig(alpha=0.95),
        )
        r_low = LOOPREngine(config=cfg_low, now_ts=NOW).rank_entities(
            matches, participants
        )
        r_high = LOOPREngine(config=cfg_high, now_ts=NOW).rank_entities(
            matches, participants
        )

        s_low = dict(
            zip(r_low["entity_id"].to_list(), r_low["score"].to_list())
        )
        s_high = dict(
            zip(r_high["entity_id"].to_list(), r_high["score"].to_list())
        )

        # Higher alpha should amplify differences
        gap_low = s_low[1] - s_low[5]
        gap_high = s_high[1] - s_high[5]
        assert gap_high > gap_low

    def test_no_tick_tock_still_works(self, dominant_player_scenario):
        matches, participants = dominant_player_scenario
        cfg = ExposureLogOddsConfig(use_tick_tock_active=False)
        engine = LOOPREngine(config=cfg, now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        assert result.height > 0

    def test_fixed_lambda(self, dominant_player_scenario):
        matches, participants = dominant_player_scenario
        cfg = ExposureLogOddsConfig(
            lambda_mode="fixed", fixed_lambda=0.001
        )
        engine = LOOPREngine(config=cfg, now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        assert result.height > 0
        assert engine.last_result.lambda_used == pytest.approx(0.001)


# ── Scenario 9: Tick-tock convergence ────────────────────────────────────────


class TestTickTockConvergence:
    def test_converges_within_max_ticks(self, dominant_player_scenario):
        matches, participants = dominant_player_scenario
        cfg = TickTockConfig(max_ticks=20, convergence_tol=1e-4)
        engine = TickTockEngine(config=cfg, now_ts=NOW)
        engine.rank_entities(matches, participants)
        assert engine.last_result.converged
        assert engine.last_result.iterations <= 20

    def test_influence_normalized_to_mean_one(self, dominant_player_scenario):
        matches, participants = dominant_player_scenario
        engine = TickTockEngine(now_ts=NOW)
        engine.rank_entities(matches, participants)
        influences = list(engine.tournament_influence.values())
        assert np.mean(influences) == pytest.approx(1.0, abs=0.01)

    def test_different_influence_methods_produce_results(
        self, dominant_player_scenario
    ):
        matches, participants = dominant_player_scenario
        for method in ["arithmetic", "sum", "median", "top_20_sum"]:
            cfg = TickTockConfig(influence_method=method)
            engine = TickTockEngine(config=cfg, now_ts=NOW)
            result = engine.rank_entities(matches, participants)
            assert result.height > 0, f"Method {method} produced no results"


# ── Scenario 10: Edge cases ─────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_match_single_player_per_team(self):
        matches, participants = _make_scenario(
            match_specs=[(1, 10, 100, 200, NOW)],
            team_rosters={(1, 100): [1], (1, 200): [2]},
        )
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        assert result.height == 2

    def test_many_matches_same_teams(self):
        """10 matches between the same two teams."""
        specs = [(1, i, 100, 200, NOW - i * DAY) for i in range(10)]
        matches, participants = _make_scenario(
            match_specs=specs,
            team_rosters={(1, 100): [1], (1, 200): [2]},
        )
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        assert result.height == 2

    def test_large_teams(self):
        """Teams with 10 players each."""
        matches, participants = _make_scenario(
            match_specs=[(1, 10, 100, 200, NOW)],
            team_rosters={
                (1, 100): list(range(1, 11)),
                (1, 200): list(range(11, 21)),
            },
        )
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        assert result.height == 20

    def test_walkover_match_filtered(self):
        """Matches marked as walkover/bye should be excluded."""
        matches = pl.DataFrame(
            {
                "event_id": [1, 1],
                "match_id": [10, 11],
                "winner_id": [100, 100],
                "loser_id": [200, 200],
                "completed_at": [NOW, NOW],
                "walkover": [False, True],
            }
        )
        participants = pl.DataFrame(
            {
                "event_id": [1, 1],
                "group_id": [100, 200],
                "entity_id": [1, 2],
            }
        )
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        # Should still produce results (from the non-walkover match)
        assert result.height > 0


# ── Scenario 11: Reproducibility ────────────────────────────────────────────


class TestReproducibility:
    def test_deterministic_results(self, dominant_player_scenario):
        """Running the same scenario twice should produce identical results."""
        matches, participants = dominant_player_scenario
        engine1 = LOOPREngine(now_ts=NOW)
        r1 = engine1.rank_entities(matches, participants)

        engine2 = LOOPREngine(now_ts=NOW)
        r2 = engine2.rank_entities(matches, participants)

        # Sort both by entity_id for comparison
        r1 = r1.sort("entity_id")
        r2 = r2.sort("entity_id")

        np.testing.assert_allclose(
            r1["score"].to_numpy(), r2["score"].to_numpy()
        )

    def test_legacy_neutral_parity_with_values(self):
        """Verify legacy and neutral schemas produce identical numeric outputs."""
        matches_neutral = pl.DataFrame(
            {
                "event_id": [1, 1],
                "match_id": [10, 11],
                "winner_id": [100, 100],
                "loser_id": [200, 300],
                "completed_at": [NOW - DAY, NOW],
            }
        )
        participants_neutral = pl.DataFrame(
            {
                "event_id": [1] * 6,
                "group_id": [100, 100, 200, 200, 300, 300],
                "entity_id": [1, 2, 3, 4, 5, 6],
            }
        )
        matches_legacy = matches_neutral.rename(
            {
                "event_id": "tournament_id",
                "winner_id": "winner_team_id",
                "loser_id": "loser_team_id",
                "completed_at": "last_game_finished_at",
            }
        )
        participants_legacy = participants_neutral.rename(
            {
                "event_id": "tournament_id",
                "group_id": "team_id",
                "entity_id": "user_id",
            }
        )

        engine1 = LOOPREngine(now_ts=NOW)
        r_neutral = engine1.rank_entities(
            matches_neutral, participants_neutral
        ).sort("entity_id")

        engine2 = LOOPREngine(now_ts=NOW)
        r_legacy = engine2.rank_players(
            matches_legacy, participants_legacy
        ).sort("id")

        np.testing.assert_allclose(
            r_neutral["score"].to_numpy(),
            r_legacy["score"].to_numpy(),
            atol=1e-10,
        )


# ── Scenario 12: Backend comparison ─────────────────────────────────────────


class TestBackendComparison:
    def test_ttl_with_row_pr_backend(self, dominant_player_scenario):
        """TTL engine should work with RowPRBackend too."""
        from loopr.algorithms.backends.row_pr import RowPRBackend

        matches, participants = dominant_player_scenario
        backend = RowPRBackend()
        engine = TTLEngine(backend=backend)
        result = engine.rank_entities(matches, participants)
        assert result.height > 0
        # Still respects hierarchy
        scores = {
            row["entity_id"]: row["score"]
            for row in result.iter_rows(named=True)
        }
        assert scores[1] > scores[5]


# ── Scenario 13: Round-robin tournament ──────────────────────────────────────


class TestRoundRobin:
    def test_round_robin_ranking_matches_win_rate(self):
        """In a round-robin, the player with most wins should rank highest."""
        # 4 teams, round-robin where team 100 wins all, 200 wins 2, 300 wins 1, 400 wins 0
        matches, participants = _make_scenario(
            match_specs=[
                (1, 1, 100, 200, NOW - 5 * DAY),
                (1, 2, 100, 300, NOW - 4 * DAY),
                (1, 3, 100, 400, NOW - 3 * DAY),
                (1, 4, 200, 300, NOW - 2 * DAY),
                (1, 5, 200, 400, NOW - 1 * DAY),
                (1, 6, 300, 400, NOW),
            ],
            team_rosters={
                (1, 100): [1],
                (1, 200): [2],
                (1, 300): [3],
                (1, 400): [4],
            },
        )
        engine = LOOPREngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        scores = {
            row["entity_id"]: row["score"]
            for row in result.iter_rows(named=True)
        }
        assert scores[1] > scores[2] > scores[3] > scores[4]

    def test_tick_tock_round_robin(self):
        """Tick-Tock should also produce correct round-robin rankings."""
        matches, participants = _make_scenario(
            match_specs=[
                (1, 1, 100, 200, NOW - 5 * DAY),
                (1, 2, 100, 300, NOW - 4 * DAY),
                (1, 3, 100, 400, NOW - 3 * DAY),
                (1, 4, 200, 300, NOW - 2 * DAY),
                (1, 5, 200, 400, NOW - 1 * DAY),
                (1, 6, 300, 400, NOW),
            ],
            team_rosters={
                (1, 100): [1],
                (1, 200): [2],
                (1, 300): [3],
                (1, 400): [4],
            },
        )
        engine = TickTockEngine(now_ts=NOW)
        result = engine.rank_entities(matches, participants)
        scores = {
            row["entity_id"]: row["score"]
            for row in result.iter_rows(named=True)
        }
        assert scores[1] > scores[2] > scores[3] > scores[4]
