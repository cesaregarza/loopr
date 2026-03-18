import numpy as np
import polars as pl

from loopr import LOOPREngine
from loopr.analysis.loo_analyzer import (
    CachedMatch,
    delta_rho_for_match,
    exposures_for_match,
)
from loopr.example_data import build_quickstart_frames


def _match_count_for_entity(matches_df: pl.DataFrame, entity_id: int) -> int:
    count = 0
    for row in matches_df.iter_rows(named=True):
        winners = row.get("winners", []) or []
        losers = row.get("losers", []) or []
        if entity_id in winners or entity_id in losers:
            count += 1
    return count


def test_compact_cache_reconstructs_triplets_and_teleport(
    single_match_neutral_tables,
):
    engine = LOOPREngine()
    engine.rank_entities(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
        appearances=single_match_neutral_tables["appearances"],
    )
    engine.prepare_loo_analyzer()
    analyzer = engine.get_loo_analyzer()

    assert isinstance(analyzer._match_cache[1], CachedMatch)

    expected_rows_w, expected_cols_w, expected_wts_w = exposures_for_match(
        1, analyzer.matches_df, analyzer.players_df, analyzer.node_to_idx, "win"
    )
    actual_rows_w, actual_cols_w, actual_wts_w = analyzer.exposures_for_match(
        1, "win"
    )
    np.testing.assert_array_equal(actual_rows_w, expected_rows_w)
    np.testing.assert_array_equal(actual_cols_w, expected_cols_w)
    np.testing.assert_allclose(actual_wts_w, expected_wts_w)

    expected_rows_l, expected_cols_l, expected_wts_l = exposures_for_match(
        1, analyzer.matches_df, analyzer.players_df, analyzer.node_to_idx, "loss"
    )
    actual_rows_l, actual_cols_l, actual_wts_l = analyzer.exposures_for_match(
        1, "loss"
    )
    np.testing.assert_array_equal(actual_rows_l, expected_rows_l)
    np.testing.assert_array_equal(actual_cols_l, expected_cols_l)
    np.testing.assert_allclose(actual_wts_l, expected_wts_l)

    expected_delta = delta_rho_for_match(
        1,
        analyzer.matches_df,
        analyzer.players_df,
        analyzer.node_to_idx,
        analyzer._total_exposure,
    )
    actual_delta = analyzer.delta_rho_for_match(1)
    np.testing.assert_allclose(actual_delta, expected_delta)


def test_entity_match_index_matches_resolved_match_involvement(
    multi_event_neutral_tables,
):
    engine = LOOPREngine()
    engine.rank_entities(
        multi_event_neutral_tables["matches"],
        multi_event_neutral_tables["participants"],
    )
    engine.prepare_loo_analyzer()
    analyzer = engine.get_loo_analyzer()

    for entity_id in [1, 5, 9]:
        expected = _match_count_for_entity(analyzer.matches_df, entity_id)
        actual = len(analyzer._entity_match_index.get(entity_id, ()))
        assert actual == expected


def test_analyze_entity_matches_limit_none_skips_flux_prefilter(
    multi_event_neutral_tables, monkeypatch
):
    engine = LOOPREngine()
    engine.rank_entities(
        multi_event_neutral_tables["matches"],
        multi_event_neutral_tables["participants"],
    )
    engine.prepare_loo_analyzer()
    analyzer = engine.get_loo_analyzer()

    calls = 0

    def _counting_flux(
        match_id: int, entity_id: int, *, use_numba: bool = False
    ) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    def _stub_impact(
        match_id: int,
        entity_id: int,
        *,
        include_teleport: bool,
        solve_strategy: str,
        combine_rhs: bool,
        approx_steps: int,
        use_numba: bool,
        tol: float,
        max_iter: int,
    ) -> dict[str, object]:
        return {
            "ok": True,
            "old": {"score": 1.0},
            "new": {"score": 1.0},
            "delta": {"score": 0.0, "s_win": 0.0, "s_loss": 0.0},
        }

    monkeypatch.setattr(analyzer, "_estimate_match_flux", _counting_flux)
    monkeypatch.setattr(analyzer, "_impact_of_match_on_entity_variant", _stub_impact)

    df = analyzer.analyze_entity_matches(
        entity_id=1,
        limit=None,
        use_flux_ranking=True,
        parallel=False,
    )

    assert df.height == 2
    assert calls == 0


def test_analyze_entity_matches_limited_rows_are_exact():
    frames = build_quickstart_frames()
    engine = LOOPREngine()
    engine.rank_entities(
        frames["matches"],
        frames["participants"],
        appearances=frames["appearances"],
    )
    engine.prepare_loo_analyzer()
    analyzer = engine.get_loo_analyzer()

    df = analyzer.analyze_entity_matches(
        entity_id=1,
        limit=3,
        use_flux_ranking=True,
        parallel=False,
    )

    assert df.height == 3

    for row in df.iter_rows(named=True):
        impact = analyzer.impact_of_match_on_entity(row["match_id"], entity_id=1)
        assert impact["ok"] is True
        assert row["old_score"] == impact["old"]["score"]
        assert row["new_score"] == impact["new"]["score"]
        assert row["score_delta"] == impact["delta"]["score"]
        assert row["win_pr_delta"] == impact["delta"]["s_win"]
        assert row["loss_pr_delta"] == impact["delta"]["s_loss"]


def test_exact_combined_matches_exact_separate_single_impact():
    frames = build_quickstart_frames()
    engine = LOOPREngine()
    engine.rank_entities(
        frames["matches"],
        frames["participants"],
        appearances=frames["appearances"],
    )
    engine.prepare_loo_analyzer()
    analyzer = engine.get_loo_analyzer()

    combined = analyzer.impact_of_match_on_entity_variant(
        1,
        entity_id=1,
        variant="exact_combined",
    )
    separate = analyzer.impact_of_match_on_entity_variant(
        1,
        entity_id=1,
        variant="exact_separate",
    )

    assert combined["ok"] is True
    assert separate["ok"] is True
    assert combined["new"]["score"] == separate["new"]["score"]
    assert combined["delta"]["score"] == separate["delta"]["score"]
    assert combined["delta"]["s_win"] == separate["delta"]["s_win"]
    assert combined["delta"]["s_loss"] == separate["delta"]["s_loss"]


def test_exact_combined_matches_exact_separate_batch_output():
    frames = build_quickstart_frames()
    engine = LOOPREngine()
    engine.rank_entities(
        frames["matches"],
        frames["participants"],
        appearances=frames["appearances"],
    )
    engine.prepare_loo_analyzer()
    analyzer = engine.get_loo_analyzer()

    combined = analyzer.analyze_entity_matches_variant(
        entity_id=1,
        variant="exact_combined",
        limit=5,
        parallel=False,
    )
    separate = analyzer.analyze_entity_matches_variant(
        entity_id=1,
        variant="exact_separate",
        limit=5,
        parallel=False,
    )

    assert combined.to_dicts() == separate.to_dicts()


def test_perturbation_variant_returns_ranked_impacts():
    frames = build_quickstart_frames()
    engine = LOOPREngine()
    engine.rank_entities(
        frames["matches"],
        frames["participants"],
        appearances=frames["appearances"],
    )
    engine.prepare_loo_analyzer()
    analyzer = engine.get_loo_analyzer()

    df = analyzer.analyze_entity_matches_variant(
        entity_id=1,
        variant="perturb_2",
        limit=5,
        parallel=False,
    )

    assert df.height == 5
    assert "match_id" in df.columns
    assert "score_delta" in df.columns
