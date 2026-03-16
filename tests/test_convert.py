import polars as pl
import pytest

from loopr import convert_matches_dataframe


def test_convert_matches_accepts_neutral_schema(single_match_neutral_tables):
    df = convert_matches_dataframe(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
        tournament_influence={},
        now_timestamp=1_700_000_100.0,
        decay_rate=0.0,
    )

    assert df.height == 1
    assert set(df["winners"][0]) == {1, 2, 3, 4}
    assert set(df["losers"][0]) == {5, 6, 7, 8}


def test_convert_matches_neutral_appearances_override_rosters(
    single_match_neutral_tables,
):
    df = convert_matches_dataframe(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
        tournament_influence={},
        now_timestamp=1_700_000_100.0,
        decay_rate=0.0,
        appearances=single_match_neutral_tables["appearances"],
    )

    assert set(df["winners"][0]) == {1, 2}
    assert set(df["losers"][0]) == {5, 6}


def test_convert_matches_derives_missing_appearance_team_ids(
    single_match_neutral_tables,
):
    appearances_without_groups = single_match_neutral_tables[
        "appearances"
    ].drop("group_id")

    df = convert_matches_dataframe(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
        tournament_influence={},
        now_timestamp=1_700_000_100.0,
        decay_rate=0.0,
        appearances=appearances_without_groups,
    )

    assert set(df["winners"][0]) == {1, 2}
    assert set(df["losers"][0]) == {5, 6}


def test_convert_matches_positional_entity_results_expand_all_implied_pairs():
    df = convert_matches_dataframe(
        pl.DataFrame(
            {
                "event_id": [1, 1, 1],
                "match_id": [10, 10, 10],
                "entity_id": [1, 2, 3],
                "placement": [1, 2, 3],
                "completed_at": [1_700_000_100.0] * 3,
            }
        ),
        None,
        tournament_influence={},
        now_timestamp=1_700_000_100.0,
        decay_rate=0.0,
        result_mode="positional",
    )

    assert df.height == 3
    assert {
        (tuple(row["winners"]), tuple(row["losers"]))
        for row in df.iter_rows(named=True)
    } == {
        ((1,), (2,)),
        ((1,), (3,)),
        ((2,), (3,)),
    }


def test_convert_matches_positional_group_results_use_appearances():
    df = convert_matches_dataframe(
        pl.DataFrame(
            {
                "event_id": [1, 1, 1],
                "match_id": [10, 10, 10],
                "group_id": [100, 200, 300],
                "placement": [1, 2, 3],
                "completed_at": [1_700_000_100.0] * 3,
            }
        ),
        pl.DataFrame(
            {
                "event_id": [1, 1, 1, 1, 1, 1],
                "group_id": [100, 100, 200, 200, 300, 300],
                "entity_id": [1, 2, 3, 4, 5, 6],
            }
        ),
        tournament_influence={},
        now_timestamp=1_700_000_100.0,
        decay_rate=0.0,
        appearances=pl.DataFrame(
            {
                "event_id": [1, 1, 1, 1],
                "match_id": [10, 10, 10, 10],
                "group_id": [100, 200, 300, 300],
                "entity_id": [1, 3, 5, 6],
            }
        ),
        result_mode="positional",
    )

    assert {
        (tuple(row["winners"]), tuple(row["losers"]))
        for row in df.iter_rows(named=True)
    } == {
        ((1,), (3,)),
        ((1,), (5, 6)),
        ((3,), (5, 6)),
    }


def test_convert_matches_positional_pairwise_average_normalizes_total_weight():
    results = pl.DataFrame(
        {
            "event_id": [1, 1, 1],
            "match_id": [10, 10, 10],
            "entity_id": [1, 2, 3],
            "placement": [1, 2, 3],
            "completed_at": [1_700_000_100.0] * 3,
        }
    )

    full = convert_matches_dataframe(
        results,
        None,
        tournament_influence={},
        now_timestamp=1_700_000_100.0,
        decay_rate=0.0,
        include_share=False,
        result_mode="positional",
        positional_weight_mode="pairwise_full",
    )
    average = convert_matches_dataframe(
        results,
        None,
        tournament_influence={},
        now_timestamp=1_700_000_100.0,
        decay_rate=0.0,
        include_share=False,
        result_mode="positional",
        positional_weight_mode="pairwise_average",
    )

    assert full["weight"].sum() == pytest.approx(3.0)
    assert average["weight"].sum() == pytest.approx(1.0)
