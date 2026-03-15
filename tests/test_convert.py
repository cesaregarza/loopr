import polars as pl

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
