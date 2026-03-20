import polars as pl

from loopr import assess_dataset_fit


def test_assess_dataset_fit_reports_good_fit_with_full_appearances(
    single_match_neutral_tables,
):
    report = assess_dataset_fit(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
        single_match_neutral_tables["appearances"],
    )

    assert report.overall_status == "good_fit"
    assert report.metrics["appearance_match_coverage"] == 1.0
    assert {check.status for check in report.checks} == {"pass"}


def test_assess_dataset_fit_warns_on_large_rosters_without_appearances(
    single_match_neutral_tables,
):
    report = assess_dataset_fit(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
    )

    assert report.overall_status == "usable_with_caution"
    assert any(
        check.name == "roster_fallback_risk" and check.status == "warn"
        for check in report.checks
    )


def test_assess_dataset_fit_flags_multi_team_membership_without_appearances():
    matches = pl.DataFrame(
        {
            "event_id": [1],
            "match_id": [10],
            "winner_id": [100],
            "loser_id": [200],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [1, 1, 1],
            "group_id": [100, 200, 100],
            "entity_id": [1, 1, 2],
        }
    )

    report = assess_dataset_fit(matches, participants)

    assert report.overall_status == "poor_fit"
    assert any(
        check.name == "multi_team_membership_without_appearances"
        and check.status == "fail"
        for check in report.checks
    )


def test_assess_dataset_fit_flags_missing_team_rosters(
    single_match_neutral_tables,
):
    participants = single_match_neutral_tables["participants"].filter(
        pl.col("group_id") != 11
    )
    report = assess_dataset_fit(
        single_match_neutral_tables["matches"],
        participants,
        single_match_neutral_tables["appearances"],
    )

    assert report.overall_status == "poor_fit"
    assert any(
        check.name == "team_rosters" and check.status == "warn"
        for check in report.checks
    )
    assert any(
        check.name == "entity_graph_connectivity" and check.status == "fail"
        for check in report.checks
    )


def test_assess_dataset_fit_allows_tiny_tail_of_unresolved_matches():
    matches = pl.DataFrame(
        {
            "event_id": [1] * 101,
            "match_id": list(range(101)),
            "winner_id": [100] * 101,
            "loser_id": ([200] * 100) + [300],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [1, 1],
            "group_id": [100, 200],
            "entity_id": [1, 2],
        }
    )

    report = assess_dataset_fit(matches, participants)

    assert report.overall_status == "good_fit"
    assert report.metrics["resolvable_match_count"] == 100
    assert report.metrics["entity_graph_component_count"] == 1
    assert any(
        check.name == "team_rosters" and check.status == "pass"
        for check in report.checks
    )
    assert any(
        check.name == "entity_graph_connectivity" and check.status == "pass"
        for check in report.checks
    )


def test_assess_dataset_fit_warns_on_partial_resolvable_coverage():
    matches = pl.DataFrame(
        {
            "event_id": [1] * 10,
            "match_id": list(range(10)),
            "winner_id": ([100] * 4) + ([100] * 6),
            "loser_id": ([200] * 4) + ([300] * 6),
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [1, 1],
            "group_id": [100, 200],
            "entity_id": [1, 2],
        }
    )

    report = assess_dataset_fit(matches, participants)

    assert report.overall_status == "usable_with_caution"
    assert report.metrics["resolvable_match_count"] == 4
    assert report.metrics["entity_graph_component_count"] == 1
    assert any(
        check.name == "team_rosters" and check.status == "warn"
        for check in report.checks
    )
    assert any(
        check.name == "entity_graph_connectivity" and check.status == "pass"
        for check in report.checks
    )


def test_assess_dataset_fit_warns_on_sparse_appearance_coverage():
    matches = pl.DataFrame(
        {
            "event_id": [1],
            "match_id": [10],
            "winner_id": [100],
            "loser_id": [200],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [1, 1],
            "group_id": [100, 200],
            "entity_id": [1, 2],
        }
    )
    appearances = pl.DataFrame(
        {
            "event_id": [1],
            "match_id": [10],
            "entity_id": [1],
            "group_id": [100],
        }
    )

    report = assess_dataset_fit(matches, participants, appearances)

    assert report.overall_status == "usable_with_caution"
    assert report.metrics["appearance_match_coverage"] == 0.5
    assert any(
        check.name == "appearance_match_coverage" and check.status == "warn"
        for check in report.checks
    )


def test_assess_dataset_fit_flags_fragmented_entity_graph():
    matches = pl.DataFrame(
        {
            "event_id": [1, 2],
            "match_id": [10, 20],
            "winner_id": [100, 300],
            "loser_id": [200, 400],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [1, 1, 2, 2],
            "group_id": [100, 200, 300, 400],
            "entity_id": [1, 2, 3, 4],
        }
    )

    report = assess_dataset_fit(matches, participants)

    assert report.overall_status == "poor_fit"
    assert report.metrics["entity_graph_component_count"] == 2
    assert any(
        check.name == "entity_graph_connectivity" and check.status == "fail"
        for check in report.checks
    )
