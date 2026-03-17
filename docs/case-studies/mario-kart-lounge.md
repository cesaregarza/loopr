# Mario Kart Lounge Mogis

This case study shows what `loopr`'s positional-result path looks like on a
real ordered-finish dataset.

Unlike the Sendou Plus case study, this is not an external-validation report.
It is a same-source case study focused on:

- converting messy public table results into positional `loopr` inputs
- running the exposure-log-odds graph pipeline on those finishes
- checking whether the resulting ranking behaves sensibly under different
  positional weighting choices

## Context

The source data was scraped locally from public MK8DX Lounge table pages for
Seasons `14` and `15`.

Those raw pages are naturally positional rather than binary:

- one row per finisher
- exactly `12` finishers per table
- repeated placements in team formats such as `2v2`, `3v3`, `4v4`, and `6v6`
- per-player score and MMR-change fields on each table row

This makes them a better fit for `loopr`'s positional interpretation than for
the main public binary group-result path.

The local conversion used:

- `event_id = season`
- `match_id = table_id`
- `entity_id = player_id`
- `completed_at = time_verified`, falling back to `time_created`

The ranking side used the internal exposure-log-odds pipeline with these
choices:

- time decay enabled with `half_life_days = 30`
- post-ranking inactivity decay enabled with the default delay/rate
- tournament influence disabled with `beta = 0`
- simple active slice defined as activity within `90` days of the latest scraped
  table
- an output hygiene filter of `min_exposure = 1.0`

That `beta = 0` choice is important. Lounge mogi tables are not naturally
organized into the sort of tournament hierarchy that `tick-tock` influence is
trying to model, so the case study keeps the signal focused on positional match
exposure rather than event-strength feedback loops.

Two positional weighting modes were compared:

- `pairwise_average`
- `pairwise_full`

## Results

### Dataset Shape

The converted dataset contained:

- `31,083` parsed tables
- `372,996` finisher rows
- `9,305` distinct players appearing in parsed tables

Season split:

| Season | Tables | Finisher Rows |
|---|---:|---:|
| `14` | 24,167 | 290,004 |
| `15` | 6,916 | 82,992 |

Format mix:

| Format | Tables |
|---|---:|
| `2v2` | 20,764 |
| `FFA` | 4,647 |
| `3v3` | 3,796 |
| `4v4` | 1,538 |
| `6v6` | 338 |

One practical lesson from this case study is that the raw lounge pages needed
more careful parsing than a fixed-width table extractor. Some rows contained a
`Multiplier` token while adjacent rows in the same table did not. The local
pipeline had to handle that per-row optional field before the final dataset
cleanly resolved to `12` rows per parsed table.

### Positional Weighting Sensitivity

After the `min_exposure = 1.0` filter:

- `pairwise_average` ranked `1,882` entities
- `pairwise_full` ranked `5,616` entities

Across common ranked entities, the two scoring modes were almost perfectly
correlated:

- all-entity Spearman: `0.999638`
- active-only Spearman: `0.999639`

But the exposed top of the ranking moved much more than that correlation
suggests:

- all-entity top-100 overlap: `0.38`
- active-only top-100 overlap: `0.38`

That is the main modeling result from this case study.

`pairwise_full` and `pairwise_average` agree on the broad ordering, but in a
dataset with many tied/team placements they can disagree materially about which
high-exposure players belong near the very top.

Qualitatively, `pairwise_full` tended to promote more low-volume recent spikes,
while `pairwise_average` produced a more conservative top slice once an
exposure floor was applied. That makes `pairwise_average` a better default
candidate for this kind of positional team-table data.

### Same-Source Sanity Check Against Lounge MMR

This is not an external validation target. Lounge MMR is derived from the same
underlying ecosystem, not from an independent human or downstream outcome
process.

Still, as a sanity check on the active Season `15` slice, the
`pairwise_average` ranking aligned fairly well with the latest leaderboard
snapshot:

- common active entities in Season `15`: `392`
- Spearman(score, leaderboard MMR): `0.94857`
- top-100 overlap with leaderboard MMR: `0.85`

That does not prove the ranking is better than the native lounge system. It
does show that the positional `loopr` signal is not behaving wildly out of
distribution relative to a different same-domain rating signal.

### Top Active Players (`pairwise_average`)

These are the top active players in the filtered `pairwise_average` run:

| Rank | Player | Score | Exposure | Tables | Lounge MMR |
|---|---|---:|---:|---:|---:|
| `1` | `RVL I see` | 1.6755 | 4.3523 | 95 | 15165 |
| `2` | `WeeklyShonenJump` | 1.6481 | 25.5003 | 811 | 15312 |
| `3` | `Postscript` | 1.6288 | 5.2877 | 89 | 16098 |
| `4` | `hukurou` | 1.6273 | 8.1690 | 332 | 14845 |
| `5` | `TZUYU` | 1.5311 | 13.4595 | 892 | 14895 |
| `6` | `KING OF THE KILL` | 1.5046 | 18.3226 | 409 | 15862 |
| `7` | `VINI JR.` | 1.4808 | 6.6792 | 272 | 14458 |
| `8` | `Nakamura Shido` | 1.4686 | 8.8032 | 623 | 14030 |
| `9` | `SuperhumanHD` | 1.4523 | 2.5506 | 52 | 14489 |
| `10` | `Fuyuki` | 1.3920 | 5.7790 | 89 | 13926 |

The interesting part of that table is not the exact order. It is that the top
slice looks exposure-supported after filtering, rather than being dominated by
one-table spikes.

### What This Case Study Shows

This case study is useful for three reasons:

- it exercises `loopr` on a genuinely positional real-world dataset rather than
  a binary team-result dataset
- it shows that pairwise expansion details matter in tied/team finish data
- it demonstrates that helper-level positional support is already enough to run
  a real ranking workflow, even before positional inputs are wired into the
  main `rank_entities(...)` public path

### Limitations

- This is not an external-validation case study.
- The raw acquisition tooling is local-only and intentionally kept outside the
  published package.
- The current public `LOOPREngine.rank_entities(...)` path still documents
  binary group-result inputs first; this case study uses the helper-level
  positional graph-prep path.

## Related Reading

- [../result-modes.md](../result-modes.md)
- [../validation-harness.md](../validation-harness.md)
- [../validation-and-benchmarks.md](../validation-and-benchmarks.md)
- [sendou-plus.md](sendou-plus.md)
