# Sendou Plus

This case study adapts a real validation report from the SendouQ analysis
project into a compact `loopr` example.

It is intentionally domain-specific. The point is not that every `loopr`
deployment will look like this, but that this is one concrete example of how
the library was evaluated against an external human signal.

## Context

The target domain was competitive Splatoon tournament data from `sendou.ink`.

The external validation signal was Plus-server peer voting:

- `+1`: highly skilled, often more selective or semi-retired
- `+2`: very strong and typically active
- `+3`: strong, but broader and more heterogeneous

Each voting row represented a nominated candidate within a Plus tier, with a
pass/fail outcome determined by peer votes. The ranking model did not see those
votes during fitting; they were used only for evaluation.

The ranking side used an outcomes-only `loopr`-style signal:

- ranked tournaments only
- temporal freeze before the vote snapshot cutoff
- active-only filter of at most 90 days since last activity
- exposure-log-odds ranking with exposure-based teleport
- time decay
- tick-tock tournament influence

The comparison set included:

- LOOPR
- OpenSkill ordinal
- OpenSkill `mu`
- simple activity baselines such as tournaments played, matches played, and
  90-day win rate

The primary evaluation metric was ROC AUC, with bootstrap confidence intervals.

## Results

### Main Alignment Result

Across active-only rows, the outcomes-only LOOPR signal aligned well with Plus
peer voting:

- macro AUC: `0.781` with 95% CI `0.729–0.830`
- sample-weighted macro AUC: `0.781` with 95% CI `0.736–0.824`
- micro AUC: `0.728` with 95% CI `0.681–0.775`

Per-tier AUCs were:

| Tier | AUC | 95% CI | n |
|---|---:|:---:|---:|
| `+1` | 0.7489 | 0.617–0.864 | 73 |
| `+2` | 0.8331 | 0.757–0.898 | 131 |
| `+3` | 0.7620 | 0.696–0.825 | 228 |

The strongest separation appeared in `+2`. `+1` was noisier, which is
consistent with a smaller and more selective pool.

### Baseline Comparison

On the same active-only window, LOOPR outperformed the non-LOOPR baselines used
in the original report.

OpenSkill coverage was slightly smaller (`n = 431`), but still enough for a
useful comparison:

- OpenSkill ordinal overall AUC: `0.610` with 95% CI `0.555–0.656`
- OpenSkill `mu` overall AUC: `0.634` with 95% CI `0.582–0.681`

On the active-only intersection set, the reported macro-AUC deltas were:

- LOOPR minus OpenSkill ordinal: about `+0.184`
- LOOPR minus OpenSkill `mu`: about `+0.153`

Simple activity-only baselines also underperformed:

| Baseline | AUC | n |
|---|---:|---:|
| `tournaments_90d` | 0.437 | 432 |
| `matches_90d` | 0.476 | 431 |
| `winrate_90d` | 0.602 | 431 |

That is a useful sanity check: the result was not just rediscovering who had
played the most recently or the most often.

### Ablation Snapshot

The originating project also ran a compact ablation table. A few rows are
especially informative:

| Variant | Tier agreement (macro AUC) | Stability (Spearman@100) | Prediction (log loss) |
|---|---:|---:|---:|
| Forward PR only | 0.622 | 0.943 | 0.691 |
| LOOPR (no decay) | 0.754 | 0.954 | 0.526 |
| LOOPR + exposure teleport | 0.772 | 0.959 | 0.498 |
| LOOPR + tick-tock | 0.779 | 0.953 | 0.499 |
| OpenSkill `mu` (native) | 0.584 | 0.876 | 0.576 |
| Winrate 90d | 0.470 | 0.649 | 0.639 |

The main pattern is that the full LOOPR-style design materially improved the
agreement metric over simpler PageRank-only and activity-only baselines, while
remaining competitive on stability and predictive loss.

### What This Case Study Shows

This case study is useful because it demonstrates a full external-validation
loop:

- the ranking signal was built only from outcomes
- the evaluation target came from an independent human process
- the comparison included both stronger model baselines and weaker activity
  baselines

That makes it a good example of the kind of evidence `loopr` benefits from when
used inside a domain project.

### How This Maps To The Generic Validation Harness

In the validation-harness terms used elsewhere in the docs, this case study
looks like:

- dataset bundle: tournament outcomes, rosters, and appearances
- external target: Plus peer-vote outcomes
- split policy: temporal freeze before the vote snapshot
- filter policy: active-only evaluation window
- baseline: default LOOPR-style outcomes-only signal
- variants: simpler PageRank-style, no-decay, exposure-teleport, tick-tock, and
  non-`loopr` baselines
- outputs: AUC summary, per-tier breakdown, and compact ablation table

That mapping is important because it shows how a domain-specific report can fit
inside a reusable evaluation contract instead of remaining a one-off writeup.

## Related Reading

- [../validation-harness.md](../validation-harness.md)
- [../validation-and-benchmarks.md](../validation-and-benchmarks.md)
- [../ablations.md](../ablations.md)
- [../engines-and-configuration.md](../engines-and-configuration.md)
