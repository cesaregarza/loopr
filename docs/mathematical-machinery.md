# Mathematical Machinery

This page explains the math behind `loopr` in a self-contained way.

The intended reader is a technically literate engineer who knows basic graphs, PageRank, and linear algebra, but has not read the rest of the repository. The goal is to explain what the system computes, why leave-one-out analysis is possible, and what the fast approximation is doing.

## The Problem We Are Solving

We want to rank individuals from grouped outcomes.

Examples:

- a team match where one side beats another
- a set of matches where several players contribute to the result
- a community leaderboard where the observed data is really a collection of shared outcomes, not direct one-vs-one comparisons

The raw data is usually not a simple “player A beat player B” table. It is more like:

- this event happened
- these players participated
- one side won
- the event may carry a weight such as recency or tournament importance

`loopr` turns that into a directed graph and then runs a PageRank-like computation to estimate individual strength.

## The Basic Data Model

For each event, assume we know:

- the event weight `w >= 0`
- the set of winners `W`
- the set of losers `L`
- optionally, extra weighting from time decay or tournament influence

The event is converted into pairwise winner-loser relations.

It is important to be explicit about matrix convention here.

- the raw pair table is stored as `(winner, loser)`
- on the **win graph**, that means `row = winner`, `col = loser`
- after column normalization, a loser column distributes support into the winner rows that beat that loser

So when you later read that “a node gains support from incoming mass,” the picture is:

- loser columns point their mass into winner rows on the win graph
- winner columns point their mass into loser rows on the loss graph

If there are `m` winners and `n` losers, the event contributes across `m * n` winner-loser pairs.

The important point is that the event is still the unit of meaning. The pairwise edges are just a convenient representation for the PageRank-style solver.

## PageRank View Of The Problem

Let `A` be a column-stochastic transition matrix built from the directed graph.

Let `alpha` be the damping factor, with `0 < alpha < 1`.

Let `rho` be the teleport vector, which acts like a prior or restart distribution.

Then the PageRank-style score vector `s` is the fixed point of

```text
s = alpha A s + (1 - alpha) rho
```

Equivalently,

```text
(I - alpha A) s = (1 - alpha) rho
```

This is the standard linear system form. The score of a player increases when they receive more incoming support from other players who themselves are well supported.

`loopr` keeps two coupled graphs:

- a “win” graph
- a “loss” graph

The final score is the log ratio between the two PageRank vectors.

## Win And Loss Scores

Let `s_win` be the PageRank vector computed on the win graph.

Let `s_loss` be the PageRank vector computed on the loss graph.

The core ranking idea is a log ratio between win-side support and loss-side support, but the implementation uses a smoothed version:

```text
score_i
= log(max(s_win_i, f_i) + lambda rho_i)
- log(max(s_loss_i, f_i) + lambda rho_i)
```

with

```text
f_i = 0.5 (1 - alpha) rho_i
```

and `lambda >= 0` a smoothing constant.

This has a useful interpretation:

- if a player is supported more strongly in the win graph than the loss graph, their score is positive
- if the opposite is true, their score is negative
- the log ratio keeps the scale symmetric and easy to compare
- the `rho`-based floor and smoothing keep the score numerically stable for sparse or weakly connected nodes

So the rough mental model is still “win PageRank minus loss PageRank on a log scale,” but the `rho`-anchored smoothing is an important part of why the score behaves well in practice.

## Why Leave-One-Out Matters

Leave-one-out analysis asks a simple question:

> If we remove one event, how much would the rankings change?

This is useful for two reasons:

1. It tells you which events matter most.
2. It gives you a principled way to explain a ranking.

The naive way to answer this question would be to rebuild the entire graph and rerun PageRank from scratch after removing each event.

That is too expensive if you want to do this many times.

The key observation is that removing one event changes only a small part of the transition matrix and the right-hand side.

## Exact Leave-One-Out As A Low-Rank Update

Suppose the original linear system is

```text
K s = b
```

where

```text
K = I - alpha A
```

and `A` is the transition matrix for the full graph.

If we remove one event, the matrix changes by a small update:

```text
A' = A - DeltaA
```

so the new system becomes

```text
K' s' = b'
```

with

```text
K' = K + DeltaK
```

where `DeltaK = alpha DeltaA`.

The clean conceptual picture is:

```text
delta s_e = s' - s = (K + DeltaK_e)^-1 (Delta b_e - DeltaK_e s)
```

So one event creates a local residual

```text
r_e = Delta b_e - DeltaK_e s
```

and the system propagates that residual through the graph.

In practice, the event update touches only a few source columns and has small support relative to the full graph. That makes it natural to use a low-rank or small-effective-rank correction instead of refactorizing the whole system.

### Woodbury / Sherman-Morrison Style Idea

Write the update as

```text
K' = K - U V^T
```

for some low-rank factors `U` and `V`.

If we already know how to apply `K^-1`, then

```text
(K - U V^T)^-1
```

can be computed through a small dense correction system rather than a full sparse refactorization.

The exact update has the form

```text
(K - U V^T)^-1
= K^-1 + K^-1 U (I - V^T K^-1 U)^-1 V^T K^-1
```

The important computational pattern is:

1. apply the existing sparse solver to a small set of right-hand sides
2. solve a tiny dense system
3. combine the result to get the updated score

That is the exact leave-one-out method.

## What Gets Solved In Practice

The update needs two kinds of solves:

- one for the local graph correction term
- one for the teleport or prior correction term

If those are solved separately, the exact method is still correct, but it does more sparse-solver work.

The reduced-solve exact path combines multiple right-hand sides into one sparse solve per graph, which keeps the math exact but reduces overhead.

In the actual implementation, the exact path does this:

1. build a small matrix `U` of per-column update directions for the match removal
2. optionally build a teleport correction vector `v`
3. apply the resolvent `(I - alpha A)^-1` to `U` and `v`
4. solve the resulting small dense correction system

The approximate path keeps steps 1, 2, and 4, but replaces step 3 with a truncated propagation approximation.

So there are three distinct ideas here:

1. **Exact leave-one-out**
   - use the exact inverse identity
   - preserve the exact score change

2. **Reduced-solve exact leave-one-out**
   - same exact math
   - fewer sparse solves because multiple right-hand sides are handled together

3. **Approximate perturbation leave-one-out**
   - skip the exact sparse inverse application
   - approximate the inverse with a short series expansion

## The Perturbation Approximation

The exact method still depends on applying `K^-1` to one or more right-hand sides.

The perturbation method replaces that expensive inverse with a truncated Neumann series:

```text
(I - alpha A)^-1
= I + alpha A + alpha^2 A^2 + alpha^3 A^3 + ...
```

If we stop after `m` terms, we get

```text
(I - alpha A)^-1 \approx \sum_{k=0}^{m-1} (alpha A)^k
```

This is the approximation used for the fast explanation modes.

For a column-stochastic `A`, we have `||A||_1 = 1`, so for any residual vector `r` the truncated tail obeys the simple bound

```text
|| sum_{t=m}^{infinity} (alpha A)^t r ||_1
<= (alpha^m / (1 - alpha)) ||r||_1
```

This is not, by itself, a full bound on exact-versus-approximate LOO score error, because the complete LOO update also includes the small dense correction step. But it does give a principled explanation for why a short propagation truncation can work well in a damped PageRank system.

### What This Means Intuitively

Instead of solving the exact global system, we approximate how a local change propagates through the graph by allowing only a small number of propagation steps.

That is often good enough when:

- the damping factor is below 1
- the removed event is small relative to the whole graph
- you only care about the top-ranked explanations, not machine-precision exactness

### Why It Helps

Removing one event is usually a small perturbation of a stable PageRank system.

So the highest-impact events tend to stay the same even if you approximate the response.

That makes the perturbation method a good fit for:

- top-20 or top-50 explanations
- interactive diagnostics
- fast candidate ranking before an exact final pass

It is not the right choice if you need exact leave-one-out scores for every single event.

## Exact Versus Approximate Leave-One-Out

The distinction is important.

### Exact leave-one-out

Use this when you need faithful score changes.

- deterministic
- mathematically exact up to floating-point error
- slower
- best for final reports or validation

### Perturbation leave-one-out

Use this when you need speed and the exact tail of the ranking does not matter much.

- approximate
- much faster
- usually preserves the top-ranked explanations well
- best for exploratory analysis and candidate filtering

In practice, the fast path can rank candidate events cheaply, and then the exact path can be applied only to the most interesting candidates.

## Why The Approximation Is Often Good Enough

There are a few reasons the approximation works well in this setting:

- PageRank damps long paths, so far-away propagation fades quickly.
- The event being removed is usually small compared with the full graph.
- Explanation tasks care most about relative ordering near the top.
- The low-rank correction still captures the local structure of the change.

This is why a short perturbation series can preserve the top-K event ordering even when it does not match exact deltas perfectly.

## Complexity Intuition

The costs break down roughly like this:

- building the graph is linear in the amount of input data
- exact leave-one-out is dominated by sparse linear solves
- reduced-solve exact leave-one-out saves some solver calls, but does not change the algorithmic class
- perturbation leave-one-out trades exact solves for sparse matrix-vector multiplies

That is why the benchmark results usually show:

- exact path: slower but faithful
- perturbation path: much faster and often good enough for ranking explanations

## Practical Guidance

If you are using the system as an explanation tool:

- use the exact method when you need definitive numbers
- use the perturbation method when you need to shortlist events quickly
- apply the exact method only to the shortlisted events if you want a final audit

If you are only comparing overall rankings:

- the exact ranking engine is the reference
- the perturbation method is not meant to replace the main ranking

## Summary

The math in `loopr` is built around three ideas:

1. convert grouped outcomes into a directed weighted graph
2. score individuals with a PageRank-style win/loss log ratio
3. explain the effect of removing one event with either exact low-rank updates or a faster perturbation approximation

The exact method is faithful but expensive.
The perturbation method is approximate but much faster, and it is often good enough for ranking the most important explanations.

## Related Reading

- `docs/how-loopr-works.md`
- `docs/analysis-and-diagnostics.md`
- `docs/validation-and-benchmarks.md`
