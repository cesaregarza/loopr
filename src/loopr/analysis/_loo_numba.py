"""Optional numba kernels for LOO experiments.

These are intentionally narrow helper kernels. The exact LOO hot path is
mostly dominated by sparse linear solves, so numba is only applied to the
small dense loops we can swap in and out safely.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False
else:  # pragma: no cover - exercised only when numba is installed
    NUMBA_AVAILABLE = True


if NUMBA_AVAILABLE:  # pragma: no cover - exercised only when numba is installed

    @njit(cache=True)
    def delta_rho_numba(
        participant_indices: np.ndarray,
        sigma: float,
        n: int,
    ) -> np.ndarray:
        delta = np.zeros(n, dtype=np.float64)
        for idx in participant_indices:
            delta[idx] -= sigma
        return delta

    @njit(cache=True)
    def estimate_match_flux_numba(
        entity_idx: int,
        winners: np.ndarray,
        losers: np.ndarray,
        pair_weight: float,
        alpha: float,
        s_win: np.ndarray,
        s_loss: np.ndarray,
        t_win: np.ndarray,
        t_loss: np.ndarray,
    ) -> float:
        total = 0.0
        is_winner = False
        is_loser = False

        for idx in winners:
            if idx == entity_idx:
                is_winner = True
                break

        for idx in losers:
            if idx == entity_idx:
                is_loser = True
                break

        if is_winner:
            for loser_idx in losers:
                denom = t_win[loser_idx]
                if denom > 0.0:
                    total += alpha * pair_weight * (s_win[loser_idx] / denom)
        elif is_loser:
            denom = t_win[entity_idx]
            if denom > 0.0:
                total += (
                    winners.size
                    * alpha
                    * s_win[entity_idx]
                    * (pair_weight / denom)
                )

        if is_loser:
            for winner_idx in winners:
                denom = t_loss[winner_idx]
                if denom > 0.0:
                    total += alpha * pair_weight * (
                        s_loss[winner_idx] / denom
                    )
        elif is_winner:
            denom = t_loss[entity_idx]
            if denom > 0.0:
                total += (
                    losers.size
                    * alpha
                    * s_loss[entity_idx]
                    * (pair_weight / denom)
                )

        return total

