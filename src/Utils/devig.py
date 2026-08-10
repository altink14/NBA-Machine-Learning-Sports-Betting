"""
devig.py
========
Remove the bookmaker's margin ("vig") from a set of quoted odds to recover the
market's fair opinion of each outcome's probability.

Raw implied probability (1 / decimal odds) overstates every outcome because
the quotes sum to more than 100% — that surplus is the book's margin. Any code
that treats raw implied probability as "what the market thinks" is biased;
de-vig first.

Two methods:

- Shin (preferred): models the margin as the book protecting itself against
  insider traders, which places proportionally more of the margin on longshots.
  This matches the well-documented favourite-longshot bias, so Shin fair
  probabilities beat plain normalisation empirically, especially on lopsided
  two-outcome markets (e.g. NBA moneyline pairs). Backed by the `shin` package
  (mberk/shin, MIT, Rust-accelerated).
- Multiplicative (fallback): divide each implied probability by their sum.
  Simple, always available, and exact when the margin really is spread
  proportionally.

IMPORTANT: de-vig only replaces the PROBABILITY estimate. The payout of a bet
is still governed by the QUOTED odds — never de-vig the payout side of an EV
calculation.

If the `shin` package is not installed the module still imports and
`fair_probs()` silently uses the multiplicative method (same pattern as the
optional slowapi dependency in main_api.py). `ACTIVE_METHOD` reports which
method `fair_probs()` will use, so callers can log it at startup.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

try:
    import shin as _shin
    SHIN_AVAILABLE = True
except ImportError:
    _shin = None
    SHIN_AVAILABLE = False

ACTIVE_METHOD = "shin" if SHIN_AVAILABLE else "multiplicative"


def _validate_decimal_odds(decimal_odds: List[float]) -> List[float]:
    if decimal_odds is None or len(decimal_odds) < 2:
        raise ValueError("De-vigging needs the quoted odds of at least two outcomes.")
    odds = [float(d) for d in decimal_odds]
    for d in odds:
        if d <= 1.0:
            raise ValueError(f"Decimal odds must be greater than 1.0 (got {d:g}).")
    return odds


def fair_probs_multiplicative(decimal_odds: List[float]) -> List[float]:
    """
    Fair probabilities via proportional normalisation: each raw implied
    probability divided by the overround. Always available; used as the
    documented fallback when the shin package is missing.
    """
    odds = _validate_decimal_odds(decimal_odds)
    implied = [1.0 / d for d in odds]
    total = sum(implied)
    return [p / total for p in implied]


def fair_probs_shin(decimal_odds: List[float]) -> List[float]:
    """
    Fair probabilities via Shin's (1992/1993) insider-trading model, using the
    `shin` package. For two-outcome markets this has a closed-form solution.
    Raises RuntimeError if the shin package is not installed.
    """
    odds = _validate_decimal_odds(decimal_odds)
    if not SHIN_AVAILABLE:
        raise RuntimeError("The 'shin' package is not installed. Run: pip install shin")
    return [float(p) for p in _shin.calculate_implied_probabilities(odds)]


def fair_probs(decimal_odds: List[float]) -> List[float]:
    """
    Best-available de-vig: Shin when the package is importable, multiplicative
    otherwise (or if shin errors on a pathological input, e.g. quotes with no
    overround). Input validation errors are raised either way.
    """
    odds = _validate_decimal_odds(decimal_odds)
    if SHIN_AVAILABLE:
        try:
            return fair_probs_shin(odds)
        except ValueError:
            raise
        except Exception as exc:
            logger.warning(f"Shin de-vig failed ({exc}); falling back to multiplicative.")
    return fair_probs_multiplicative(odds)
