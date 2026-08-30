"""
BuildLab.py
===========
The bridge between NBA 2K's MyPLAYER builder and the real league.

Two jobs, both built ONLY on statistics this project already computes from
its own archive:

  1. DNA translation — take a real player-season's measured profile and
     phrase it as build priorities ("his diet was 61% rim: close shot and
     driving dunk before anything else"). The numbers are ours; the phrasing
     is editorial and every page that shows it says so.

  2. Comp finding — take a user's build sliders, convert each slider to a
     percentile TARGET over ~7,400 real qualified player-seasons (1,000+
     minutes, 1996-97 onward), and return the nearest real profiles by
     weighted z-scored distance. "Your build is 2016 Draymond with a worse
     handle" — descriptive resemblance, not a rating of the build.

WHAT THIS FILE REFUSES TO KNOW: anything about 2K's internals. No badge
thresholds, no animation requirements, no tested in-game outcomes — that is
other people's lab work and we neither copy it nor fake it. Slider names
here are generic basketball skills, not 2K attribute names.
"""

import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

# ── The comp pool ────────────────────────────────────────────────────────

POOL_MIN_MINUTES = 1000

# feature key -> (label, weight in the distance metric)
FEATURES: List[Tuple[str, str, float]] = [
    ("fg3m36", "threes made /36", 1.25),
    ("fg2m36", "twos made /36", 1.15),
    ("fta36", "free throws drawn /36", 0.7),
    ("ast36", "assists /36", 1.25),
    ("reb36", "rebounds /36", 1.25),
    ("stl36", "steals /36", 1.0),
    ("blk36", "blocks /36", 1.0),
    ("tov36", "turnovers /36", 0.5),
    ("height_in", "height", 1.6),
    ("weight_lb", "weight", 0.8),
]

_pool_cache: Optional[Dict[str, Any]] = None


def _parse_height(h: Optional[str]) -> Optional[float]:
    """'6-9' -> 81 inches; anything unparseable -> None."""
    if not h or "-" not in str(h):
        return None
    try:
        ft, inch = str(h).split("-", 1)
        return int(ft) * 12 + int(inch)
    except ValueError:
        return None


def _build_pool(conn: sqlite3.Connection) -> Dict[str, Any]:
    rows = conn.execute(
        """
        SELECT t.player_id, t.season, t.gp, t.min, t.fgm, t.fga, t.fg3m, t.fg3a,
               t.ftm, t.fta, t.reb, t.ast, t.stl, t.blk, t.tov, t.pts,
               p.full_name, p.height, p.weight
        FROM player_season_totals t
        JOIN players p ON p.player_id = t.player_id
        WHERE t.season_type = 'Regular Season' AND t.min >= ?
        """,
        (POOL_MIN_MINUTES,),
    ).fetchall()

    entries: List[Dict[str, Any]] = []
    for r in rows:
        height = _parse_height(r["height"])
        try:
            weight = float(r["weight"]) if r["weight"] else None
        except ValueError:
            weight = None
        if height is None or weight is None or not r["min"]:
            continue
        f = {
            "fg3m36": r["fg3m"] * 36.0 / r["min"],
            "fg2m36": (r["fgm"] - r["fg3m"]) * 36.0 / r["min"],
            "fta36": r["fta"] * 36.0 / r["min"],
            "ast36": r["ast"] * 36.0 / r["min"],
            "reb36": r["reb"] * 36.0 / r["min"],
            "stl36": r["stl"] * 36.0 / r["min"],
            "blk36": r["blk"] * 36.0 / r["min"],
            "tov36": r["tov"] * 36.0 / r["min"],
            "height_in": height,
            "weight_lb": weight,
        }
        entries.append({
            "player_id": r["player_id"],
            "name": r["full_name"],
            "season": r["season"],
            "gp": r["gp"],
            "pts36": r["pts"] * 36.0 / r["min"],
            "features": f,
        })

    # Per-feature sorted values (for percentile lookups) and mean/sd (for z).
    stats: Dict[str, Dict[str, Any]] = {}
    for key, _, _ in FEATURES:
        vals = sorted(e["features"][key] for e in entries)
        n = len(vals)
        mean = sum(vals) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) or 1.0
        stats[key] = {"sorted": vals, "mean": mean, "sd": sd}
    return {"entries": entries, "stats": stats}


def _pool(conn: sqlite3.Connection) -> Dict[str, Any]:
    global _pool_cache
    if _pool_cache is None:
        _pool_cache = _build_pool(conn)
    return _pool_cache


def _percentile_value(stats: Dict[str, Any], key: str, pct: float) -> float:
    vals = stats[key]["sorted"]
    idx = min(len(vals) - 1, max(0, int(pct * (len(vals) - 1))))
    return vals[idx]


def nearest_comps(
    conn: sqlite3.Connection,
    sliders: Dict[str, float],
    height_in: float,
    weight_lb: float,
    k: int = 5,
) -> Dict[str, Any]:
    """
    sliders: 25-99 values for three, inside, playmaking, rebounding, steals,
    blocks, plus a ball-security slider (higher = fewer turnovers).
    Each becomes a percentile target over the real pool; distance is weighted
    z-score euclidean. Height gets the heaviest weight because 2K builds are
    height-first and so is basketball.
    """
    pool = _pool(conn)
    stats = pool["stats"]

    def pct(v: float) -> float:
        return min(0.98, max(0.02, (v - 25.0) / 74.0))

    targets = {
        "fg3m36": _percentile_value(stats, "fg3m36", pct(sliders.get("three", 50))),
        "fg2m36": _percentile_value(stats, "fg2m36", pct(sliders.get("inside", 50))),
        "fta36": _percentile_value(stats, "fta36", pct(sliders.get("inside", 50))),
        "ast36": _percentile_value(stats, "ast36", pct(sliders.get("playmaking", 50))),
        "reb36": _percentile_value(stats, "reb36", pct(sliders.get("rebounding", 50))),
        "stl36": _percentile_value(stats, "stl36", pct(sliders.get("steals", 50))),
        "blk36": _percentile_value(stats, "blk36", pct(sliders.get("blocks", 50))),
        # ball security is inverted: a high slider means a LOW turnover rate
        "tov36": _percentile_value(stats, "tov36", 1.0 - pct(sliders.get("security", 50))),
        "height_in": height_in,
        "weight_lb": weight_lb,
    }

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for e in pool["entries"]:
        d = 0.0
        for key, _, w in FEATURES:
            z = (e["features"][key] - targets[key]) / stats[key]["sd"]
            d += w * z * z
        scored.append((d, e))
    scored.sort(key=lambda t: t[0])

    # One row per player: the closest season only, so the list reads as five
    # different comps rather than five seasons of one player.
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for d, e in scored:
        if e["player_id"] in seen:
            continue
        seen.add(e["player_id"])
        gaps = sorted(
            ((key, label, (e["features"][key] - targets[key]) / stats[key]["sd"])
             for key, label, _ in FEATURES if key not in ("height_in", "weight_lb")),
            key=lambda t: -abs(t[2]),
        )
        biggest = gaps[0]
        out.append({
            "player_id": e["player_id"],
            "name": e["name"],
            "season": e["season"],
            "distance": round(math.sqrt(d), 2),
            "gp": e["gp"],
            "pts36": round(e["pts36"], 1),
            "profile": {key: round(e["features"][key], 2) for key, _, _ in FEATURES},
            "biggest_gap": {
                "stat": biggest[1],
                "direction": "more" if biggest[2] > 0 else "less",
                "z": round(abs(biggest[2]), 1),
            },
        })
        if len(out) >= k:
            break

    return {
        "targets": {key: round(v, 2) for key, v in targets.items()},
        "pool_size": len(pool["entries"]),
        "comps": out,
        "method": (
            "Each slider becomes a percentile target over every qualified real "
            "player-season in our archive (1,000+ minutes, 1996-97 onward); comps "
            "are the closest real statistical profiles by weighted z-scored "
            "distance, height weighted hardest. This is a resemblance between "
            "your sliders and real production - it is not a rating of the build "
            "and knows nothing about 2K's internals."
        ),
    }


# ── DNA translation ──────────────────────────────────────────────────────

def per36(totals: Dict[str, Any]) -> Dict[str, float]:
    m = totals.get("min") or 0
    if not m:
        return {}
    f = lambda v: round((v or 0) * 36.0 / m, 1)
    return {
        "pts": f(totals.get("pts")), "reb": f(totals.get("reb")),
        "ast": f(totals.get("ast")), "stl": f(totals.get("stl")),
        "blk": f(totals.get("blk")), "tov": f(totals.get("tov")),
        "fg3m": f(totals.get("fg3m")), "fg3a": f(totals.get("fg3a")),
        "fta": f(totals.get("fta")),
    }


def build_notes(
    totals: Dict[str, Any],
    p36: Dict[str, float],
    sq: Optional[Dict[str, Any]],
    reb: Optional[Dict[str, Any]],
    clutch: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """The editorial layer: measured facts phrased as build priorities."""
    notes: List[Dict[str, str]] = []
    fga = totals.get("fga") or 0
    fg3a = totals.get("fg3a") or 0
    three_share = fg3a / fga if fga else 0

    if sq and sq.get("rim_share") is not None:
        rim = sq["rim_share"]
        if rim >= 0.5:
            notes.append({"area": "Scoring diet", "fact": f"{rim:.0%} of his shots came inside 10 feet",
                          "note": "Finishing first: prioritize close-range scoring and vertical before any jumper."})
        elif rim <= 0.25:
            notes.append({"area": "Scoring diet", "fact": f"only {rim:.0%} of his shots came inside 10 feet",
                          "note": "A perimeter diet - the jumper carries this build; don't overspend inside."})
        else:
            notes.append({"area": "Scoring diet", "fact": f"{rim:.0%} rim / {1-rim:.0%} outside",
                          "note": "A mixed diet: balance finishing and shooting rather than maxing either."})
        if sq.get("soe_per100") is not None:
            soe = sq["soe_per100"]
            if soe >= 8:
                notes.append({"area": "Shot-making", "fact": f"+{soe:.0f} points per 100 shots over expectation",
                              "note": "He beat his own looks - tough-shot ability was real, not diet."})
            elif soe <= -8:
                notes.append({"area": "Shot-making", "fact": f"{soe:.0f} points per 100 shots vs expectation",
                              "note": "Production leaned on quality looks; recreate the LOOKS, not hero shots."})
    elif fga:
        if three_share >= 0.45:
            notes.append({"area": "Scoring diet", "fact": f"{three_share:.0%} of his attempts were threes",
                          "note": "Perimeter-first build: the three-ball is the foundation."})
        elif three_share <= 0.12:
            notes.append({"area": "Scoring diet", "fact": f"only {three_share:.0%} of attempts were threes",
                          "note": "Pre-spacing-era interior game: finishing, post moves, and free throws."})

    if p36.get("fta", 0) >= 7:
        notes.append({"area": "Downhill pressure", "fact": f"{p36['fta']} free throws per 36",
                      "note": "He lived at the line - strength and driving contact matter as much as touch."})
    if p36.get("ast", 0) >= 7:
        notes.append({"area": "Playmaking", "fact": f"{p36['ast']} assists per 36",
                      "note": "Primary-creator passing: ball-handle and pass accuracy are core, not garnish."})
    elif p36.get("ast", 0) <= 2 and fga:
        notes.append({"area": "Playmaking", "fact": f"{p36['ast']} assists per 36",
                      "note": "A finisher, not a hub - playmaking is a place to save build points."})

    if reb:
        cs = reb.get("reb", {}).get("contest_pct")
        if cs is not None and cs >= 0.4:
            notes.append({"area": "The glass", "fact": f"{cs:.0%} of his boards were contested",
                          "note": "He won rebounds in traffic - strength and rebounding both, not just height."})
        elif cs is not None and cs <= 0.2:
            notes.append({"area": "The glass", "fact": f"only {cs:.0%} of his boards were contested",
                          "note": "Positioning rebounder: box-out IQ over max rebounding stat."})

    if p36.get("stl", 0) >= 1.8:
        notes.append({"area": "Defense", "fact": f"{p36['stl']} steals per 36",
                      "note": "Ball-pressure defender: perimeter defense and steal, hands-first."})
    if p36.get("blk", 0) >= 1.8:
        notes.append({"area": "Defense", "fact": f"{p36['blk']} blocks per 36",
                      "note": "Rim protection is identity - block and interior defense before offense."})

    if clutch and clutch.get("fga", 0) >= 25:
        diff = clutch.get("fg_diff")
        notes.append({
            "area": "Clutch",
            "fact": f"{clutch['pts']} clutch points on {clutch['fgm']}/{clutch['fga']} shooting",
            # Within 3pp of his own baseline counts as holding up: clutch
            # samples are ~100 shots and a rounding-level dip is not a story.
            "note": ("His late-game shooting held up against his own season baseline."
                     if diff is None or diff >= -0.03 else
                     "Late-game volume was real but efficiency dipped - clutch reps came with a cost."),
        })

    return notes
