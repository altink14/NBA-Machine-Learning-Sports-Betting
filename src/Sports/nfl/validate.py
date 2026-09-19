"""
validate.py (NFL)
=================
The archive validation suite. It runs on every ingest and it is a shipping
gate: a sport whose suite is not green does not get a model, a page or a
prediction.

Checks are of three kinds:

  STRUCTURAL   things that must be true of any archive: no duplicate games, no
               team playing itself, no game whose teams are not in the team
               table, no final score dated in the future.
  COVERAGE     row counts against the schedule the NFL actually played, and
               market-line coverage per season, so a silent partial download
               cannot masquerade as a complete archive.
  SEMANTIC     the checks that catch the errors which do not look like errors:
               above all the SIGN of the spread. A flipped spread sign produces
               a perfectly well-formed archive in which every downstream number
               is backwards, and no structural check would ever notice.

Exit code 0 when every check passes, 1 when any FAIL. WARNs do not fail the
run but are printed.

Usage:
    venv/Scripts/python.exe src/Sports/nfl/validate.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import date

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DB_PATH = os.path.join(REPO_ROOT, "Data", "NflData.sqlite")

#: Regular-season games the NFL actually played, by season. 1999-2020 is a
#: 16-game season (256 games); 2021 onward is 17 games (272). 1999-2001 had 31
#: teams, so 248; 2002 expanded to 32. Playoffs: 11 games through 2019, 13 from
#: 2020 (expanded field), 13 thereafter.
EXPECTED_REG = {**{y: 248 for y in range(1999, 2002)},
                **{y: 256 for y in range(2002, 2021)},
                **{y: 272 for y in range(2021, 2027)},
                # 2022 played 271: Bills at Bengals on 2023-01-02 was abandoned
                # after Damar Hamlin's cardiac arrest and never replayed, so
                # BUF and CIN each played 16. Verified in our own rows: week 17
                # holds 15 games and exactly those two teams are short one.
                2022: 271}
EXPECTED_POST = {**{y: 11 for y in range(1999, 2020)},
                 **{y: 13 for y in range(2020, 2027)}}

#: Games the nflverse play-by-play release simply does not contain. Verified
#: 2026-09-19 by counting distinct game_ids in the source files themselves:
#: play_by_play_1999.csv holds 258 games and play_by_play_2000.csv holds 257,
#: which is exactly what we ingested, so nothing was dropped on our side.
PBP_MISSING_UPSTREAM = {
    "1999_01_BAL_STL",
    "2000_03_SD_KC",
    "2000_06_BUF_MIA",
}

#: Games whose play-by-play running score does not reconcile with the final
#: score. Sixteen of the seventeen are Jacksonville HOME games in 2001 and
#: 2002, where the combined total is right but the home/away attribution
#: inside the drive-level scoring is wrong: 2001_01_PIT_JAX finished 21-3 and
#: the play-by-play ends 3-21. This is an upstream defect in nflfastR's
#: retro-applied early-season data, not a parsing error here. The seventeenth,
#: 2011_13_DET_NO, is a different fault: both sides run exactly six points
#: high (37-23 against a 31-17 final).
#:
#: These ids are pinned rather than the count being relaxed, so a NEW
#: reconciliation failure anywhere still fails the suite. Any page computing
#: a score-dependent figure from play-by-play must exclude them.
PBP_SCORE_MISMATCH_UPSTREAM = {
    "2001_01_PIT_JAX", "2001_02_TEN_JAX", "2001_03_CLE_JAX", "2001_06_BUF_JAX",
    "2001_09_CIN_JAX", "2001_11_BAL_JAX", "2001_12_GB_JAX", "2001_16_KC_JAX",
    "2002_01_IND_JAX", "2002_04_NYJ_JAX", "2002_05_PHI_JAX", "2002_08_HOU_JAX",
    "2002_10_WAS_JAX", "2002_13_PIT_JAX", "2002_14_CLE_JAX", "2002_16_TEN_JAX",
    "2011_13_DET_NO",
}

FAILS: list[str] = []
WARNS: list[str] = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    if ok:
        print(f"  PASS  {name}" + (f" - {detail}" if detail else ""))
        return
    (WARNS if warn_only else FAILS).append(f"{name}: {detail}")
    print(f"  {'WARN' if warn_only else 'FAIL'}  {name} - {detail}")


def main() -> int:
    if not os.path.exists(DB_PATH):
        print(f"No database at {DB_PATH}")
        return 1
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    q = lambda sql, *a: conn.execute(sql, a).fetchall()  # noqa: E731
    one = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731

    print("\n=== STRUCTURAL ===")
    total = one("SELECT COUNT(*) FROM games")
    check("games present", total > 7000, f"{total} games")
    check("no duplicate game ids", one("SELECT COUNT(*) - COUNT(DISTINCT game_id) FROM games") == 0)
    check("no team plays itself", one("SELECT COUNT(*) FROM games WHERE home_team_id = away_team_id") == 0)
    orphans = one("""SELECT COUNT(*) FROM games g WHERE NOT EXISTS
                     (SELECT 1 FROM teams t WHERE t.team_id = g.home_team_id)
                      OR NOT EXISTS (SELECT 1 FROM teams t WHERE t.team_id = g.away_team_id)""")
    check("every game's teams are in the team table", orphans == 0, f"{orphans} orphans")
    future = one("SELECT COUNT(*) FROM games WHERE status = 'final' AND local_date > ?", date.today().isoformat())
    check("no final score dated in the future", future == 0, f"{future} games")
    check("no null local_date", one("SELECT COUNT(*) FROM games WHERE local_date IS NULL") == 0)
    both_or_neither = one("""SELECT COUNT(*) FROM games
                             WHERE (home_score IS NULL) != (away_score IS NULL)""")
    check("scores are present in pairs", both_or_neither == 0, f"{both_or_neither} half-scored games")

    print("\n=== COVERAGE ===")
    per_season = {(r["season"], r["season_type"]): r["n"] for r in
                  q("SELECT season, season_type, COUNT(*) n FROM games GROUP BY season, season_type")}
    bad = []
    for year, expected in EXPECTED_REG.items():
        got = per_season.get((str(year), "REG"), 0)
        if got != expected and year < date.today().year:
            bad.append(f"{year} REG {got}!={expected}")
    check("regular-season game counts match the schedule", not bad, "; ".join(bad[:6]))

    post_bad = []
    for year, expected in EXPECTED_POST.items():
        got = sum(n for (s, t), n in per_season.items() if s == str(year) and t not in ("REG",))
        if got != expected and year < date.today().year:
            post_bad.append(f"{year} post {got}!={expected}")
    check("playoff game counts match", not post_bad, "; ".join(post_bad[:6]))

    finals_wo_score = one("SELECT COUNT(*) FROM games WHERE season < ? AND home_score IS NULL",
                          str(date.today().year))
    check("every completed season's games have scores", finals_wo_score == 0, f"{finals_wo_score} missing")

    spread_cov = one("""SELECT ROUND(100.0 * COUNT(DISTINCT m.game_id) / COUNT(DISTINCT g.game_id), 1)
                        FROM games g LEFT JOIN market_lines m
                          ON m.game_id = g.game_id AND m.market_type = 'spread'
                        WHERE g.season < ?""", str(date.today().year))
    check("closing spread coverage 1999+ is complete", spread_cov >= 99.9, f"{spread_cov}%")

    ml_2010 = one("""SELECT ROUND(100.0 * COUNT(DISTINCT m.game_id) / COUNT(DISTINCT g.game_id), 1)
                     FROM games g LEFT JOIN market_lines m
                       ON m.game_id = g.game_id AND m.market_type = 'moneyline'
                     WHERE g.season >= '2010' AND g.season < ?""", str(date.today().year))
    check("moneyline coverage 2010+ is complete", ml_2010 >= 99.9, f"{ml_2010}%")
    ml_pre2006 = one("""SELECT COUNT(*) FROM market_lines m JOIN games g ON g.game_id = m.game_id
                        WHERE m.market_type = 'moneyline' AND g.season < '2006'""")
    check("no moneylines invented before 2006", ml_pre2006 == 0, f"{ml_pre2006} rows",
          warn_only=True)

    print("\n=== SEMANTIC ===")
    # THE important one. spread_line is the HOME side's expected margin, so a
    # positive spread means the home team was favoured. If the sign were
    # flipped, home favourites would lose most of the time and every cover,
    # every EV number and every model feature downstream would be backwards.
    row = one_row = conn.execute("""
        SELECT
          SUM(CASE WHEN m.line > 0 AND g.home_score > g.away_score THEN 1 ELSE 0 END) AS fav_home_won,
          SUM(CASE WHEN m.line > 0 THEN 1 ELSE 0 END)                                 AS fav_home,
          SUM(CASE WHEN m.line < 0 AND g.away_score > g.home_score THEN 1 ELSE 0 END)  AS fav_away_won,
          SUM(CASE WHEN m.line < 0 THEN 1 ELSE 0 END)                                 AS fav_away
        FROM games g JOIN market_lines m ON m.game_id = g.game_id AND m.market_type = 'spread'
        WHERE g.status = 'final'
    """).fetchone()
    fav_win_pct = 100.0 * (row["fav_home_won"] + row["fav_away_won"]) / max(1, row["fav_home"] + row["fav_away"])
    check("favourites win between 62% and 72% straight up (spread sign is right way round)",
          62.0 <= fav_win_pct <= 72.0, f"{fav_win_pct:.1f}% of {row['fav_home'] + row['fav_away']} games")

    # Home-field advantage must be positive but modest, and it must not be so
    # large that it implies the home/away columns are swapped.
    home_win = one("""SELECT ROUND(100.0 * AVG(CASE WHEN home_score > away_score THEN 1.0
                                                    WHEN home_score = away_score THEN 0.5 ELSE 0 END), 1)
                      FROM games WHERE status = 'final'""")
    check("home teams win between 52% and 60%", 52.0 <= home_win <= 60.0, f"{home_win}%")

    # Against the spread, both sides should sit near a coin flip. A market that
    # is not near 50% here means our sign, our scores or our lines are wrong.
    ats = conn.execute("""
        SELECT SUM(CASE WHEN (g.home_score - g.away_score) > m.line THEN 1 ELSE 0 END) AS home_cover,
               SUM(CASE WHEN (g.home_score - g.away_score) < m.line THEN 1 ELSE 0 END) AS away_cover,
               SUM(CASE WHEN (g.home_score - g.away_score) = m.line THEN 1 ELSE 0 END) AS push
        FROM games g JOIN market_lines m ON m.game_id = g.game_id AND m.market_type = 'spread'
        WHERE g.status = 'final'
    """).fetchone()
    decided = ats["home_cover"] + ats["away_cover"]
    ats_pct = 100.0 * ats["home_cover"] / max(1, decided)
    check("home teams cover between 47% and 53%", 47.0 <= ats_pct <= 53.0,
          f"{ats_pct:.1f}% ({ats['home_cover']}-{ats['away_cover']}-{ats['push']})")

    over = conn.execute("""
        SELECT SUM(CASE WHEN (g.home_score + g.away_score) > m.line THEN 1 ELSE 0 END) AS o,
               SUM(CASE WHEN (g.home_score + g.away_score) < m.line THEN 1 ELSE 0 END) AS u
        FROM games g JOIN market_lines m ON m.game_id = g.game_id AND m.market_type = 'total'
        WHERE g.status = 'final'
    """).fetchone()
    over_pct = 100.0 * over["o"] / max(1, over["o"] + over["u"])
    check("games go over between 47% and 53% of the time", 47.0 <= over_pct <= 53.0,
          f"{over_pct:.1f}% ({over['o']}-{over['u']})")

    # Key numbers: 3 and 7 must be the two most common margins in the NFL. If
    # they are not, our scores are not NFL scores.
    margins = q("""SELECT ABS(home_score - away_score) m, COUNT(*) n FROM games
                   WHERE status = 'final' GROUP BY m ORDER BY n DESC LIMIT 3""")
    top = [r["m"] for r in margins]
    check("3 and 7 are the most common margins of victory", 3 in top[:2] and 7 in top[:3],
          f"top margins {top}")

    # A moneyline favourite (negative price) should beat the underdog most of
    # the time; this independently confirms the home/away price mapping.
    ml = conn.execute("""
        SELECT SUM(CASE WHEN m.price_home < m.price_away AND g.home_score > g.away_score THEN 1
                        WHEN m.price_away < m.price_home AND g.away_score > g.home_score THEN 1
                        ELSE 0 END) AS fav_won,
               COUNT(*) AS n
        FROM games g JOIN market_lines m ON m.game_id = g.game_id AND m.market_type = 'moneyline'
        WHERE g.status = 'final' AND m.price_home IS NOT NULL AND m.price_away IS NOT NULL
    """).fetchone()
    ml_pct = 100.0 * ml["fav_won"] / max(1, ml["n"])
    check("moneyline favourites win between 62% and 72%", 62.0 <= ml_pct <= 72.0,
          f"{ml_pct:.1f}% of {ml['n']}")

    # The spread and the moneyline must agree about who was favoured.
    disagree = one("""
        SELECT COUNT(*) FROM games g
        JOIN market_lines s ON s.game_id = g.game_id AND s.market_type = 'spread'
        JOIN market_lines m ON m.game_id = g.game_id AND m.market_type = 'moneyline'
        WHERE s.line != 0 AND m.price_home IS NOT NULL AND m.price_away IS NOT NULL
          AND ((s.line > 0 AND m.price_home > m.price_away)
            OR (s.line < 0 AND m.price_away > m.price_home))""")
    total_both = one("""SELECT COUNT(*) FROM games g
        JOIN market_lines s ON s.game_id = g.game_id AND s.market_type = 'spread'
        JOIN market_lines m ON m.game_id = g.game_id AND m.market_type = 'moneyline'
        WHERE s.line != 0 AND m.price_home IS NOT NULL AND m.price_away IS NOT NULL""")
    pct_disagree = 100.0 * disagree / max(1, total_both)
    check("spread and moneyline agree on the favourite", pct_disagree < 2.0,
          f"{pct_disagree:.2f}% disagree ({disagree}/{total_both})")

    print("\n=== PLAY-BY-PLAY ===")
    have_pbp = one("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='nfl_plays'") > 0
    if not have_pbp:
        print("  SKIP  nfl_plays not ingested yet")
    else:
        plays = one("SELECT COUNT(*) FROM nfl_plays")
        pbp_games = one("SELECT COUNT(DISTINCT game_id) FROM nfl_plays")
        check("play-by-play present", plays > 0, f"{plays:,} plays across {pbp_games:,} games")

        orphan_pbp = one("SELECT COUNT(DISTINCT p.game_id) FROM nfl_plays p "
                         "WHERE NOT EXISTS (SELECT 1 FROM games g WHERE g.game_id = p.game_id)")
        check("every play's game is in the games table", orphan_pbp == 0, f"{orphan_pbp} orphan games")

        seasons_done = {r[0] for r in q("SELECT DISTINCT season FROM nfl_plays")}
        missing = [r["game_id"] for r in q(
            "SELECT g.game_id FROM games g LEFT JOIN nfl_plays p ON p.game_id = g.game_id "
            "WHERE g.status = 'final' AND p.game_id IS NULL GROUP BY g.game_id")
            if r["game_id"].split("_")[0] in seasons_done]
        unexpected = sorted(set(missing) - PBP_MISSING_UPSTREAM)
        check("every ingested season covers all its games, bar the known upstream gaps",
              not unexpected,
              f"{len(unexpected)} unexpected: {', '.join(unexpected[:5])}")
        stale = sorted(PBP_MISSING_UPSTREAM - set(missing))
        check("the pinned upstream gaps are still gaps", not stale,
              f"{', '.join(stale)} now has play-by-play; remove it from PBP_MISSING_UPSTREAM",
              warn_only=True)

        ev = one("SELECT COUNT(*) FROM events")
        check("events mirrors nfl_plays row for row", ev == plays, f"events={ev:,} plays={plays:,}")

        # SEMANTIC: expected points added must behave like expected points. A
        # touchdown is worth a lot, a turnover costs a lot. If these signs were
        # flipped or a column were misaligned, nothing structural would notice
        # and every model feature downstream would be quietly poisoned.
        epa_td = one("SELECT AVG(epa) FROM nfl_plays WHERE touchdown = 1 AND epa IS NOT NULL")
        epa_int = one("SELECT AVG(epa) FROM nfl_plays WHERE interception = 1 AND epa IS NOT NULL")
        epa_sack = one("SELECT AVG(epa) FROM nfl_plays WHERE sack = 1 AND epa IS NOT NULL")
        check("touchdowns carry large positive EPA", 1.0 <= (epa_td or 0) <= 4.0, f"{epa_td:.2f}")
        check("interceptions carry large negative EPA", -6.0 <= (epa_int or 0) <= -2.0, f"{epa_int:.2f}")
        check("sacks carry negative EPA", -3.0 <= (epa_sack or 0) <= -0.5, f"{epa_sack:.2f}")

        bad_wp = one("SELECT COUNT(*) FROM nfl_plays WHERE (wp IS NOT NULL AND (wp < 0 OR wp > 1)) "
                     "OR (vegas_wp IS NOT NULL AND (vegas_wp < 0 OR vegas_wp > 1))")
        check("win probabilities lie between 0 and 1", bad_wp == 0, f"{bad_wp} rows")

        # A play that swings win probability by a full 100 percentage points is
        # almost always an artifact. 27 of 1.28 M plays do (0.002%), clustered
        # in 1999-2001 and mostly extra points after a walk-off score, which is
        # tolerable; a jump in this rate means the column has come loose.
        extreme = one("SELECT COUNT(*) FROM nfl_plays WHERE ABS(wpa) >= 0.99")
        tot_wpa = one("SELECT COUNT(*) FROM nfl_plays WHERE wpa IS NOT NULL")
        ex_pct = 100.0 * extreme / max(1, tot_wpa)
        check("implausible full-swing win-probability plays stay under 0.05%",
              ex_pct < 0.05, f"{ex_pct:.4f}% ({extreme} of {tot_wpa:,})")
        mid_pinned = one("SELECT COUNT(*) FROM nfl_plays WHERE qtr <= 3 AND (wp = 0 OR wp = 1)")
        check("no win probability is pinned at 0 or 1 before the fourth quarter",
              mid_pinned == 0, f"{mid_pinned} rows")

        bad_down = one("SELECT COUNT(*) FROM nfl_plays WHERE down IS NOT NULL AND down NOT BETWEEN 1 AND 4")
        check("downs are 1 to 4", bad_down == 0, f"{bad_down} rows")
        bad_qtr = one("SELECT COUNT(*) FROM nfl_plays WHERE qtr IS NOT NULL AND qtr NOT BETWEEN 1 AND 6")
        check("quarters are 1 to 6 (four plus overtime)", bad_qtr == 0, f"{bad_qtr} rows")

        # Third-down conversion is one of the most stable rates in the sport;
        # a figure far from 38-42% means our down or conversion columns are
        # misaligned.
        third = conn.execute("SELECT SUM(third_down_converted) c, SUM(third_down_failed) f "
                             "FROM nfl_plays WHERE down = 3").fetchone()
        rate = 100.0 * (third["c"] or 0) / max(1, (third["c"] or 0) + (third["f"] or 0))
        check("third downs convert between 36% and 44%", 36.0 <= rate <= 44.0, f"{rate:.1f}%")

        pass_no_passer = one("SELECT COUNT(*) FROM nfl_plays WHERE play_type = 'pass' "
                             "AND passer_player_id IS NULL")
        pass_total = one("SELECT COUNT(*) FROM nfl_plays WHERE play_type = 'pass'")
        pct = 100.0 * pass_no_passer / max(1, pass_total)
        check("pass plays name their passer", pct < 3.0, f"{pct:.2f}% missing ({pass_no_passer:,})")

        # The running score on the last play of a game must equal the final
        # score in the games table: two independent files agreeing.
        bad = [r["game_id"] for r in q(
            "SELECT p.game_id, MAX(p.total_home_score) mh, MAX(p.total_away_score) ma "
            "FROM nfl_plays p JOIN games g ON g.game_id = p.game_id "
            "WHERE g.status = 'final' GROUP BY p.game_id "
            "HAVING mh != g.home_score OR ma != g.away_score")]
        new_bad = sorted(set(bad) - PBP_SCORE_MISMATCH_UPSTREAM)
        check("play-by-play scores reconcile with the final score, bar the known upstream defects",
              not new_bad, f"{len(new_bad)} new mismatch(es): {', '.join(new_bad[:5])}")
        fixed = sorted(PBP_SCORE_MISMATCH_UPSTREAM - set(bad))
        check("the pinned upstream mismatches are still wrong", not fixed,
              f"{len(fixed)} fixed upstream ({', '.join(fixed[:3])}); remove from PBP_SCORE_MISMATCH_UPSTREAM",
              warn_only=True)

    print("\n=== TIME ===")
    bad_utc = one("""SELECT COUNT(*) FROM games WHERE date_utc IS NOT NULL
                     AND ABS(JULIANDAY(SUBSTR(date_utc,1,10)) - JULIANDAY(local_date)) > 1""")
    check("UTC timestamps are within a day of the local date", bad_utc == 0, f"{bad_utc} rows")
    no_utc = one("SELECT COUNT(*) FROM games WHERE date_utc IS NULL")
    check("games missing a UTC kickoff are only the 1999 season", no_utc <= 260,
          f"{no_utc} rows (1999 has no gametime in the source)", warn_only=True)

    print("\n=== PROVENANCE ===")
    check("every game row names its source",
          one("SELECT COUNT(*) FROM games WHERE source IS NULL OR fetched_at IS NULL") == 0)
    check("every market line names its source",
          one("SELECT COUNT(*) FROM market_lines WHERE source IS NULL") == 0)
    check("no market line claims a capture time we do not have",
          one("SELECT COUNT(*) FROM market_lines WHERE captured_at IS NOT NULL") == 0,
          "historical lines are book-unspecified and time-unknown by design")
    runs = one("SELECT COUNT(*) FROM ingest_runs")
    check("ingest runs are recorded", runs > 0, f"{runs} runs")

    print("\n" + "=" * 60)
    if FAILS:
        print(f"VALIDATION FAILED: {len(FAILS)} check(s)")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print(f"VALIDATION PASSED ({len(WARNS)} warning(s))")
    for w in WARNS:
        print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
