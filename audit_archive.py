"""Cross-table integrity audit of the stats archive. Read-only.

Written 2026-09-23, the day three bugs turned up that no existing check could
see, because each table was fine on its own and wrong only against another
one: leaders ranked from totals and re-ranked per game, a traded player's
stints read as his season, and 1,988 advanced rows holding 0.0 placeholders.
The existing validator (src/Utils/nba_validation.py) compares one game's
ratings with nba.com; this compares the archive with itself.

Every check is a statement that must hold, with a count of rows that break it
and a few examples. A check can carry an explained, permanent exception (the
known empty box scores) rather than being loosened.

    venv/Scripts/python.exe audit_archive.py            # all seasons
    venv/Scripts/python.exe audit_archive.py --season 2024-25

Exit code 1 when any check fails, so it can gate a job.
"""

import argparse
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "Data", "TeamData.sqlite")

# nba.com's own boxscoreadvancedv3 returns nothing for these on every retry
# (CLAUDE.md, "four permanent holes"), so they have results but no players.
KNOWN_EMPTY_GAMES = ("0029600332", "0029600370", "0029800661", "0020300778")
# nba.com's own box score lists players whose points total 97 for a team it
# scores at 101 (BOS, 1999-00); the line score and results agree on 101.
KNOWN_SHORT_BOX_SCORES = ("0029900712",)
# Did he play: minutes, or any recorded stat (the ingest's own rule since
# 2026-09-23, when 0:00 appearances with free throws or fouls were restored).
PLAYED = ("(pg.min > 0 OR pg.pts <> 0 OR pg.reb <> 0 OR pg.ast <> 0 OR pg.stl <> 0 OR pg.blk <> 0 "
          "OR pg.tov <> 0 OR pg.pf <> 0 OR pg.fga <> 0 OR pg.fta <> 0)")


def checks(season_filter):
    s_bs = "AND b.season = :season" if season_filter else ""
    s_t = "AND t.season = :season" if season_filter else ""
    holes = ",".join(f"'{g}'" for g in KNOWN_EMPTY_GAMES)
    short = ",".join(f"'{g}'" for g in KNOWN_EMPTY_GAMES + KNOWN_SHORT_BOX_SCORES)
    return [
        ("every archived game has exactly two team rows",
         f"""SELECT b.game_id, b.season, COUNT(t.team_id) AS teams
             FROM box_scores b LEFT JOIN team_game_advanced t ON t.game_id = b.game_id
             WHERE 1=1 {s_bs} GROUP BY b.game_id HAVING COUNT(t.team_id) <> 2"""),

        ("the two team rows are the game's home and away teams",
         f"""SELECT b.game_id, b.season, b.home_team_id, b.away_team_id
             FROM box_scores b
             WHERE 1=1 {s_bs} AND (
               NOT EXISTS (SELECT 1 FROM team_game_advanced t WHERE t.game_id = b.game_id AND t.team_id = b.home_team_id)
               OR NOT EXISTS (SELECT 1 FROM team_game_advanced t WHERE t.game_id = b.game_id AND t.team_id = b.away_team_id))"""),

        ("a team's points equal the sum of its players' points",
         f"""SELECT t.game_id, t.season, t.team_id, t.pts AS team_pts, SUM(g.pts) AS player_pts
             FROM team_game_advanced t
             JOIN player_game_log g ON g.game_id = t.game_id AND g.team_id = t.team_id
             WHERE t.game_id NOT IN ({short}) {s_t}
             GROUP BY t.game_id, t.team_id HAVING t.pts <> SUM(g.pts)"""),

        ("every archived team-game has player rows (bar the four known empty games)",
         f"""SELECT t.game_id, t.season, t.team_id FROM team_game_advanced t
             WHERE t.game_id NOT IN ({holes}) {s_t}
               AND NOT EXISTS (SELECT 1 FROM player_game_log g WHERE g.game_id = t.game_id AND g.team_id = t.team_id)"""),

        ("opponent points match the other team's points",
         f"""SELECT t.game_id, t.season, t.team_id, t.opp_pts, o.pts AS other_team_pts
             FROM team_game_advanced t
             JOIN team_game_advanced o ON o.game_id = t.game_id AND o.team_id = t.opp_team_id
             WHERE t.opp_pts <> o.pts {s_t}"""),

        ("the line score adds up to the final score",
         f"""SELECT l.game_id, t.season, l.team_id, l.pts AS line_pts, t.pts AS team_pts
             FROM game_line_scores l
             JOIN team_game_advanced t ON t.game_id = l.game_id AND t.team_id = l.team_id
             WHERE l.pts <> t.pts {s_t}"""),

        ("the quarters add up to the line score's total",
         f"""SELECT l.game_id, t.season, l.team_id, l.pts,
                    COALESCE(l.q1,0)+COALESCE(l.q2,0)+COALESCE(l.q3,0)+COALESCE(l.q4,0)
                    +COALESCE(l.ot1,0)+COALESCE(l.ot2,0)+COALESCE(l.ot3,0)+COALESCE(l.ot4,0)+COALESCE(l.ot5,0)
                    +COALESCE(l.ot6,0)+COALESCE(l.ot7,0)+COALESCE(l.ot8,0)+COALESCE(l.ot9,0)+COALESCE(l.ot10,0) AS quarters
             FROM game_line_scores l
             JOIN team_game_advanced t ON t.game_id = l.game_id AND t.team_id = l.team_id
             WHERE l.q1 IS NOT NULL {s_t} AND l.pts <>
                    COALESCE(l.q1,0)+COALESCE(l.q2,0)+COALESCE(l.q3,0)+COALESCE(l.q4,0)
                    +COALESCE(l.ot1,0)+COALESCE(l.ot2,0)+COALESCE(l.ot3,0)+COALESCE(l.ot4,0)+COALESCE(l.ot5,0)
                    +COALESCE(l.ot6,0)+COALESCE(l.ot7,0)+COALESCE(l.ot8,0)+COALESCE(l.ot9,0)+COALESCE(l.ot10,0)"""),

        ("the results history agrees with the box scores on the score",
         f"""SELECT r.game_id, r.season, r.team_id, r.pts AS results_pts, t.pts AS box_pts
             FROM game_results r
             JOIN team_game_advanced t ON t.game_id = r.game_id AND t.team_id = r.team_id
             WHERE r.pts <> t.pts {s_t}"""),

        ("season totals equal the game log they are built from (games, points)",
         f"""SELECT t.player_id, t.season, t.season_type, t.team_id, t.gp, t.pts, g.gp AS log_gp, g.pts AS log_pts
             FROM player_season_totals t
             JOIN (SELECT pg.player_id, pg.team_id, b.season, b.season_type,
                          COUNT(DISTINCT pg.game_id) AS gp, SUM(pg.pts) AS pts
                   FROM player_game_log pg JOIN box_scores b ON b.game_id = pg.game_id
                   WHERE {PLAYED}
                   GROUP BY pg.player_id, pg.team_id, b.season, b.season_type) g
               ON g.player_id = t.player_id AND g.team_id = t.team_id
              AND g.season = t.season AND g.season_type = t.season_type
             WHERE (t.pts <> g.pts OR t.gp <> g.gp) {s_t}"""),

        ("league wins equal league losses",
         f"""SELECT t.season, t.season_type, SUM(t.wins) AS wins, SUM(t.losses) AS losses
             FROM team_season_advanced t WHERE 1=1 {s_t}
             GROUP BY t.season, t.season_type HAVING SUM(t.wins) <> SUM(t.losses)"""),

        ("a team's wins and losses add up to its games",
         f"""SELECT t.team_id, t.season, t.season_type, t.games, t.wins, t.losses
             FROM team_season_advanced t WHERE t.wins + t.losses <> t.games {s_t}"""),

        ("a team's season games match its game rows",
         f"""SELECT t.team_id, t.season, t.season_type, t.games, COUNT(g.game_id) AS game_rows
             FROM team_season_advanced t
             LEFT JOIN team_game_advanced g ON g.team_id = t.team_id AND g.season = t.season AND g.season_type = t.season_type
             WHERE 1=1 {s_t}
             GROUP BY t.team_id, t.season, t.season_type HAVING t.games <> COUNT(g.game_id)"""),

        ("every season-totals row has its advanced row, and no orphans",
         f"""SELECT t.player_id, t.season, t.season_type, t.team_id, 'no advanced row' AS problem
             FROM player_season_totals t
             WHERE NOT EXISTS (SELECT 1 FROM player_season_advanced a WHERE a.player_id = t.player_id
                   AND a.season = t.season AND a.season_type = t.season_type AND a.team_id = t.team_id) {s_t}
             UNION ALL
             SELECT t.player_id, t.season, t.season_type, t.team_id, 'advanced row without totals'
             FROM player_season_advanced t
             WHERE NOT EXISTS (SELECT 1 FROM player_season_totals a WHERE a.player_id = t.player_id
                   AND a.season = t.season AND a.season_type = t.season_type AND a.team_id = t.team_id) {s_t}"""),

        ("no advanced row holds the old 0.0 placeholders",
         f"""SELECT t.player_id, t.season, t.season_type, t.team_id FROM player_season_advanced t
             WHERE t.pace = 0 AND t.off_rating = 0 AND t.def_rating = 0 {s_t}"""),

        ("a traded player's stints carry no season-only figures",
         f"""SELECT t.player_id, t.season, t.season_type, t.team_id, t.usg_pct FROM player_season_advanced t
             WHERE t.usg_pct IS NOT NULL {s_t}
               AND (SELECT COUNT(*) FROM player_season_totals x WHERE x.player_id = t.player_id
                    AND x.season = t.season AND x.season_type = t.season_type) > 1"""),

        ("every player in the game log exists in players",
         f"""SELECT DISTINCT g.player_id FROM player_game_log g
             JOIN box_scores b ON b.game_id = g.game_id
             WHERE NOT EXISTS (SELECT 1 FROM players p WHERE p.player_id = g.player_id) {s_bs}"""),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", help="limit to one season, e.g. 2024-25")
    ap.add_argument("--db", default=DB)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args(argv)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    failed = 0
    for name, sql in checks(args.season):
        try:
            rows = conn.execute(sql, {"season": args.season} if args.season else {}).fetchall()
        except sqlite3.Error as e:
            print(f"  ERROR  {name}: {e}")
            failed += 1
            continue
        if rows:
            failed += 1
            print(f"   FAIL  {name}: {len(rows):,}")
            for r in rows[: args.examples]:
                print(f"           {dict(r)}")
        else:
            print(f"   PASS  {name}")
    print(f"\n{failed} of {len(checks(args.season))} checks failed" if failed else "\nall checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
