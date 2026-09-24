"""Restore player appearances recorded as "0:00" minutes with real stats.

The ingest skipped every player whose minutes were 0:00, but nba.com records
a player who came in to shoot a technical free throw, or checked in for a
dead-ball second, that way, with real points, fouls or rebounds. 14 such
appearances were missing across 30 seasons (4 with points, so their team's
players summed short of its score). Found 2026-09-23 by audit_archive.py; the
filter is fixed in src/Utils/nba_pipeline.py (save_players_and_game_log).

This re-reads the box scores we already store (no nba.com calls), re-saves
the affected teams' player rows through the fixed ingest, then rebuilds those
seasons' player aggregates from the game log. Dry run by default.
"""

import argparse
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src", "Process-Data"))
DB = os.path.join(HERE, "Data", "TeamData.sqlite")
COUNTING = ("points", "reboundsTotal", "assists", "steals", "blocks", "turnovers",
            "foulsPersonal", "fieldGoalsAttempted", "freeThrowsAttempted")


def _minutes(v) -> float:
    from src.Utils.nba_pipeline import parse_minutes
    s = str(v or "").strip()
    return parse_minutes(s) if s else 0.0


def find(conn):
    """[(game_id, season, season_type, game_date, team_id, players)] for teams with a missing appearance."""
    out = []
    logged = {(g, p) for g, p in conn.execute("SELECT game_id, player_id FROM player_game_log")}
    for gid, season, stype, gdate, raw in conn.execute(
            "SELECT game_id, season, season_type, game_date, traditional_json FROM box_scores WHERE traditional_json IS NOT NULL"):
        box = (json.loads(raw) or {}).get("boxScoreTraditional") or {}
        for side in ("homeTeam", "awayTeam"):
            team = box.get(side) or {}
            players = team.get("players") or []
            missing = [p for p in players
                       if (gid, int(p["personId"])) not in logged
                       and _minutes((p.get("statistics") or {}).get("minutes")) <= 0
                       and any(int((p.get("statistics") or {}).get(k) or 0) for k in COUNTING)]
            if missing:
                out.append((gid, season, stype, gdate, int(team["teamId"]), players, missing))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--db", default=DB)
    args = ap.parse_args(argv)
    conn = sqlite3.connect(args.db, timeout=30)
    todo = find(conn)
    for gid, season, stype, _, team_id, _, missing in todo:
        for p in missing:
            st = p.get("statistics") or {}
            print(f"  {gid} {season} {stype:14s} team {team_id} {p.get('firstName', '')} {p.get('familyName', '')}: "
                  + ", ".join(f"{k}={st.get(k)}" for k in COUNTING if st.get(k)))
    seasons = sorted({(t[1], t[2]) for t in todo})
    print(f"{sum(len(t[6]) for t in todo)} appearance(s) in {len(todo)} team-game(s), {len(seasons)} season(s) to rebuild")
    if not args.apply or not todo:
        print("dry run: nothing written." if not args.apply else "nothing to do.")
        return 0
    from src.Utils.nba_pipeline import save_players_and_game_log
    with conn:
        for gid, _, _, gdate, team_id, players, _ in todo:
            save_players_and_game_log(conn, gid, gdate, players, team_id)
    conn.close()
    from backfill import compute_and_save_player_season_aggregates
    for season, stype in seasons:
        compute_and_save_player_season_aggregates(season, stype, args.db)
        print(f"rebuilt {season} {stype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
