"""Repair the starter flag, and the games-started (GS) totals built from it.

Found 2026-09-27 (nav audit, bug 17). player_game_log.starter was set to
"the player has a position in nba.com's box score" (nba_pipeline.py). From
2017-18 on nba.com fills `position` only for the five starters, so that was
right. Before 2017-18 the same feed gives EVERY player who played his roster
position, so every appearance was flagged a start and every GS column built
from it (player_season_totals.gs, player_splits.gs) was a copy of GP: 2009-10
summed to 22,243 league starts where a 30-team, 1,230-game season has exactly
12,300, and James Harden's rookie year read 76 GS (he started none).

The true starters are still in the same box score: nba.com lists each team's
five starters first. Checked before trusting it:
  - 2017-18 onward (position = starter), "first five listed" and the position
    flag agree on every one of 315,815 player-games;
  - before 2017-18, the first five of every one of 52,958 team-games all
    played, and the season GS they give agrees with nba.com's own career rows
    (player_career_official) on 10,413 of 10,475 player-team-seasons; 55 more
    are within one start. League totals come out at exactly 12,300 in 2009-10.
So for games before 2017-18 the starter flag becomes "among the first five
listed for his team". A team-game whose first five do not all show minutes
(none today) is left UNKNOWN (NULL), never guessed, and the GS totals that
include it become NULL too.

Everything is read from box_scores.traditional_json already in the database:
no network. Dry run by default. --apply first copies every row it will change
(player_game_log starter, player_season_totals gs, player_splits gs) into
Data/backups/starters_before_<timestamp>.sqlite, then writes in one
transaction and checks the league totals afterwards.

    venv/Scripts/python.exe repair_starters.py [--apply]
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, Optional, Set

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "Data", "TeamData.sqlite")
POSITION_FLAG_FIRST_SEASON = "2017-18"   # from here on the stored flag is already right
_ZERO_MINUTES = {"", "0", "0:00", "00:00", "PT00M00.00S"}


def first_five(traditional_json: Optional[str]) -> Dict[int, Optional[Set[int]]]:
    """{team_id: the five players listed first, or None when that can't be trusted}."""
    if not traditional_json:
        return {}
    box = json.loads(traditional_json).get("boxScoreTraditional", {})
    out: Dict[int, Optional[Set[int]]] = {}
    for side in ("homeTeam", "awayTeam"):
        team = box.get(side) or {}
        if team.get("teamId") is None:
            continue
        players = team.get("players") or []
        five = players[:5]
        played = [str((p.get("statistics") or {}).get("minutes") or "").strip() not in _ZERO_MINUTES for p in five]
        out[int(team["teamId"])] = (
            {int(p["personId"]) for p in five} if len(five) == 5 and all(played) else None
        )
    return out


def _gs(flags: Iterable[Optional[int]]) -> Optional[int]:
    """Games started from per-game flags; unknown if any game is unknown."""
    total = 0
    for f in flags:
        if f is None:
            return None
        total += f
    return total


def plan(conn: sqlite3.Connection) -> dict:
    """Everything the repair would write, computed read-only."""
    new_flag: Dict[int, Optional[int]] = {}   # player_game_log.id -> starter
    unknown_team_games = 0
    team_games = 0
    games = conn.execute(
        "SELECT game_id, traditional_json FROM box_scores WHERE season < ?",
        (POSITION_FLAG_FIRST_SEASON,),
    )
    starters: Dict[str, Dict[int, Optional[Set[int]]]] = {}
    for gid, tj in games:
        starters[gid] = first_five(tj)
    for log_id, gid, pid, tid, old in conn.execute(
        """SELECT g.id, g.game_id, g.player_id, g.team_id, g.starter
           FROM player_game_log g JOIN box_scores b ON b.game_id = g.game_id
           WHERE b.season < ?""",
        (POSITION_FLAG_FIRST_SEASON,),
    ):
        five = starters.get(gid, {}).get(tid, "missing")
        if five == "missing":
            continue   # no box score for this team-game: leave the row alone
        flag = None if five is None else int(pid in five)
        if flag != old:
            new_flag[log_id] = flag
    for gid, teams in starters.items():
        for five in teams.values():
            team_games += 1
            unknown_team_games += five is None

    # Re-derive GS exactly as the backfill groups GP (player_game_log joined
    # to team_game_advanced), so a total is only replaced where its GP agrees.
    per_total = defaultdict(list)
    per_split = defaultdict(list)
    rows = conn.execute(
        """SELECT g.id, g.player_id, g.team_id, g.starter, t.season, t.season_type,
                  CASE WHEN g.team_id = b.home_team_id THEN 'Home' ELSE 'Road' END,
                  CASE WHEN t.pts > t.opp_pts THEN 'Wins' ELSE 'Losses' END,
                  strftime('%m', g.game_date)
           FROM player_game_log g
           JOIN team_game_advanced t ON t.game_id = g.game_id AND t.team_id = g.team_id
           JOIN box_scores b ON b.game_id = g.game_id
           WHERE t.season < ?""",
        (POSITION_FLAG_FIRST_SEASON,),
    )
    months = {f"{i:02d}": m for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"], start=1)}
    for log_id, pid, tid, old, season, stype, loc, wl, mm in rows:
        flag = new_flag.get(log_id, old)
        per_total[(pid, season, stype, tid)].append(flag)
        for split_type, value in (("Location", loc), ("Wins/Losses", wl), ("Month", months.get(mm, "Unknown"))):
            per_split[(pid, season, stype, split_type, value)].append(flag)

    total_updates, total_skipped = [], 0
    for rid, pid, season, stype, tid, gp, gs in conn.execute(
        "SELECT id, player_id, season, season_type, team_id, gp, gs FROM player_season_totals WHERE season < ?",
        (POSITION_FLAG_FIRST_SEASON,),
    ):
        flags = per_total.get((pid, season, stype, tid))
        if flags is None or len(flags) != gp:
            total_skipped += 1
            continue
        new = _gs(flags)
        if new != gs:
            total_updates.append((rid, gs, new))

    split_updates, split_skipped = [], 0
    for rid, pid, season, stype, split_type, value, gp, gs in conn.execute(
        """SELECT id, player_id, season, season_type, split_type, split_value, gp, gs
           FROM player_splits WHERE season < ?""",
        (POSITION_FLAG_FIRST_SEASON,),
    ):
        flags = per_split.get((pid, season, stype, split_type, value))
        if flags is None or len(flags) != gp:
            split_skipped += 1
            continue
        new = _gs(flags)
        if new != gs:
            split_updates.append((rid, gs, new))

    return {
        "flag_updates": new_flag, "team_games": team_games, "unknown_team_games": unknown_team_games,
        "total_updates": total_updates, "total_skipped": total_skipped,
        "split_updates": split_updates, "split_skipped": split_skipped,
    }


def league_starts(conn: sqlite3.Connection, season: str) -> tuple:
    """(summed GS, team-games x 5) for a regular season."""
    gs = conn.execute(
        "SELECT SUM(gs) FROM player_season_totals WHERE season = ? AND season_type = 'Regular Season'",
        (season,),
    ).fetchone()[0]
    games = conn.execute(
        "SELECT COUNT(*) FROM box_scores WHERE season = ? AND season_type = 'Regular Season'", (season,)
    ).fetchone()[0]
    return gs, games * 2 * 5


def _backup(conn: sqlite3.Connection, p: dict, path: str) -> None:
    b = sqlite3.connect(path)
    try:
        b.execute("CREATE TABLE player_game_log_starter (id INTEGER PRIMARY KEY, starter INTEGER)")
        b.execute("CREATE TABLE player_season_totals_gs (id INTEGER PRIMARY KEY, gs INTEGER)")
        b.execute("CREATE TABLE player_splits_gs (id INTEGER PRIMARY KEY, gs INTEGER)")
        ids = list(p["flag_updates"])
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            rows = conn.execute(
                f"SELECT id, starter FROM player_game_log WHERE id IN ({','.join('?' * len(chunk))})", chunk
            ).fetchall()
            b.executemany("INSERT INTO player_game_log_starter VALUES (?, ?)", rows)
        b.executemany("INSERT INTO player_season_totals_gs VALUES (?, ?)", [(r, old) for r, old, _ in p["total_updates"]])
        b.executemany("INSERT INTO player_splits_gs VALUES (?, ?)", [(r, old) for r, old, _ in p["split_updates"]])
        b.commit()
    finally:
        b.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write the repair (default: dry run)")
    ap.add_argument("--db", default=DB)
    args = ap.parse_args(argv)

    conn = sqlite3.connect(args.db)
    try:
        before = {s: league_starts(conn, s) for s in ("1996-97", "2009-10", "2016-17", "2017-18")}
        p = plan(conn)
        print(f"team-games before {POSITION_FLAG_FIRST_SEASON}: {p['team_games']:,} "
              f"({p['unknown_team_games']} with an untrusted first five, left NULL)")
        print(f"player_game_log.starter rows to change: {len(p['flag_updates']):,}")
        print(f"player_season_totals.gs rows to change: {len(p['total_updates']):,} "
              f"(skipped, GP disagrees with the log: {p['total_skipped']})")
        print(f"player_splits.gs rows to change: {len(p['split_updates']):,} (skipped: {p['split_skipped']})")
        for s, (gs, exact) in before.items():
            print(f"  {s}: league GS {gs:,} vs 5 x team-games {exact:,}")
        if not args.apply:
            print("Dry run. Pass --apply to write.")
            return 0

        os.makedirs(os.path.join(HERE, "Data", "backups"), exist_ok=True)
        path = os.path.join(HERE, "Data", "backups", f"starters_before_{datetime.now():%Y%m%d_%H%M%S}.sqlite")
        _backup(conn, p, path)
        print(f"Backed up the old values to {path}")
        with conn:
            conn.executemany("UPDATE player_game_log SET starter = ? WHERE id = ?",
                             [(flag, i) for i, flag in p["flag_updates"].items()])
            conn.executemany("UPDATE player_season_totals SET gs = ? WHERE id = ?",
                             [(new, rid) for rid, _, new in p["total_updates"]])
            conn.executemany("UPDATE player_splits SET gs = ? WHERE id = ?",
                             [(new, rid) for rid, _, new in p["split_updates"]])
        bad = []
        for s in before:
            gs, exact = league_starts(conn, s)
            print(f"  after  {s}: league GS {gs:,} vs 5 x team-games {exact:,}")
            if gs != exact:
                bad.append(s)
        if bad:
            print(f"WARNING: league starts differ from 5 per team-game in {bad}")
            return 1
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
