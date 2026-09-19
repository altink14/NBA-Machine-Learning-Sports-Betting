"""
teams.py (NFL)
==============
The NFL team registry: our canonical abbreviation, the full name, conference
and division, and the alternate names outside sources use.

WHY THIS IS HAND-WRITTEN RATHER THAN FETCHED. Team names are the join of last
resort, and joining on a raw name is how archives quietly corrupt themselves:
"LA Chargers" and "Los Angeles Chargers" and "Chargers" all mean LAC, while
"LA Rams" does not. The blueprint's rule is never to join on names, so this
file is the single place a name becomes an id, it is auditable in one screen,
and anything it does not recognise is reported rather than guessed.

RELOCATIONS keep their historical codes, because a 2015 St. Louis Rams game
was played by the St. Louis Rams. `games` therefore carries STL, SD and OAK
rows, and `CURRENT_OF` maps each to the franchise's present code for anyone
who wants continuity instead of history. Displaying relocated franchises under
modern names is documented debt on the NBA side; here both are available.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: abbrev -> (full name, conference, division)
NFL_TEAMS: Dict[str, Tuple[str, str, str]] = {
    "ARI": ("Arizona Cardinals", "NFC", "West"),
    "ATL": ("Atlanta Falcons", "NFC", "South"),
    "BAL": ("Baltimore Ravens", "AFC", "North"),
    "BUF": ("Buffalo Bills", "AFC", "East"),
    "CAR": ("Carolina Panthers", "NFC", "South"),
    "CHI": ("Chicago Bears", "NFC", "North"),
    "CIN": ("Cincinnati Bengals", "AFC", "North"),
    "CLE": ("Cleveland Browns", "AFC", "North"),
    "DAL": ("Dallas Cowboys", "NFC", "East"),
    "DEN": ("Denver Broncos", "AFC", "West"),
    "DET": ("Detroit Lions", "NFC", "North"),
    "GB":  ("Green Bay Packers", "NFC", "North"),
    "HOU": ("Houston Texans", "AFC", "South"),
    "IND": ("Indianapolis Colts", "AFC", "South"),
    "JAX": ("Jacksonville Jaguars", "AFC", "South"),
    "KC":  ("Kansas City Chiefs", "AFC", "West"),
    "LA":  ("Los Angeles Rams", "NFC", "West"),
    "LAC": ("Los Angeles Chargers", "AFC", "West"),
    "LV":  ("Las Vegas Raiders", "AFC", "West"),
    "MIA": ("Miami Dolphins", "AFC", "East"),
    "MIN": ("Minnesota Vikings", "NFC", "North"),
    "NE":  ("New England Patriots", "AFC", "East"),
    "NO":  ("New Orleans Saints", "NFC", "South"),
    "NYG": ("New York Giants", "NFC", "East"),
    "NYJ": ("New York Jets", "AFC", "East"),
    "PHI": ("Philadelphia Eagles", "NFC", "East"),
    "PIT": ("Pittsburgh Steelers", "AFC", "North"),
    "SEA": ("Seattle Seahawks", "NFC", "West"),
    "SF":  ("San Francisco 49ers", "NFC", "West"),
    "TB":  ("Tampa Bay Buccaneers", "NFC", "South"),
    "TEN": ("Tennessee Titans", "AFC", "South"),
    "WAS": ("Washington Commanders", "NFC", "East"),
    # Historical codes that appear in games before a relocation or rename.
    "OAK": ("Oakland Raiders", "AFC", "West"),
    "SD":  ("San Diego Chargers", "AFC", "West"),
    "STL": ("St. Louis Rams", "NFC", "West"),
}

#: Historical code -> the franchise's current code, for continuity views.
CURRENT_OF: Dict[str, str] = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

#: Names other sources use, mapped to our abbreviation. The Odds API and ESPN
#: both say "Los Angeles Rams"; older files and casual sources vary. Anything
#: absent here is reported as unmatched rather than guessed at.
_ALIASES: Dict[str, str] = {
    "LA Rams": "LA", "St. Louis Rams": "STL", "St Louis Rams": "STL",
    "LA Chargers": "LAC", "San Diego Chargers": "SD",
    "Las Vegas Raiders": "LV", "Oakland Raiders": "OAK",
    "Washington Football Team": "WAS", "Washington Redskins": "WAS",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Green Bay Packers": "GB", "New Orleans Saints": "NO",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "New England Patriots": "NE",
}

#: full name (and alias) -> abbrev. Built once; the aliases win over the
#: generated entries where they overlap, which is what we want for LA/STL.
NFL_TEAM_BY_NAME: Dict[str, str] = {name: ab for ab, (name, _, _) in NFL_TEAMS.items()}
NFL_TEAM_BY_NAME.update(_ALIASES)


def conference_division(abbrev: str) -> Tuple[str, str]:
    t = NFL_TEAMS.get(abbrev)
    return (t[1], t[2]) if t else ("", "")
