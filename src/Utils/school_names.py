"""Common names for the schools nba.com's draft history spells formally.

nba.com's `drafthistory` records the university's official name, so the
second-most prolific program in the draft reads "California-Los Angeles" and
readers look for it as UCLA. This table maps the official spellings to the
names basketball fans use. Rules for adding a row:

- Only a rename, never a merge of two different schools. Every key here is
  one school; the value is that same school's common name.
- The name the school goes by in college basketball today, unless it changed
  identity (Texas-Pan American became UTRGV by merger, so it is left as
  recorded; the page shows where a player was drafted from).
- Parenthesised state tags ("Miami (OH)") are how the sport itself tells two
  schools apart, so they stay, except where the tagless name is unambiguous
  in college basketball ("Miami (FL)" is Miami; "St. John's (NY)" is
  St. John's).
- Anything not listed passes through unchanged, so a new spelling shows as
  nba.com wrote it rather than disappearing.

`organization_official` keeps nba.com's spelling next to the common one on
every row that carries it.
"""
from typing import Optional

COMMON_SCHOOL_NAMES = {
    "California-Los Angeles": "UCLA",
    "Southern California": "USC",
    "Louisiana State": "LSU",
    "Nevada-Las Vegas": "UNLV",
    "Brigham Young": "BYU",
    "Connecticut": "UConn",
    "North Carolina State": "NC State",
    "Texas-El Paso": "UTEP",
    "Southern Methodist": "SMU",
    "Texas Christian": "TCU",
    "Virginia Commonwealth": "VCU",
    "Alabama-Birmingham": "UAB",
    "Central Florida": "UCF",
    "Miami (FL)": "Miami",
    "St. John's (NY)": "St. John's",
    "Mississippi": "Ole Miss",
    "Southern Mississippi": "Southern Miss",
    "Pennsylvania": "Penn",
    "Massachusetts": "UMass",
    "Nevada-Reno": "Nevada",
    "North Carolina-Charlotte": "Charlotte",
    "North Carolina-Wilmington": "UNC Wilmington",
    "Loyola-Chicago": "Loyola Chicago",
    "Loyola-Marymount": "Loyola Marymount",
    "Loyola-Maryland": "Loyola Maryland",
    "Long Island-Brooklyn": "LIU",
    "Louisiana-Lafayette": "Louisiana",
    "California-Santa Barbara": "UC Santa Barbara",
    "California-Irvine": "UC Irvine",
    "California-Riverside": "UC Riverside",
    "California-Davis": "UC Davis",
    "California-San Diego": "UC San Diego",
    "Cal State-Fullerton": "Cal State Fullerton",
    "Cal State-Los Angeles": "Cal State LA",
    "Cal State-Northridge": "Cal State Northridge",
    "Cal State-Bakersfield": "Cal State Bakersfield",
    "Tennessee-Chattanooga": "Chattanooga",
    "Tennessee-Martin": "UT Martin",
    "Wisconsin-Green Bay": "Green Bay",
    # Same school as the older rows spelled plain "Milwaukee" (UW-Milwaukee).
    "Wisconsin-Milwaukee": "Milwaukee",
    "Texas-Arlington": "UT Arlington",
    "Texas-San Antonio": "UTSA",
    "Arkansas-Little Rock": "Little Rock",
    "Illinois-Chicago": "UIC",
    "Missouri-Kansas City": "Kansas City",
    "Maryland-Baltimore County": "UMBC",
    "Maryland-Eastern Shore": "Maryland Eastern Shore",
    "Nebraska-Omaha": "Omaha",
}


def common_school_name(official: Optional[str]) -> Optional[str]:
    """The common name for nba.com's spelling; blank stays None, unknown passes through."""
    if official is None:
        return None
    name = official.strip()
    if not name:
        return None
    return COMMON_SCHOOL_NAMES.get(name, name)
