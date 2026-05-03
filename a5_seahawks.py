"""
A5: Pandas analysis of Seattle Seahawks play-by-play data
Author: Jasmine Sayed
HCDE 530, Spring 2026

Dataset: NFL play-by-play data from the nflverse project, accessed via the
nflreadpy Python package (https://github.com/nflverse/nflreadpy).
Scope: 2016–2025 regular and postseason. Filtered to plays where Seattle
was the offense (posteam == 'SEA').

The script answers three analytical questions from MP1a:
    1. Across DK Metcalf's six seasons with the Seahawks (2019–2024), how do
       his targets per game, yards per target, and EPA per target vary year
       over year, and which season was his most efficient?
    2. What share of the Seahawks' red zone trips ended in a touchdown
       across the past ten seasons, and how has that touchdown conversion
       rate trended over time?
    3. How does the Seahawks' offensive identity differ between the Pete
       Carroll era (2016–2023) and the Mike Macdonald era (2024–2025),
       measured by early-down pass rate and average depth of target per
       play?

It uses three of the five pandas operations from class:
    df.head() and df.info(),
    df[df['column'] > value]  (boolean filtering),
    df.groupby('column')['other'].mean().
"""

import nflreadpy as nfl
import pandas as pd

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
# Pull play-by-play for the past ten seasons. nflreadpy returns a Polars
# DataFrame, so I convert to pandas right away because the assignment
# (and the rest of class) is built around pandas.
print("Loading nflverse play-by-play data for 2016–2025...")
pbp = nfl.load_pbp(seasons=list(range(2016, 2026))).to_pandas()

# Keep only plays where Seattle was on offense.
sea = pbp[pbp['posteam'] == 'SEA'].copy()
print(f"Loaded {len(sea):,} Seattle offensive plays across "
      f"{sea['season'].nunique()} seasons.\n")


# ======================================================================
# Initial exploration
# ======================================================================

# OPERATION 1: df.head() and df.info()
# Question: What does this dataset look like before I start asking real
# questions of it? How many rows and columns am I working with, and what
# are the most relevant column types?
# What the answer means: head() shows me a few real plays so I can sanity
# check that I have the right team and the columns I expect (season,
# play_type, down, epa, air_yards, etc.). info() tells me the row count and
# which columns are stored as numbers vs. strings, which matters for any
# numeric aggregation later.
print("=" * 60)
print("HEAD: first 5 rows of selected columns")
print("=" * 60)
print(sea[['season', 'week', 'posteam', 'play_type', 'down',
           'yards_gained', 'epa']].head())

print("\n" + "=" * 60)
print("INFO: structure of the filtered Seahawks dataframe")
print("=" * 60)
sea.info()


# ======================================================================
# QUESTION 1: DK Metcalf's per-season efficiency, 2019–2024
# ======================================================================
# Across DK Metcalf's six seasons with the Seahawks (2019–2024), how do
# his targets per game, yards per target, and EPA per target vary year
# over year, and which season was his most efficient?

# OPERATION 2: df[df['column'] > value]  (boolean filtering)
# Question: How do I narrow this dataset down to just the plays that
# answer my question? I need plays where DK Metcalf was the targeted
# receiver, and only during the seasons he was actually with Seattle.
# I use str.contains so I'm robust to nflfastR's name format
# ("D.Metcalf" vs. "DK.Metcalf"), and I bound the seasons with >= and <=.
# What the answer means: this gives me only the plays where Metcalf was
# the targeted receiver as a Seahawk. Filtering before grouping keeps the
# per-season stats clean, because anything I aggregate later only reflects
# his actual targets.
metcalf = sea[
    sea['receiver_player_name'].str.contains('Metcalf', na=False)
    & (sea['season'] >= 2019)
    & (sea['season'] <= 2024)
].copy()
print("\n" + "=" * 60)
print(f"QUESTION 1: DK METCALF, {len(metcalf)} targeted plays found")
print("=" * 60)


# OPERATION 3: df.groupby('column')['other'].mean()  (and .agg)
# Question: Year over year, how productive was Metcalf per target?
# I group by season and compute targets, mean yards per target, and mean
# EPA per target. groupby + agg is the right tool here because each season
# is a natural bucket and I want a single row per year.
# What the answer means: this is the year-over-year picture of how
# productive Metcalf was per target, which is closer to true efficiency
# than raw season totals because it controls for volume. To answer "which
# season was his most efficient," I take the season with the highest EPA
# per target, since EPA captures down, distance, and field position rather
# than just raw yardage.
metcalf_summary = metcalf.groupby('season').agg(
    targets=('play_id', 'count'),
    yards_per_target=('yards_gained', 'mean'),
    epa_per_target=('epa', 'mean'),
).round(3)

# Targets per game requires dividing the targets by games played in each
# season. nunique() of game_id within Seattle plays gives me that.
games_per_season = sea.groupby('season')['game_id'].nunique()
metcalf_summary['targets_per_game'] = (
    metcalf_summary['targets'] / games_per_season
).round(2)

print("\nDK Metcalf, season-by-season:")
print(metcalf_summary[['targets', 'targets_per_game',
                       'yards_per_target', 'epa_per_target']])

# Identify his most efficient season by EPA per target.
best_season = metcalf_summary['epa_per_target'].idxmax()
print(f"\nMost efficient season: {best_season} "
      f"({metcalf_summary.loc[best_season, 'epa_per_target']} EPA/target)")


# ======================================================================
# QUESTION 2: Red zone trip TD conversion rate, 2016–2025
# ======================================================================
# What share of the Seahawks' red zone trips ended in a touchdown across
# the past ten seasons, and how has that touchdown conversion rate trended
# over time?
# A red zone "trip" is a drive that reaches inside the opponent's 20-yard
# line, so this is a drive-level question, not a play-level one. I have
# to roll plays up into drives first, then ask whether each drive both
# (a) reached the red zone and (b) ended in a touchdown.

# Reusing OPERATION 3 (groupby) here. I group by (season, game_id, drive)
# to collapse plays into drives, then check two things per drive:
#   - did any play on that drive have yardline_100 <= 20 (red zone reached)
#   - what was the drive's outcome (fixed_drive_result)
# fixed_drive_result is the same on every play within a drive, so 'first'
# is a safe aggregation.
drive_keys = ['season', 'game_id', 'fixed_drive']
drives = sea.groupby(drive_keys).agg(
    reached_red_zone=('yardline_100', lambda x: (x <= 20).any()),
    drive_result=('fixed_drive_result', 'first'),
).reset_index()

# Reusing OPERATION 2 (boolean filter) to keep only drives that actually
# reached the red zone, since those are the "trips" the question asks about.
red_zone_trips = drives[drives['reached_red_zone']].copy()
red_zone_trips['ended_in_td'] = (
    red_zone_trips['drive_result'] == 'Touchdown'
).astype(int)

print("\n" + "=" * 60)
print(f"QUESTION 2: RED ZONE TRIPS, {len(red_zone_trips)} drives reached the 20")
print("=" * 60)

# Now group the trips by season and take the mean of the 0/1 ended_in_td
# column, which gives the per-season TD conversion rate.
rz_td_rate = red_zone_trips.groupby('season')['ended_in_td'].mean().round(3)
print("\nRed zone TD conversion rate by season:")
print(rz_td_rate)
# What the answer means: a value of 0.55 means 55% of red zone trips
# ended in a touchdown that year. A trend down over time would support
# the fan perception that the offense leaves points on the field in the
# red zone; a flat or rising trend would push back on it.


# ======================================================================
# QUESTION 3: Carroll era vs. Macdonald era offensive identity
# ======================================================================
# How does the Seahawks' offensive identity differ between the Pete
# Carroll era (2016–2023) and the Mike Macdonald era (2024–2025),
# measured by early-down pass rate and average depth of target per play?

# Tag each play with the coaching era based on season.
def tag_era(season):
    if season <= 2023:
        return 'Carroll (2016–2023)'
    return 'Macdonald (2024–2025)'

sea['era'] = sea['season'].apply(tag_era)

# Reusing OPERATION 2 (boolean filter) and OPERATION 3 (groupby).
# Early-down (1st and 2nd down) pass rate.
early = sea[
    sea['down'].isin([1, 2])
    & sea['play_type'].isin(['pass', 'run'])
].copy()
early['is_pass'] = (early['play_type'] == 'pass').astype(int)
era_pass_rate = early.groupby('era')['is_pass'].mean().round(3)
print("\n" + "=" * 60)
print("QUESTION 3: CARROLL vs. MACDONALD ERA")
print("=" * 60)
print("\nEarly-down pass rate by era (1st and 2nd downs):")
print(era_pass_rate)
# What the answer means: a higher pass rate signals a more aggressive,
# pass-first identity on the downs where the offense has the most options.

# Average depth of target (aDOT) on pass plays only.
pass_plays = sea[sea['play_type'] == 'pass'].copy()
era_adot = pass_plays.groupby('era')['air_yards'].mean().round(2)
print("\nAverage depth of target (aDOT) by era, pass plays only:")
print(era_adot)
# What the answer means: aDOT shows whether the passing game leans on
# short, quick throws or pushes the ball downfield. It's a useful proxy
# for offensive aggression on a per-attempt basis. Caveat: the Macdonald
# era is only two seasons in this dataset, so any difference is suggestive
# rather than conclusive.

print("\n" + "=" * 60)
print("Done. All three MP1 questions answered.")
print("=" * 60)
