"""
A5 — Pandas analysis of Seattle Seahawks play-by-play data
Author: Jasmine Sayed
HCDE 530, Spring 2026

Dataset: NFL play-by-play data from the nflverse project, accessed via the
nflreadpy Python package (https://github.com/nflverse/nflreadpy).
Scope: 2016-2025 regular and postseason. Filtered to plays where Seattle
was the offense (posteam == 'SEA').

The script answers the three analytical questions from MP1a:
    1. DK Metcalf's per-season efficiency across his Seahawks tenure (2019-2024)
    2. Seahawks red zone touchdown rate trend across the past 10 seasons
    3. Carroll era vs. Macdonald era offensive identity comparison

It uses all five pandas operations from class:
    df.head(), df.info(), df.isnull().sum(),
    df['column'].value_counts(), df[df['column'] > value], df.groupby(...).
"""

import nflreadpy as nfl
import pandas as pd

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
# Pull play-by-play for the past ten seasons. nflreadpy returns a Polars
# DataFrame, so I convert to pandas right away because the assignment
# (and the rest of class) is built around pandas.
print("Loading nflverse play-by-play data for 2016-2025...")
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


# OPERATION 2: df.isnull().sum()
# Question: Where is data missing in the columns I plan to use? Some
# columns (air_yards, receiver_player_name) only exist on pass plays, and
# epa is undefined on certain non-football plays.
# What the answer means: this tells me which fields I need to filter or
# guard against before aggregating, so I don't accidentally average NaNs
# into my results.
print("\n" + "=" * 60)
print("MISSING VALUES (selected columns)")
print("=" * 60)
missing = sea[['epa', 'air_yards', 'play_type', 'down', 'yardline_100',
               'touchdown', 'receiver_player_name']].isnull().sum()
print(missing)


# OPERATION 3: df['column'].value_counts()
# Question: What is the rough mix of play types Seattle has run over the
# past ten seasons? This is a sanity check and a quick look at the offense.
# What the answer means: if pass and run dominate the counts (with smaller
# numbers for kicks, punts, and no_play penalties), the data looks right.
# The pass-to-run ratio is also a quick first read on offensive identity.
print("\n" + "=" * 60)
print("PLAY TYPE COUNTS, 2016-2025")
print("=" * 60)
print(sea['play_type'].value_counts())


# ======================================================================
# QUESTION 1: DK Metcalf's per-season efficiency, 2019-2024
# ======================================================================
# Across DK Metcalf's six seasons with the Seahawks, how do his targets
# per game, yards per target, and EPA per target vary year over year, and
# which season was his most efficient by EPA per target?

# OPERATION 4: df[df['column'] > value]  (boolean filtering)
# I filter to pass plays targeting DK Metcalf. I use str.contains so I'm
# robust to nflfastR's name format ("D.Metcalf" or "DK.Metcalf"), and I
# also restrict to the seasons he was actually with Seattle.
metcalf = sea[
    sea['receiver_player_name'].str.contains('Metcalf', na=False)
    & (sea['season'] >= 2019)
    & (sea['season'] <= 2024)
].copy()
# What the answer means: this gives me only the plays where Metcalf was
# the targeted receiver as a Seahawk. Filtering before grouping keeps the
# per-season stats clean.
print("\n" + "=" * 60)
print(f"QUESTION 1: DK METCALF — {len(metcalf)} targeted plays found")
print("=" * 60)


# OPERATION 5: df.groupby('column')['other'].mean()  (and .agg)
# Group by season and compute targets, mean yards per target, and mean EPA
# per target. groupby + agg is the right tool here because each season is
# a natural bucket and I want a single row per year.
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
print(f"\nMost efficient season by EPA per target: {best_season} "
      f"({metcalf_summary.loc[best_season, 'epa_per_target']} EPA/target)")
# What the answer means: this is the year-over-year picture of how
# productive Metcalf was per target, which is closer to true efficiency
# than raw season totals because it controls for volume.


# ======================================================================
# QUESTION 2: Seahawks red zone TD rate, 2016-2025
# ======================================================================
# What share of the Seahawks' red zone plays (yardline_100 <= 20) ended in
# a touchdown across the past ten seasons, and how has that rate trended?

# Filter to red zone plays that were either passes or runs (excluding
# kicks, penalties, etc., which would distort the rate).
red_zone = sea[
    (sea['yardline_100'] <= 20)
    & (sea['play_type'].isin(['pass', 'run']))
].copy()
print("\n" + "=" * 60)
print(f"QUESTION 2: RED ZONE — {len(red_zone)} plays inside the 20")
print("=" * 60)

# groupby season and take the mean of the touchdown column. Since
# touchdown is a 0/1 column, the mean is the per-play TD rate.
rz_td_rate = red_zone.groupby('season')['touchdown'].mean().round(3)
print("\nRed zone touchdown rate (per play) by season:")
print(rz_td_rate)
# What the answer means: a rate of, say, 0.110 means roughly 11% of red
# zone plays produced a TD that year. A trend down over time would support
# the fan perception of red zone struggles; a flat or rising trend would
# push back on it.


# ======================================================================
# QUESTION 3: Carroll era vs. Macdonald era offensive identity
# ======================================================================
# How does the Seahawks' offensive identity differ between the Pete
# Carroll era (head coach 2010-2023, focusing on 2016-2023 in my window)
# and the Mike Macdonald era (2024-2025)?
# Comparison metrics: early-down pass rate, average depth of target (aDOT),
# and EPA per play.

# Tag each play with the coaching era based on season.
def tag_era(season):
    if season <= 2023:
        return 'Carroll (2016-2023)'
    return 'Macdonald (2024-2025)'

sea['era'] = sea['season'].apply(tag_era)

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
# for offensive aggression on a per-attempt basis.

# EPA per play across all offensive plays (passes and runs).
offense = sea[sea['play_type'].isin(['pass', 'run'])].copy()
era_epa = offense.groupby('era')['epa'].mean().round(3)
print("\nEPA per play by era, offensive plays only:")
print(era_epa)
# What the answer means: EPA per play is the single best summary metric
# for offensive efficiency. Comparing the two eras on this number tells me
# whether the Macdonald offense has actually been better or worse on a
# per-snap basis, with the caveat that two seasons is a small sample.

print("\n" + "=" * 60)
print("Done. All three MP1 questions answered.")
print("=" * 60)
