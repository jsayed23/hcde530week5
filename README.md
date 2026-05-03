# A5: Pandas analysis of Seattle Seahawks play-by-play data

Author: Jasmine Sayed
HCDE 530, Spring 2026

This repo contains the A5 deliverable: a pandas script that analyzes ten seasons of Seattle Seahawks play-by-play data to answer the three analytical questions from MP1a.

## Files

- `a5_seahawks.py`: the pandas script. Loads NFL play-by-play data via `nflreadpy`, filters to Seattle, and answers three analytical questions using three pandas operations from class (`df.head()` / `df.info()`, boolean filtering, and `groupby`).
- `week5.md`: competency claims for C5 (Data Analysis with Pandas) and C2 (Code Literacy and Documentation).

## How to run

```bash
pip install nflreadpy pandas
python a5_seahawks.py
```

`nflreadpy` will download the play-by-play data on first run and cache it locally. The script prints results to stdout.

## Data source

NFL play-by-play data from the [nflverse](https://github.com/nflverse) project, accessed via the [`nflreadpy`](https://github.com/nflverse/nflreadpy) Python package.

## Questions answered

1. Across DK Metcalf's six seasons with the Seahawks (2019–2024), how do his targets per game, yards per target, and EPA per target vary year over year, and which season was his most efficient?
2. What share of the Seahawks' red zone trips ended in a touchdown across the past ten seasons, and how has that touchdown conversion rate trended over time?
3. How does the Seahawks' offensive identity differ between the Pete Carroll era (2016–2023) and the Mike Macdonald era (2024–2025), measured by early-down pass rate and average depth of target per play?
