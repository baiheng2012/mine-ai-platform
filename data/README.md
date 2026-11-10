# Data

This directory contains the two public datasets used by the project.
Raw files are not all redistributed here; see each section.

## 1. Seismic-bumps (geohazard module)

- File: `seismic-bumps.arff` (redistributed unmodified)
- Source: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/266/seismic-bumps
- Authors: Sikora, M. and Wrobel, L.
- Content: seismic and seismoacoustic monitoring records from a Polish coal mine (longwall),
  with a binary label `class` marking high-energy seismic events ("seismic bumps").
- License: CC BY 4.0
- Used by: `src/data.py`, `src/train.py`

## 2. Quality Prediction in a Mining Process (flotation module)

- File: `flotation_hourly.csv` (derived, see below)
- Source: Kaggle, "Quality Prediction in a Mining Process" by Eduardo Magalhaes Oliveira
- Content: process data from a mineral flotation plant (reagent flows, pulp pH and density,
  column air flows and levels) and hourly laboratory results (iron/silica in feed and concentrate).
- License: CC0 1.0 (Public Domain)
- Used by: `src/prepare_flotation.py`, `src/train_flotation_final.py`

### How `flotation_hourly.csv` was derived

The raw file contains readings every 20 seconds, but the laboratory values change only once
per hour. To avoid leakage between near-identical rows and to keep the repository small,
the raw data was aggregated to one row per hour:

- every column: hourly mean
- process variables (everything except the four laboratory columns): hourly standard deviation,
  stored with the suffix `__std`
- hours with missing aggregates were dropped; time gaps between hours are kept as they are
  (feature code checks hour continuity before building lag features)

The raw file (`MiningProcess_Flotation_Plant_Database.csv`, tens of MB) is not committed.
To reproduce the hourly file:

1. Download the dataset from Kaggle and extract the CSV into `data/raw/`.
2. Run `uv run python src/prepare_flotation.py`.

## Notes

- `data/raw/` is git-ignored on purpose.
- Both datasets are public and small enough for CI, so the pipeline can be reproduced end to end
  from a fresh clone (CI trains both models from these files).
- Known limitations (single plant, short time range, laboratory values with a delay, no
  controlled experiments) are discussed in the main README.
