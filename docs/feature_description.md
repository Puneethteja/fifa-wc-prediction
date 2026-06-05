# FIFA World Cup 2026 Feature Engineering Documentation

## Overview

This repository contains the complete feature-engineering pipeline developed for FIFA World Cup 2026 match prediction and tournament simulation.

The project transforms raw historical football match data into machine-learning-ready datasets containing team performance metrics, form indicators, FIFA SUM ratings, encoded categorical variables, and match-level training features.

---

# Data Sources

The feature engineering pipeline utilizes:

* Historical International Match Results (1990–2026)
* FIFA World Cup Historical Results
* Continental Championship Results
* FIFA World Cup Qualification Matches
* FIFA World Cup 2026 Fixture Data

---

# Feature Engineering Workflow

The complete workflow consists of three stages:

```text
Historical Match Data
        ↓
FIFA SUM Rating Generation
        ↓
Team Feature Engineering
        ↓
Match Dataset Generation
```

---

# Team-Level Feature Dataset

## File

`training_features.csv`

## Description

Contains engineered football features for all 48 FIFA World Cup 2026 teams.

## Dataset Statistics

* Teams: 48
* Features: 23
* Missing Values: 0

---

## Team Identification Features

| Feature       | Description                          |
| ------------- | ------------------------------------ |
| team          | Team name                            |
| team_encoded  | Encoded team identifier              |
| group         | FIFA World Cup 2026 group assignment |
| group_encoded | Encoded group identifier             |

---

## Match Performance Features

| Feature         | Description                       |
| --------------- | --------------------------------- |
| matches_played  | Total matches played              |
| wins            | Total wins                        |
| draws           | Total draws                       |
| losses          | Total losses                      |
| goals_for       | Total goals scored                |
| goals_against   | Total goals conceded              |
| goal_difference | Goals scored minus goals conceded |

---

## Efficiency Features

| Feature                   | Description                       |
| ------------------------- | --------------------------------- |
| win_rate                  | Win percentage                    |
| goals_per_match           | Average goals scored per match    |
| goals_against_per_match   | Average goals conceded per match  |
| points_per_match          | Average points earned per match   |
| goal_difference_per_match | Average goal difference per match |

---

## Recent Form Features

Recent form uses the standard football points system:

* Win = 3 points
* Draw = 1 point
* Loss = 0 points

| Feature | Description                      |
| ------- | -------------------------------- |
| form5   | Points earned in last 5 matches  |
| form10  | Points earned in last 10 matches |

---

## Defensive Features

| Feature         | Description                                    |
| --------------- | ---------------------------------------------- |
| clean_sheet_pct | Percentage of matches without conceding a goal |

---

## Neutral Venue Features

| Feature          | Description                       |
| ---------------- | --------------------------------- |
| neutral_matches  | Number of neutral venue matches   |
| neutral_win_rate | Win rate in neutral venue matches |

---

## Weighted Form Feature

Recent matches are assigned greater importance than older matches.

| Feature       | Description                       |
| ------------- | --------------------------------- |
| weighted_form | Recency-weighted team form metric |

---

## FIFA SUM Rating

The FIFA SUM rating is a long-term team strength metric inspired by FIFA's official ranking methodology.

The rating is calculated using historical international matches from 1990–2026 and incorporates match importance factors.

Examples:

* Friendly Matches
* Nations League Matches
* Continental Qualifiers
* Continental Championships
* FIFA World Cup Qualification
* FIFA World Cup Final Tournament

| Feature         | Description                    |
| --------------- | ------------------------------ |
| fifa_sum_rating | Long-term team strength rating |

---

# FIFA SUM Rating Dataset

## File

`fifa_sum_ratings.csv`

## Description

Contains FIFA SUM ratings for international football teams.

## Dataset Statistics

* Teams: 326
* Features: 2

## Columns

| Column          | Description     |
| --------------- | --------------- |
| team            | Team name       |
| fifa_sum_rating | FIFA SUM rating |

Purpose:

Provides a universal strength rating for all teams, including non-World Cup participants.

---

# Match-Level Training Dataset

## File

`match_training_dataset.csv`

## Description

Machine-learning-ready dataset created by combining historical match results with engineered team-level features.

## Dataset Statistics

* Matches: 1675
* Features: 57
* Missing Values: 0

---

## Match Metadata

| Feature    |
| ---------- |
| date       |
| home_team  |
| away_team  |
| home_score |
| away_score |
| tournament |
| city       |
| country    |
| neutral    |
| year       |
| winner     |

---

## Target Variable

Match outcome is encoded as:

| Value | Meaning  |
| ----- | -------- |
| 1     | Home Win |
| 0     | Draw     |
| -1    | Away Win |

Feature:

`target`

---

## Home Team Features

For every match, the full feature profile of the home team is attached.

Examples:

* home_win_rate
* home_form5
* home_form10
* home_weighted_form
* home_clean_sheet_pct
* home_goals_per_match
* home_group_encoded
* home_team_encoded
* home_fifa_sum_rating

---

## Away Team Features

For every match, the full feature profile of the away team is attached.

Examples:

* away_win_rate
* away_form5
* away_form10
* away_weighted_form
* away_clean_sheet_pct
* away_goals_per_match
* away_group_encoded
* away_team_encoded
* away_fifa_sum_rating

---

## Rating-Based Features

### home_fifa_sum_rating

FIFA SUM rating of the home team.

### away_fifa_sum_rating

FIFA SUM rating of the away team.

### rating_difference

Computed as:

```text
rating_difference =
home_fifa_sum_rating -
away_fifa_sum_rating
```

This feature captures the relative strength difference between competing teams and serves as an important predictive variable.

---

# Supporting Files

## worldcup2026_groups.csv

Contains reconstructed FIFA World Cup 2026 group assignments.

Statistics:

* Teams: 48
* Groups: 12

---

## team_encoding_map.csv

Contains mappings between team names and encoded identifiers.

Columns:

* team
* team_encoded

---

# Scripts

The repository includes reproducible data-generation pipelines.

## 01_team_feature_engineering.py

Generates:

* training_features.csv
* team_encoding_map.csv

Responsibilities:

* Generate team performance statistics
* Generate recent form metrics
* Generate defensive metrics
* Generate neutral venue metrics
* Generate weighted form metrics
* Encode categorical variables
* Merge FIFA SUM ratings

---

## 02_fifa_sum_rating.py

Generates:

* fifa_sum_ratings.csv

Responsibilities:

* Process historical match data
* Apply FIFA SUM methodology
* Calculate long-term team ratings

---

## 03_match_dataset_generation.py

Generates:

* match_training_dataset.csv

Responsibilities:

* Merge team features
* Merge FIFA SUM ratings
* Create target variable
* Generate rating_difference
* Handle missing values

---

# Repository Structure

```text
data/
├── training_features.csv
├── final_wc2026_features_with_groups.csv
├── fifa_sum_ratings.csv
├── match_training_dataset.csv
├── worldcup2026_groups.csv
├── team_encoding_map.csv

docs/
├── feature_description.md

scripts/
├── 01_team_feature_engineering.py
├── 02_fifa_sum_rating.py
├── 03_match_dataset_generation.py
```

---

# Reproducibility

The complete feature-engineering pipeline can be reproduced using:

1. 02_fifa_sum_rating.py
2. 01_team_feature_engineering.py
3. 03_match_dataset_generation.py

Execution Order:

```text
Historical Match Data
        ↓
02_fifa_sum_rating.py
        ↓
fifa_sum_ratings.csv
        ↓
01_team_feature_engineering.py
        ↓
training_features.csv
        ↓
03_match_dataset_generation.py
        ↓
match_training_dataset.csv
```

---

# Intended Usage

The generated datasets are suitable for:

* XGBoost
* Random Forest
* Logistic Regression
* Gradient Boosting
* Ensemble Learning
* Monte Carlo Tournament Simulation

Recommended Workflow:

1. Train models using match_training_dataset.csv
2. Predict FIFA World Cup 2026 group-stage fixtures
3. Simulate group standings
4. Estimate qualification probabilities
5. Simulate knockout rounds
6. Forecast tournament outcomes

The match-level training dataset serves as the primary machine-learning input for FIFA World Cup 2026 prediction and tournament forecasting.
