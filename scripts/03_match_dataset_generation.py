from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

MATCHES_FILE = DATA_DIR / "historical_wc2026_matches.csv"
FEATURES_FILE = DATA_DIR / "training_features.csv"
OUTPUT_FILE = DATA_DIR / "match_training_dataset.csv"


def get_result(row: pd.Series) -> int:
    if row["home_score"] > row["away_score"]:
        return 1
    if row["home_score"] < row["away_score"]:
        return -1
    return 0


def add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    derived = {}
    pairs = [
        ("win_rate", "win_rate_difference"),
        ("points_per_match", "points_per_match_difference"),
        ("goals_per_match", "attack_difference"),
        ("goals_against_per_match", "defense_difference"),
        ("goal_difference_per_match", "goal_difference_pm_diff"),
        ("form5", "form5_difference"),
        ("form10", "form10_difference"),
        ("form20", "form20_difference"),
        ("weighted_form", "weighted_form_difference"),
        ("weighted_recent_points", "recent_points_difference"),
        ("clean_sheet_pct", "clean_sheet_difference"),
        ("failed_to_score_pct", "failed_to_score_difference"),
        ("neutral_win_rate", "neutral_win_rate_difference"),
        ("fifa_sum_rating", "fifa_sum_difference"),
        ("elo_rating", "elo_difference"),
        ("team_strength_score", "team_strength_difference"),
    ]

    for base, out_col in pairs:
        home_col = f"home_{base}"
        away_col = f"away_{base}"
        if home_col in df.columns and away_col in df.columns:
            derived[out_col] = df[home_col] - df[away_col]

    if {"home_elo_rating", "away_elo_rating"}.issubset(df.columns):
        derived["elo_ratio"] = df["home_elo_rating"] / df["away_elo_rating"].replace(0, np.nan)

    if {"home_fifa_sum_rating", "away_fifa_sum_rating"}.issubset(df.columns):
        derived["fifa_sum_ratio"] = df["home_fifa_sum_rating"] / df["away_fifa_sum_rating"].replace(0, np.nan)

    if {"home_attack_10", "away_defense_10", "away_attack_10", "home_defense_10"}.issubset(df.columns):
        derived["home_expected_goal_edge"] = df["home_attack_10"] - df["away_defense_10"]
        derived["away_expected_goal_edge"] = df["away_attack_10"] - df["home_defense_10"]
        derived["expected_goal_edge_difference"] = (
            derived["home_expected_goal_edge"] - derived["away_expected_goal_edge"]
        )

    if derived:
        df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)

    return df.copy()


matches = pd.read_csv(MATCHES_FILE)
features = pd.read_csv(FEATURES_FILE)

matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
matches = matches.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])

if "year" not in matches.columns:
    matches["year"] = matches["date"].dt.year

matches["target"] = matches.apply(get_result, axis=1)

home_features = features.add_prefix("home_")
away_features = features.add_prefix("away_")

matches = matches.merge(home_features, left_on="home_team", right_on="home_team", how="left")
matches = matches.merge(away_features, left_on="away_team", right_on="away_team", how="left")
matches = add_difference_features(matches)

for col in ["home_group", "away_group"]:
    if col in matches.columns:
        matches[col] = matches[col].fillna("NON_WC")

numeric_cols = matches.select_dtypes(include="number").columns
matches[numeric_cols] = (
    matches[numeric_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(matches[numeric_cols].median(numeric_only=True))
    .fillna(0)
)

matches.to_csv(OUTPUT_FILE, index=False)

print("\nDataset generated")
print(f"Rows: {len(matches)}")
print(f"Columns: {len(matches.columns)}")
print(f"Missing values: {matches.isnull().sum().sum()}")
print("\nTarget distribution:")
print(matches["target"].value_counts().sort_index())