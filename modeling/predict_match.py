from pathlib import Path
import pickle

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

FEATURES_FILE = DATA_DIR / "training_features.csv"
FIXTURES_FILE = DATA_DIR / "worldcup2026_fixtures.csv"
OUTPUT_FILE = DATA_DIR / "worldcup_2026_group_stage_probabilities.csv"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


rf = load_pickle(MODEL_DIR / "rf_model.pkl")
xgb_data = load_pickle(MODEL_DIR / "xgb_model.pkl")
xgb = xgb_data["model"]
feature_cols = load_pickle(MODEL_DIR / "feature_cols.pkl")

team_features = pd.read_csv(FEATURES_FILE)
team_features = team_features.set_index("team", drop=False)


def rf_probabilities(model, X: pd.DataFrame) -> dict[int, float]:
    probs = model.predict_proba(X)[0]
    return {int(cls): float(prob) for cls, prob in zip(model.classes_, probs)}


def xgb_probabilities(model, X: pd.DataFrame) -> dict[int, float]:
    probs = model.predict_proba(X)[0]
    return {-1: float(probs[0]), 0: float(probs[1]), 1: float(probs[2])}


def difference_features(home: pd.Series, away: pd.Series) -> dict[str, float]:
    values = {}
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
        values[out_col] = float(home.get(base, 0)) - float(away.get(base, 0))

    away_elo = float(away.get("elo_rating", 0))
    away_fifa = float(away.get("fifa_sum_rating", 0))
    values["elo_ratio"] = float(home.get("elo_rating", 0)) / away_elo if away_elo else 1.0
    values["fifa_sum_ratio"] = float(home.get("fifa_sum_rating", 0)) / away_fifa if away_fifa else 1.0

    values["home_expected_goal_edge"] = float(home.get("attack_10", 0)) - float(away.get("defense_10", 0))
    values["away_expected_goal_edge"] = float(away.get("attack_10", 0)) - float(home.get("defense_10", 0))
    values["expected_goal_edge_difference"] = values["home_expected_goal_edge"] - values["away_expected_goal_edge"]
    return values


def build_match_features(home_team: str, away_team: str) -> pd.DataFrame:
    if home_team not in team_features.index:
        raise ValueError(f"No features found for home team: {home_team}")
    if away_team not in team_features.index:
        raise ValueError(f"No features found for away team: {away_team}")

    home = team_features.loc[home_team]
    away = team_features.loc[away_team]
    row: dict[str, float] = {}

    for col in feature_cols:
        if col.startswith("home_"):
            row[col] = home.get(col.removeprefix("home_"), 0)
        elif col.startswith("away_"):
            row[col] = away.get(col.removeprefix("away_"), 0)

    row.update(difference_features(home, away))
    X = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def predict_match(home_team: str, away_team: str, verbose: bool = True) -> dict[str, float]:
    X = build_match_features(home_team, away_team)
