from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

MATCHES_FILE = DATA_DIR / "historical_wc2026_matches.csv"
GROUPS_FILE = DATA_DIR / "worldcup2026_groups.csv"
FIFA_SUM_FILE = DATA_DIR / "fifa_sum_ratings.csv"
ELO_FILE = DATA_DIR / "eloratings.csv"
ELO_FALLBACK_FILE = DATA_DIR / "elo_ratings.csv"

OUTPUT_FILE = DATA_DIR / "training_features.csv"
FINAL_FEATURES_FILE = DATA_DIR / "final_wc2026_features_with_groups.csv"
ENCODING_FILE = DATA_DIR / "team_encoding_map.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def latest_team_ratings(path: Path, value_col: str, default_value: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["team", value_col])

    df = pd.read_csv(path)
    if "team" not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=["team", value_col])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["team", "date"])
        df = df.groupby("team", as_index=False).tail(1)
    else:
        df = df.groupby("team", as_index=False).tail(1)

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(default_value)
    return df[["team", value_col]]


def safe_mean(values: list[float], n: int | None = None) -> float:
    if not values:
        return 0.0
    values = values[-n:] if n else values
    return float(np.mean(values))


def safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def match_points_for_team(row: pd.Series, team: str) -> int:
    if row["home_score"] == row["away_score"]:
        return 1
    home_win = row["home_score"] > row["away_score"]
    return 3 if (row["home_team"] == team and home_win) or (row["away_team"] == team and not home_win) else 0


def goals_for_team(row: pd.Series, team: str) -> tuple[float, float]:
    if row["home_team"] == team:
        return float(row["home_score"]), float(row["away_score"])
    return float(row["away_score"]), float(row["home_score"])


matches = read_csv(MATCHES_FILE)
groups = read_csv(GROUPS_FILE)
fifa_sum = latest_team_ratings(FIFA_SUM_FILE, "fifa_sum_rating", 1000.0)

elo_path = ELO_FILE if ELO_FILE.exists() else ELO_FALLBACK_FILE
elo = latest_team_ratings(elo_path, "rating", 1500.0).rename(columns={"rating": "elo_rating"})

matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
matches["neutral"] = matches["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])
matches = matches.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
matches = matches.sort_values("date")

wc_teams = sorted(groups["team"].dropna().unique())
rows: list[dict[str, float | int | str]] = []

for team in wc_teams:
    team_matches = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ].sort_values("date")

    matches_played = len(team_matches)
    if matches_played == 0:
        rows.append({"team": team, "matches_played": 0})
        continue

    points: list[int] = []
    scored: list[float] = []
    conceded: list[float] = []
    goal_margins: list[float] = []
    neutral_points: list[int] = []
    clean_sheets = 0
    failed_to_score = 0

    for _, row in team_matches.iterrows():
        gf, ga = goals_for_team(row, team)
        pts = match_points_for_team(row, team)

        points.append(pts)
        scored.append(gf)
        conceded.append(ga)
        goal_margins.append(gf - ga)
        clean_sheets += int(ga == 0)
        failed_to_score += int(gf == 0)

        if bool(row["neutral"]):
            neutral_points.append(pts)

    wins = int(sum(1 for p in points if p == 3))
    draws = int(sum(1 for p in points if p == 1))
    losses = matches_played - wins - draws
    goals_for = float(sum(scored))
    goals_against = float(sum(conceded))
    goal_difference = goals_for - goals_against

    form5 = int(sum(points[-5:]))
    form10 = int(sum(points[-10:]))
    form20 = int(sum(points[-20:]))
    recent_weights = np.linspace(0.35, 1.0, min(len(points), 10))
    weighted_recent_points = float(np.average(points[-10:], weights=recent_weights))

    attack_5 = safe_mean(scored, 5)
    defense_5 = safe_mean(conceded, 5)
    attack_10 = safe_mean(scored, 10)
    defense_10 = safe_mean(conceded, 10)
    attack_20 = safe_mean(scored, 20)
    defense_20 = safe_mean(conceded, 20)

    rows.append({
        "team": team,
        "matches_played": matches_played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goal_difference,
        "win_rate": safe_rate(wins, matches_played),
        "draw_rate": safe_rate(draws, matches_played),
        "loss_rate": safe_rate(losses, matches_played),
        "unbeaten_rate": safe_rate(wins + draws, matches_played),
        "goals_per_match": safe_rate(goals_for, matches_played),
        "goals_against_per_match": safe_rate(goals_against, matches_played),
        "goal_difference_per_match": safe_rate(goal_difference, matches_played),
        "points_per_match": safe_rate((wins * 3) + draws, matches_played),
        "form5": form5,
        "form10": form10,
        "form20": form20,
        "weighted_form": (form5 * 0.55) + (form10 * 0.30) + (form20 * 0.15),
        "weighted_recent_points": weighted_recent_points,
        "attack_5": attack_5,
        "defense_5": defense_5,
        "attack_10": attack_10,
        "defense_10": defense_10,
        "attack_20": attack_20,
        "defense_20": defense_20,
        "goal_diff_10": attack_10 - defense_10,
        "goal_diff_20": attack_20 - defense_20,
        "recent_goal_trend": (attack_5 - defense_5) - (attack_20 - defense_20),
        "avg_goal_margin": safe_mean(goal_margins),
        "clean_sheet_pct": safe_rate(clean_sheets, matches_played),
        "failed_to_score_pct": safe_rate(failed_to_score, matches_played),
        "neutral_matches": len(neutral_points),
        "neutral_points_per_match": safe_mean(neutral_points),
        "neutral_win_rate": safe_rate(sum(1 for p in neutral_points if p == 3), len(neutral_points)),
    })

features = pd.DataFrame(rows)
features = features.merge(groups, on="team", how="left")
features = features.merge(fifa_sum, on="team", how="left")
features = features.merge(elo, on="team", how="left")

features["fifa_sum_rating"] = features["fifa_sum_rating"].fillna(features["fifa_sum_rating"].median()).fillna(1000.0)
features["elo_rating"] = features["elo_rating"].fillna(features["elo_rating"].median()).fillna(1500.0)

features["team_encoded"] = pd.factorize(features["team"], sort=True)[0]
features["group_encoded"] = pd.factorize(features["group"].fillna("NON_WC"), sort=True)[0]

pd.DataFrame({
    "team": features["team"],
    "team_encoded": features["team_encoded"],
    "group": features["group"],
    "group_encoded": features["group_encoded"],
}).to_csv(ENCODING_FILE, index=False)

features["elo_rank"] = features["elo_rating"].rank(ascending=False, method="dense")
features["fifa_sum_rank"] = features["fifa_sum_rating"].rank(ascending=False, method="dense")
features["attack_defense_balance"] = features["attack_10"] - features["defense_10"]
features["elo_adjusted_form"] = features["form10"] * (features["elo_rating"] / 2000.0)
features["fifa_adjusted_form"] = features["form10"] * (features["fifa_sum_rating"] / 1500.0)
features["team_strength_score"] = (
    (features["elo_rating"] / 2000.0) * 0.35
    + (features["fifa_sum_rating"] / 1600.0) * 0.30
    + features["points_per_match"] / 3.0 * 0.20
    + features["goal_difference_per_match"].clip(-3, 3) / 6.0 * 0.10
    + features["weighted_recent_points"] / 3.0 * 0.05
)

numeric_cols = features.select_dtypes(include="number").columns
features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

features = features.sort_values(["group", "team"]).reset_index(drop=True)
features.to_csv(OUTPUT_FILE, index=False)
features.to_csv(FINAL_FEATURES_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(f"Saved: {FINAL_FEATURES_FILE}")
print(f"Shape: {features.shape}")
