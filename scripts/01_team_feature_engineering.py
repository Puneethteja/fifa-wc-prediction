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

CORRECT_GROUPS = {
    "Brazil": "C",
    "Haiti": "C",
    "Morocco": "C",
    "Scotland": "C",
    "Australia": "D",
    "Paraguay": "D",
    "Turkey": "D",
    "United States": "D",
}


def normalize_groups(groups: pd.DataFrame) -> pd.DataFrame:
    groups = groups.copy()
    groups["group"] = groups["team"].map(CORRECT_GROUPS).fillna(groups["group"])
    return groups


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


def weighted_recent_average(values: list[int]) -> float:
    if not values:
        return 0.0
    recent = np.array(values[-20:], dtype=float)
    weights = np.linspace(0.6, 1.4, len(recent))
    return float(np.average(recent, weights=weights))


def build_team_features() -> pd.DataFrame:
    matches = pd.read_csv(MATCHES_FILE)
    groups = normalize_groups(pd.read_csv(GROUPS_FILE))
    fifa_sum = latest_team_ratings(FIFA_SUM_FILE, "fifa_sum_rating", 1000.0)

    elo_path = ELO_FILE if ELO_FILE.exists() else ELO_FALLBACK_FILE
    elo = latest_team_ratings(elo_path, "rating", 1500.0).rename(columns={"rating": "elo_rating"})

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches["neutral"] = matches["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])
    matches = matches.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    matches = matches.sort_values("date")

    rows: list[dict[str, float | int | str]] = []
    for team in sorted(groups["team"].dropna().unique()):
        team_matches = matches[
            (matches["home_team"] == team) | (matches["away_team"] == team)
        ].sort_values("date")

        matches_played = len(team_matches)
        points: list[int] = []
        scored: list[float] = []
        conceded: list[float] = []
        goal_margins: list[float] = []
        neutral_points: list[int] = []
        clean_sheets = 0
        failed_to_score = 0
        wins = draws = losses = 0

        for _, row in team_matches.iterrows():
            team_points = match_points_for_team(row, team)
            gf, ga = goals_for_team(row, team)

            points.append(team_points)
            scored.append(gf)
            conceded.append(ga)
            goal_margins.append(gf - ga)

            if team_points == 3:
                wins += 1
            elif team_points == 1:
                draws += 1
            else:
                losses += 1

            clean_sheets += int(ga == 0)
            failed_to_score += int(gf == 0)
            if bool(row.get("neutral", False)):
                neutral_points.append(team_points)

        rows.append({
            "team": team,
            "matches_played": matches_played,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_for": float(sum(scored)),
            "goals_against": float(sum(conceded)),
            "goal_difference": float(sum(goal_margins)),
            "win_rate": safe_rate(wins, matches_played),
            "draw_rate": safe_rate(draws, matches_played),
            "loss_rate": safe_rate(losses, matches_played),
            "unbeaten_rate": safe_rate(wins + draws, matches_played),
            "goals_per_match": safe_rate(sum(scored), matches_played),
            "goals_against_per_match": safe_rate(sum(conceded), matches_played),
            "goal_difference_per_match": safe_rate(sum(goal_margins), matches_played),
            "points_per_match": safe_rate(sum(points), matches_played),
            "form5": sum(points[-5:]),
            "form10": sum(points[-10:]),
            "form20": sum(points[-20:]),
            "weighted_form": weighted_recent_average(points) * 10,
            "weighted_recent_points": weighted_recent_average(points),
            "attack_5": safe_mean(scored, 5),
            "defense_5": safe_mean(conceded, 5),
            "attack_10": safe_mean(scored, 10),
            "defense_10": safe_mean(conceded, 10),
            "attack_20": safe_mean(scored, 20),
            "defense_20": safe_mean(conceded, 20),
            "goal_diff_10": safe_mean(goal_margins, 10),
            "goal_diff_20": safe_mean(goal_margins, 20),
            "recent_goal_trend": safe_mean(scored, 5) - safe_mean(scored, 20),
            "avg_goal_margin": safe_mean(goal_margins),
            "clean_sheet_pct": safe_rate(clean_sheets, matches_played),
            "failed_to_score_pct": safe_rate(failed_to_score, matches_played),
            "neutral_matches": len(neutral_points),
            "neutral_points_per_match": safe_rate(sum(neutral_points), len(neutral_points)),
            "neutral_win_rate": safe_rate(sum(1 for value in neutral_points if value == 3), len(neutral_points)),
        })

    features = pd.DataFrame(rows)
    features = features.merge(groups, on="team", how="left")
    features = features.merge(fifa_sum, on="team", how="left")
    features = features.merge(elo, on="team", how="left")
    features["fifa_sum_rating"] = features["fifa_sum_rating"].fillna(1000.0)
    features["elo_rating"] = features["elo_rating"].fillna(1500.0)

    team_map = {team: idx for idx, team in enumerate(sorted(features["team"].unique()))}
    group_map = {group: idx for idx, group in enumerate(sorted(features["group"].dropna().unique()))}
    features["team_encoded"] = features["team"].map(team_map).astype(int)
    features["group_encoded"] = features["group"].map(group_map).fillna(-1).astype(int)
    features["elo_rank"] = features["elo_rating"].rank(ascending=False, method="min")
    features["fifa_sum_rank"] = features["fifa_sum_rating"].rank(ascending=False, method="min")
    features["attack_defense_balance"] = features["attack_10"] - features["defense_10"]
    features["elo_adjusted_form"] = features["weighted_form"] * (features["elo_rating"] / features["elo_rating"].max())
    features["fifa_adjusted_form"] = features["weighted_form"] * (features["fifa_sum_rating"] / features["fifa_sum_rating"].max())

    strength_cols = [
        "points_per_match",
        "win_rate",
        "goal_difference_per_match",
        "weighted_recent_points",
        "elo_rating",
        "fifa_sum_rating",
    ]
    normalized = []
    for col in strength_cols:
        col_min = features[col].min()
        col_max = features[col].max()
        if col_max == col_min:
            normalized.append(pd.Series(0.0, index=features.index))
        else:
            normalized.append((features[col] - col_min) / (col_max - col_min))
    features["team_strength_score"] = pd.concat(normalized, axis=1).mean(axis=1)

    numeric_cols = features.select_dtypes(include="number").columns
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return features.sort_values("team").reset_index(drop=True)


if __name__ == "__main__":
    output = build_team_features()
    output.to_csv(OUTPUT_FILE, index=False)
    output.to_csv(FINAL_FEATURES_FILE, index=False)
    output[["team", "team_encoded", "group", "group_encoded"]].to_csv(ENCODING_FILE, index=False)

    print(f"Saved {len(output)} teams to {OUTPUT_FILE}")
    print(output.groupby("group")["team"].apply(list))
