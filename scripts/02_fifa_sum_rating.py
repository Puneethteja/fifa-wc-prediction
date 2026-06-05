from pathlib import Path
import math

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "all_matches.csv"
OUTPUT_FILE = DATA_DIR / "fifa_sum_ratings.csv"

STARTING_RATING = 1000.0
MODERN_ERA_START = "1990-01-01"

IMPORTANCE = {
    "Friendly": 10,
    "UEFA Nations League": 15,
    "CONCACAF Nations League": 15,
    "FIFA World Cup qualification": 25,
    "UEFA Euro qualification": 25,
    "African Cup of Nations qualification": 25,
    "AFC Asian Cup qualification": 25,
    "Gold Cup qualification": 25,
    "CONCACAF Championship qualification": 25,
    "UEFA Euro": 35,
    "Copa America": 35,
    "Copa América": 35,
    "African Cup of Nations": 35,
    "AFC Asian Cup": 35,
    "Gold Cup": 35,
    "Oceania Nations Cup": 35,
    "Confederations Cup": 40,
    "FIFA World Cup": 50,
}


def expected_score(team_rating: float, opponent_rating: float) -> float:
    return 1.0 / (math.pow(10.0, (opponent_rating - team_rating) / 600.0) + 1.0)


def actual_scores(home_score: float, away_score: float) -> tuple[float, float]:
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
df = df[df["date"] >= MODERN_ERA_START].sort_values("date")

teams = sorted(set(df["home_team"]).union(df["away_team"]))
ratings = {team: STARTING_RATING for team in teams}
matches_played = {team: 0 for team in teams}

for _, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    tournament = row.get("tournament", "Friendly")
    importance = IMPORTANCE.get(tournament, 10)

    home_rating = ratings[home]
    away_rating = ratings[away]
    home_actual, away_actual = actual_scores(row["home_score"], row["away_score"])

    home_expected = expected_score(home_rating, away_rating)
    away_expected = expected_score(away_rating, home_rating)

    ratings[home] = home_rating + importance * (home_actual - home_expected)
    ratings[away] = away_rating + importance * (away_actual - away_expected)
    matches_played[home] += 1
    matches_played[away] += 1

ratings_df = pd.DataFrame({
    "team": list(ratings.keys()),
    "fifa_sum_rating": list(ratings.values()),
    "fifa_sum_matches": [matches_played[team] for team in ratings.keys()],
})

ratings_df = ratings_df.sort_values("fifa_sum_rating", ascending=False).reset_index(drop=True)
ratings_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(ratings_df)} teams to {OUTPUT_FILE}")
print("\nTop 10 Teams:\n")
print(ratings_df.head(10))