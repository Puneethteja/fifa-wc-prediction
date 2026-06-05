from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

N_SIMULATIONS = 10000
RANDOM_SEED = 42

MATCH_FILE = DATA_DIR / "worldcup_2026_group_stage_probabilities.csv"
GROUPS_FILE = DATA_DIR / "worldcup2026_groups.csv"

OUTPUT_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

matches = pd.read_csv(MATCH_FILE)
groups_df = pd.read_csv(GROUPS_FILE)
groups = {
    group: sorted(group_rows["team"].tolist())
    for group, group_rows in groups_df.groupby("group")
}
matches_by_group = {
    group: group_matches.to_dict("records")
    for group, group_matches in matches.groupby("group")
}

qualification_count = defaultdict(int)
position_count = defaultdict(lambda: [0, 0, 0, 0])
points_total = defaultdict(float)
sample_group_standings = []


def normalize_probabilities(row: dict) -> np.ndarray:
    probs = np.array([
        row["home_win_prob"],
        row["draw_prob"],
        row["away_win_prob"],
    ], dtype=float)
    if probs.sum() <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3])
    return probs / probs.sum()


def team_strength(row: dict, team: str) -> float:
    if team == row["home_team"]:
        return float(row.get("home_team_strength_score", 0))
    return float(row.get("away_team_strength_score", 0))


def draw_score(mean_goals: float) -> int:
    return int(rng.poisson(np.clip(mean_goals, 0.2, 3.8)))


def simulate_score(row: dict, outcome: str) -> tuple[int, int]:
    home_edge = float(row.get("home_expected_points", 1.0)) - float(row.get("away_expected_points", 1.0))
    home_lambda = np.clip(1.25 + home_edge * 0.35, 0.35, 3.5)
    away_lambda = np.clip(1.25 - home_edge * 0.35, 0.35, 3.5)

    home_goals = draw_score(home_lambda)
    away_goals = draw_score(away_lambda)

    if outcome == "home" and home_goals <= away_goals:
        home_goals = away_goals + int(rng.integers(1, 3))
    elif outcome == "away" and away_goals <= home_goals:
        away_goals = home_goals + int(rng.integers(1, 3))
    elif outcome == "draw":
        level = int(round((home_goals + away_goals) / 2))
        home_goals = away_goals = max(0, level)

    return home_goals, away_goals


def sort_group_table(table: dict[str, dict[str, float]]) -> list[tuple[str, dict[str, float]]]:
    return sorted(
        table.items(),
        key=lambda item: (
            item[1]["points"],
            item[1]["goal_difference"],
            item[1]["goals_for"],
            item[1]["wins"],
            item[1]["expected_points"],
            item[1]["strength"],
            rng.random(),
        ),
        reverse=True,
    )


for sim in range(N_SIMULATIONS):
    winners = []
    runners_up = []
    third_place_teams = []

    for group_name, teams in groups.items():
        table = {
            team: {
                "points": 0,
                "wins": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_difference": 0,
                "expected_points": 0.0,
                "strength": 0.0,
            }
            for team in teams
        }

        for row in matches_by_group.get(group_name, []):
            home = row["home_team"]
            away = row["away_team"]
            if home not in table or away not in table:
                continue

            probs = normalize_probabilities(row)
            outcome = rng.choice(["home", "draw", "away"], p=probs)
            home_goals, away_goals = simulate_score(row, outcome)

            table[home]["goals_for"] += home_goals
            table[home]["goals_against"] += away_goals
            table[away]["goals_for"] += away_goals
            table[away]["goals_against"] += home_goals

            table[home]["goal_difference"] = table[home]["goals_for"] - table[home]["goals_against"]
            table[away]["goal_difference"] = table[away]["goals_for"] - table[away]["goals_against"]

            table[home]["expected_points"] += float(row.get("home_expected_points", 0))
            table[away]["expected_points"] += float(row.get("away_expected_points", 0))
            table[home]["strength"] = max(table[home]["strength"], team_strength(row, home))
            table[away]["strength"] = max(table[away]["strength"], team_strength(row, away))

            if outcome == "home":
                table[home]["points"] += 3
                table[home]["wins"] += 1
            elif outcome == "away":
                table[away]["points"] += 3
                table[away]["wins"] += 1
            else:
                table[home]["points"] += 1
                table[away]["points"] += 1

        standings = sort_group_table(table)

        for pos, (team, stats) in enumerate(standings):
            position_count[team][pos] += 1
            points_total[team] += stats["points"]

            if sim == 0:
                sample_group_standings.append({
                    "group": group_name,
                    "position": pos + 1,
                    "team": team,
                    "points": stats["points"],
                    "goal_difference": stats["goal_difference"],
                    "goals_for": stats["goals_for"],
                })

        winners.append(standings[0][0])
        runners_up.append(standings[1][0])
        third_place_teams.append({
            "team": standings[2][0],
            "points": standings[2][1]["points"],
            "goal_difference": standings[2][1]["goal_difference"],
            "goals_for": standings[2][1]["goals_for"],
            "expected_points": standings[2][1]["expected_points"],
            "strength": standings[2][1]["strength"],
        })

    third_place_teams = sorted(
        third_place_teams,
        key=lambda item: (
            item["points"],
            item["goal_difference"],
            item["goals_for"],
            item["expected_points"],
            item["strength"],
            rng.random(),
        ),
        reverse=True,
    )

    qualified = winners + runners_up + [team_data["team"] for team_data in third_place_teams[:8]]

    for team in qualified:
        qualification_count[team] += 1

position_results = []
for team, counts in position_count.items():
    position_results.append({
        "team": team,
        "1st": counts[0] / N_SIMULATIONS,
        "2nd": counts[1] / N_SIMULATIONS,
        "3rd": counts[2] / N_SIMULATIONS,
        "4th": counts[3] / N_SIMULATIONS,
        "average_points": points_total[team] / N_SIMULATIONS,
        "qualify": qualification_count[team] / N_SIMULATIONS,
    })

position_df = pd.DataFrame(position_results).sort_values("qualify", ascending=False)
position_df.to_csv(OUTPUT_DIR / "team_position_probabilities.csv", index=False)

qualification_df = pd.DataFrame([
    {
        "team": team,
        "round_of_32_probability": count / N_SIMULATIONS,
    }
    for team, count in qualification_count.items()
]).sort_values("round_of_32_probability", ascending=False)
qualification_df.to_csv(OUTPUT_DIR / "round_of_32_probabilities.csv", index=False)

predicted_32 = qualification_df.head(32).reset_index(drop=True)
predicted_32.to_csv(OUTPUT_DIR / "predicted_round_of_32.csv", index=False)

qualified_teams = predicted_32["team"].tolist()
fixtures = []
for i in range(16):
    fixtures.append({
        "match": i + 1,
        "team_1": qualified_teams[i],
        "team_2": qualified_teams[31 - i],
    })

fixtures_df = pd.DataFrame(fixtures)
fixtures_df.to_csv(OUTPUT_DIR / "round_of_32_fixtures.csv", index=False)

standings_df = pd.DataFrame(sample_group_standings)
standings_df.to_csv(OUTPUT_DIR / "predicted_group_standings.csv", index=False)

print("\nTop 32 qualifiers\n")
print(predicted_32)
print("\nRound of 32 fixtures\n")
print(fixtures_df)
print("\nFiles generated:")
print(OUTPUT_DIR / "team_position_probabilities.csv")
print(OUTPUT_DIR / "round_of_32_probabilities.csv")
print(OUTPUT_DIR / "predicted_round_of_32.csv")
print(OUTPUT_DIR / "round_of_32_fixtures.csv")
print(OUTPUT_DIR / "predicted_group_standings.csv")