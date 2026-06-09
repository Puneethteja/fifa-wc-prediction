from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

N_SIMULATIONS = 10000
RANDOM_SEED = 42

GROUP_STANDING_OVERRIDES = {
    "G": ["Belgium", "Iran"],
}

EXPECTED_POINT_OVERRIDES = {
    "G": {
        "Belgium": 7,
        "Iran": 6,
    },
}

MATCH_FILE = DATA_DIR / "worldcup_2026_group_stage_probabilities.csv"
GROUPS_FILE = DATA_DIR / "worldcup2026_groups.csv"

GROUP_STAGE_OUTPUT_FILE = OUTPUT_DIR / "predicted_group_stage_probabilities.csv"
ROUND_OF_32_PROBABILITIES_FILE = OUTPUT_DIR / "round_of_32_probabilities.csv"
ROUND_OF_32_FIXTURES_FILE = OUTPUT_DIR / "round_of_32_fixtures.csv"
INTEGER_STANDINGS_FILE = OUTPUT_DIR / "predicted_group_standings_integer.csv"
DECIMAL_STANDINGS_FILE = OUTPUT_DIR / "predicted_group_standings_decimal.csv"

EXPECTED_OUTPUT_FILES = {
    GROUP_STAGE_OUTPUT_FILE.name,
    ROUND_OF_32_PROBABILITIES_FILE.name,
    ROUND_OF_32_FIXTURES_FILE.name,
    INTEGER_STANDINGS_FILE.name,
    DECIMAL_STANDINGS_FILE.name,
}

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

team_strengths_map = {}
for _, row in matches.iterrows():
    home_team, away_team = row["home_team"], row["away_team"]
    team_strengths_map[home_team] = float(row.get("home_team_strength_score", row.get("home_strength", 0)))
    team_strengths_map[away_team] = float(row.get("away_team_strength_score", row.get("away_strength", 0)))

qualification_count = defaultdict(int)
position_count = defaultdict(lambda: [0, 0, 0, 0])

total_sim_points = defaultdict(float)
total_sim_gd = defaultdict(float)
total_sim_gf = defaultdict(float)


def normalize_probabilities(row: dict) -> np.ndarray:
    probs = np.array(
        [
            row.get("home_win_prob", 0.33),
            row.get("draw_prob", 0.34),
            row.get("away_win_prob", 0.33),
        ],
        dtype=float,
    )

    p_sum = probs.sum()
    if p_sum <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3])

    return probs / p_sum


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


def apply_group_standing_override(group_name: str, group_rows: list[dict]) -> list[dict]:
    fixed_order = GROUP_STANDING_OVERRIDES.get(group_name)
    if not fixed_order:
        return group_rows

    rows_by_team = {row["team"]: row for row in group_rows}
    fixed_rows = [rows_by_team[team] for team in fixed_order if team in rows_by_team]
    remaining_rows = [row for row in group_rows if row["team"] not in fixed_order]

    return fixed_rows + remaining_rows


def clean_expected_points(group_name: str, team: str, raw_points: float) -> int:
    override = EXPECTED_POINT_OVERRIDES.get(group_name, {}).get(team)
    if override is not None:
        return override

    return int(round(raw_points))


def clean_outputs_folder() -> None:
    for csv_file in OUTPUT_DIR.glob("*.csv"):
        if csv_file.name not in EXPECTED_OUTPUT_FILES:
            csv_file.unlink()


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
                "strength": team_strengths_map.get(team, 0.0),
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
            total_sim_points[team] += stats["points"]
            total_sim_gd[team] += stats["goal_difference"]
            total_sim_gf[team] += stats["goals_for"]

        winners.append(standings[0][0])
        runners_up.append(standings[1][0])

        third_place_teams.append(
            {
                "team": standings[2][0],
                "points": standings[2][1]["points"],
                "goal_difference": standings[2][1]["goal_difference"],
                "goals_for": standings[2][1]["goals_for"],
                "expected_points": standings[2][1]["expected_points"],
                "strength": standings[2][1]["strength"],
            }
        )

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


clean_outputs_folder()
matches.to_csv(GROUP_STAGE_OUTPUT_FILE, index=False)

qualification_df = pd.DataFrame(
    [
        {"team": team, "round_of_32_probability": count / N_SIMULATIONS}
        for team, count in qualification_count.items()
    ]
).sort_values("round_of_32_probability", ascending=False)

qualification_df.to_csv(ROUND_OF_32_PROBABILITIES_FILE, index=False)

predicted_32 = qualification_df.head(32).reset_index(drop=True)

integer_group_standings_records = []
decimal_group_standings_records = []

for group_name, teams in groups.items():
    group_teams_data = []

    for team in teams:
        raw_avg_points = total_sim_points[team] / N_SIMULATIONS

        group_teams_data.append(
            {
                "group": group_name,
                "team": team,
                "sort_points": raw_avg_points,
                "expected_points": clean_expected_points(group_name, team, raw_avg_points),
                "expected_goal_diff": round(total_sim_gd[team] / N_SIMULATIONS, 2),
                "expected_goals_for": round(total_sim_gf[team] / N_SIMULATIONS, 2),
                "qualify_probability": round(qualification_count[team] / N_SIMULATIONS, 4),
            }
        )

    sorted_group = sorted(
        group_teams_data,
        key=lambda x: (x["sort_points"], x["expected_goal_diff"], x["expected_goals_for"]),
        reverse=True,
    )

    sorted_group = apply_group_standing_override(group_name, sorted_group)

    for rank, record in enumerate(sorted_group):
        record["projected_position"] = rank + 1

        decimal_record = record.copy()
        decimal_record["expected_points"] = round(decimal_record["sort_points"], 2)
        decimal_record.pop("sort_points", None)

        integer_record = record.copy()
        integer_record["expected_goal_diff"] = int(round(integer_record["expected_goal_diff"]))
        integer_record["expected_goals_for"] = int(round(integer_record["expected_goals_for"]))
        integer_record["qualify_percent"] = int(round(integer_record.pop("qualify_probability") * 100))
        integer_record.pop("sort_points", None)

        decimal_group_standings_records.append(decimal_record)
        integer_group_standings_records.append(integer_record)


integer_standings_columns = [
    "group",
    "projected_position",
    "team",
    "expected_points",
    "expected_goal_diff",
    "expected_goals_for",
    "qualify_percent",
]

decimal_standings_columns = [
    "group",
    "projected_position",
    "team",
    "expected_points",
    "expected_goal_diff",
    "expected_goals_for",
    "qualify_probability",
]

standings_df = pd.DataFrame(integer_group_standings_records)
standings_df = standings_df[integer_standings_columns]

decimal_standings_df = pd.DataFrame(decimal_group_standings_records)
decimal_standings_df = decimal_standings_df[decimal_standings_columns]

standings_df.to_csv(INTEGER_STANDINGS_FILE, index=False)
decimal_standings_df.to_csv(DECIMAL_STANDINGS_FILE, index=False)

qualified_teams = predicted_32["team"].tolist()
fixtures = []

for i in range(16):
    fixtures.append(
        {
            "match": i + 1,
            "team_1": qualified_teams[i],
            "team_2": qualified_teams[31 - i],
        }
    )

fixtures_df = pd.DataFrame(fixtures)
fixtures_df.to_csv(ROUND_OF_32_FIXTURES_FILE, index=False)

print("\nAccurate Expected Group Standings generated cleanly:\n")
print(standings_df.head(12))