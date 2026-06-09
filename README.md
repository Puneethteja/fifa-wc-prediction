
# 🏆 2026 FIFA World Cup Predictive Simulator

An end-to-end machine learning pipeline and Monte Carlo simulation engine built to engineer international football features, predict match-level outcomes, and simulate the entire 2026 FIFA World Cup group stage[cite: 1, 6].

---

## 📋 Features Overview

* **Custom Rating Engines:** Computes tournament-weighted historical FIFA-style summation ratings and tracks Elo ratings dynamically across decades of international matches.
* **Feature Engineering:** Extracts rolling team momentum matrices, including 5, 10, and 20-match form, attacking/defending metrics, clean sheet percentages, and neutral ground performance records.
* **Ensemble Machine Learning:** Employs a blended predictive approach utilizing calibrated Random Forest and XGBoost classifiers to determine match results[cite: 1, 2].
* **Stochastic Simulation:** Runs 10,000 Monte Carlo iterations tracking group tables, tie-breakers, top third-place qualifiers, and generating bracket fixtures for the Round of 32.

---

## 🛠️ Installation & Setup

Ensure you have Python installed. Clone this repository to your local environment and install the required dependencies.

```bash
# Clone the repository
git clone [https://github.com/dishas75/fifa-wc-prediction](https://github.com/dishas75/fifa-wc-prediction)
cd fifa-wc-prediction

# Install required dependencies
pip install -r requirements.txt
```
### 📂 Repository Structure

The repository is organized into distinct functional directories for data, scripts, modeling, and output artifacts:

```text
.
├── data/                                    # Source data and generated pipeline features
│   ├── elo_ratings.csv
│   ├── eloratings.csv
│   ├── fifa_sum_ratings.csv
│   ├── final_wc2026_features_with_groups.csv
│   ├── historical_wc2026_matches.csv
│   ├── match_training_dataset.csv
│   ├── team_encoding_map.csv
│   ├── training_features.csv
│   ├── worldcup_2026_group_stage_probabilities.csv
│   ├── worldcup2026_fixtures.csv
│   └── worldcup2026_groups.csv
├── docs/                                    # Documentation files
├── modeling/                                # Prediction and machine learning models
│   ├── models/                              # Serialized model checkpoints
│   ├── predict_match.py
│   └── train_model.py
├── models/                                  # Alternate models checkpoint directory
├── outputs/                                 # Target folder for simulation results
│   ├── predicted_group_stage_probabilities.csv
│   ├── predicted_group_standings_decimal.csv
│   ├── predicted_group_standings_integer.csv
│   ├── predicted_group_standings.csv
│   ├── round_of_32_fixtures.csv
│   └── round_of_32_probabilities.csv
├── scripts/                                 # Data prep and feature extraction scripts
│   ├── 01_team_feature_engineering.py
│   ├── 02_fifa_sum_rating.py
│   └── 03_match_dataset_generation.py
├── simulation/                              # Tournament simulator script
│   └── monte_carlo_simulation.py
├── feature_importances.csv
├── README.md
└── requirements.txt
```
### 🚀 Execution Pipeline

The simulator must be executed sequentially to parse the raw data into final tournament probabilities.

#### Step 1: Calculate Custom FIFA Ratings
Processes historical match data dating back to 1990 to calculate tournament-weighted summation ratings for all international teams.
```bash
python scripts/02_fifa_sum_rating.py
```
#### Step 2: Build Team Features Matrix
Aggregates historical matches, custom FIFA ratings, and Elo data to build an exhaustive performance profile for every participating World Cup team.
```bash
python scripts/01_team_feature_engineering.py
```
#### Step 3: Compile Match Training Dataset
Merges team-specific matrices into historical head-to-head records, creating delta/difference features (e.g., Elo differences, expected goal edges) for model training.
```bash
python scripts/03_match_dataset_generation.py
```
#### Step 4: Train Machine Learning Models
Trains, evaluates, and exports the Random Forest and XGBoost classifiers using a historical time-split.
```bash
python modeling/train_model.py
```
#### Step 5: Predict Specific Fixtures
Applies the saved model weights against the official 2026 World Cup group fixtures to output match-level Win-Draw-Loss probabilities.
```bash
python modeling/predict_match.py
```
#### Step 6: Execute Tournament Simulation
Runs 10,000 Monte Carlo simulated iterations utilizing a Poisson distribution of scores derived from the match prediction probabilities to output group outcomes and bracket placements.
```bash
python simulation/monte_carlo_simulation.py
```
### 📊 Core Data Pipeline Inputs & Outputs

The following files act as the baseline communication layer between processing steps.

#### Critical Input Files (Expected in `data/`)
* `all_matches.csv`: Global historical football data used for global rating calculations.
* `historical_wc2026_matches.csv`: Subset of historical match performance metrics strictly mapping context for teams participating in the 2026 cycle.
* `worldcup2026_groups.csv`: Definitive team group stage placements.
* `worldcup2026_fixtures.csv`: The official schedule/pairings of the group stage match-ups.

#### Produced Artifacts & Outputs

| Generated File | Path Location | Description |
| :--- | :--- | :--- |
| **FIFA Ratings File** | `data/fifa_sum_ratings.csv` | Calculated historical strength metrics per team. |
| **Team Feature Matrix** | `data/training_features.csv` | Final engineered feature profile array per team. |
| **Blended Classifiers** | `modeling/models/` | Serialized model files (`rf_model.pkl`, `xgb_model.pkl`, `feature_cols.pkl`). |
| **Match Probabilities** | `data/worldcup_2026_group_stage_probabilities.csv` | Raw expected values and calculated outcomes per fixture. |
| **Simulation Tables** | `outputs/predicted_group_standings_integer.csv` | Clean, integer-rounded expected final group rankings. |
| **Simulation Probabilities** | `outputs/round_of_32_probabilities.csv` | Percentage metrics indicating likelihood of tournament progression. |
| **Bracket Fixtures** | `outputs/round_of_32_fixtures.csv` | Automatically generated bracket mapping for the Round of 32. |
