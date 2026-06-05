from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "match_training_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

DROP_COLS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
    "year",
    "winner",
    "home_group",
    "away_group",
    "target",
]

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
REVERSE_MAP = {0: -1, 1: 0, 2: 1}


def numeric_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in DROP_COLS if col in df.columns]
    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include="number")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

required = ["home_win_rate", "away_win_rate", "home_elo_rating", "away_elo_rating"]
df = df.dropna(subset=[col for col in required if col in df.columns])
df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(df["date"].dt.year)

train = df[df["year"] < 2022].copy()
test = df[df["year"] >= 2022].copy()

if train.empty or test.empty or len(train) < len(df) * 0.5:
    df = df.sort_values("date")
    split_index = int(len(df) * 0.8)
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

X_train = numeric_feature_frame(train)
y_train = train["target"].astype(int)
X_test = numeric_feature_frame(test).reindex(columns=X_train.columns, fill_value=0)
y_test = test["target"].astype(int)

print(f"Rows available for training: {len(df)}")
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
print(f"Features used: {len(X_train.columns)}")

rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\nRandom Forest")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.2%}")
print(classification_report(
    y_test,
    rf_preds,
    labels=[-1, 0, 1],
    target_names=["Away Win", "Draw", "Home Win"],
    zero_division=0,
))

y_train_xgb = y_train.map(LABEL_MAP)
y_test_xgb = y_test.map(LABEL_MAP)

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.035,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
)
xgb.fit(X_train, y_train_xgb)
xgb_preds_raw = xgb.predict(X_test)
xgb_preds = pd.Series(xgb_preds_raw).map(REVERSE_MAP).astype(int).values

print("\nXGBoost")
print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.2%}")
print(classification_report(
    y_test,
    xgb_preds,
    labels=[-1, 0, 1],
    target_names=["Away Win", "Draw", "Home Win"],
    zero_division=0,
))

MODEL_DIR.mkdir(exist_ok=True)

with open(MODEL_DIR / "rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open(MODEL_DIR / "xgb_model.pkl", "wb") as f:
    pickle.dump({"model": xgb, "label_map": LABEL_MAP, "reverse_map": REVERSE_MAP}, f)

feature_cols = list(X_train.columns)
with open(MODEL_DIR / "feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

importances = pd.DataFrame({
    "feature": feature_cols,
    "rf_importance": rf.feature_importances_,
}).sort_values("rf_importance", ascending=False)
importances.to_csv(MODEL_DIR / "feature_importances.csv", index=False)

print("\nModels saved to models folder")
print(f"Feature columns saved: {len(feature_cols)}")
print("Top 15 Random Forest features:")
print(importances.head(15).to_string(index=False))