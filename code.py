import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

# =====================================
# Load Data
# =====================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# =====================================
# Missing Values
# =====================================

train["RoadType"] = train["RoadType"].fillna("Unknown")
test["RoadType"] = test["RoadType"].fillna("Unknown")

train["Weather"] = train["Weather"].fillna("Unknown")
test["Weather"] = test["Weather"].fillna("Unknown")

temp_med = train["Temperature"].median()

train["Temperature"] = train["Temperature"].fillna(temp_med)
test["Temperature"] = test["Temperature"].fillna(temp_med)

# =====================================
# Feature Engineering
# =====================================

for df in [train, test]:

    t = pd.to_datetime(df["timestamp"], format="%H:%M")

    df["hour"] = t.dt.hour
    df["minute"] = t.dt.minute

    df["slot"] = df["hour"] * 4 + (df["minute"] // 15)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    df["geo1"] = df["geohash"].astype(str).str[:1]
    df["geo2"] = df["geohash"].astype(str).str[:2]
    df["geo3"] = df["geohash"].astype(str).str[:3]

    df["peak_morning"] = (
        (df["hour"] >= 7) &
        (df["hour"] <= 10)
    ).astype(int)

    df["peak_evening"] = (
        (df["hour"] >= 17) &
        (df["hour"] <= 20)
    ).astype(int)

    df["temp_lane"] = (
        df["Temperature"] *
        df["NumberofLanes"]
    )

# Drop original timestamp
train.drop(columns=["timestamp"], inplace=True)
test.drop(columns=["timestamp"], inplace=True)

# =====================================
# Target
# =====================================

y = train["demand"]

X = train.drop(columns=["demand"])

test_ids = test["Index"]

# =====================================
# Categorical Columns
# =====================================

cat_cols = [
    "geohash",
    "geo1",
    "geo2",
    "geo3",
    "day",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather"
]

for c in cat_cols:
    X[c] = X[c].astype(str)
    test[c] = test[c].astype(str)

# =====================================
# 5 Fold Ensemble
# =====================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

test_preds = np.zeros(len(test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):

    print(f"\nFold {fold + 1}")

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    model = CatBoostRegressor(
        iterations=2500,
        depth=8,
        learning_rate=0.02,
        loss_function="RMSE",
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    test_preds += model.predict(test) / 5

# =====================================
# Submission
# =====================================

submission = pd.DataFrame({
    "Index": test_ids,
    "demand": test_preds
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nsubmission.csv created")
print(submission.head())
