"""
ML model pro predikci ceny hotelu (opravená verze)
=================================================
✔ Bez data leakage
✔ Stabilnější (log(price))
✔ Lepší feature engineering
"""

import pandas as pd
import numpy as np
import json
import pickle

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

# ================================================================
# LOAD
# ================================================================

df = pd.read_csv("../Data_Transforming/dataset.csv")
print(f"Načteno řádků: {len(df)}")

# ================================================================
# CLEANING
# ================================================================

df["city"] = df["city"].str.capitalize()
df["breakfast"] = df["breakfast"].astype(int)
df["stars"] = df["stars"] / 2

# fallback pro event_count
if "event_count" not in df.columns:
    df["event_count"] = 0

df["event_count"] = df["event_count"].fillna(0)

df = df.dropna(subset=[
    "price", "rating", "stars", "distance_km", "review_count"
])

print(f"Po dropna: {len(df)}")

# ================================================================
# FEATURE ENGINEERING
# ================================================================

df["checkin_dt"] = pd.to_datetime(df["checkin"])

df["month"] = df["checkin_dt"].dt.month
df["week_of_year"] = df["checkin_dt"].dt.isocalendar().week.astype(int)

df["log_review_count"] = np.log1p(df["review_count"])
df["log_distance"] = np.log1p(df["distance_km"])
df["log_price"] = np.log1p(df["price"])  # 🎯 KLÍČOVÉ

# ================================================================
# TRAIN / TEST SPLIT
# ================================================================

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

print(f"Train: {len(df_train)}")
print(f"Test:  {len(df_test)}")

# ================================================================
# CITY ENCODING (SAFE)
# ================================================================

city_mean = df_train.groupby("city")["log_price"].mean()

global_mean = df_train["log_price"].mean()

df_train["city_encoded"] = df_train["city"].map(city_mean)
df_test["city_encoded"] = df_test["city"].map(city_mean).fillna(global_mean)

# ================================================================
# EVENT FEATURE
# ================================================================

event_mean = df_train["event_count"].mean()

df_train["high_event"] = (df_train["event_count"] > event_mean).astype(int)
df_test["high_event"] = (df_test["event_count"] > event_mean).astype(int)

# ================================================================
# FEATURES
# ================================================================

FEATURES = [
    "rating",
    "stars",
    "breakfast",
    "log_distance",
    "log_review_count",
    "month",
    "day_of_week",
    "is_weekend",
    "stay_length",
    "week_of_year",
    "event_count",
    "high_event",
    "city_encoded"
]

TARGET = "log_price"

X_train = df_train[FEATURES]
y_train = df_train[TARGET]

X_test = df_test[FEATURES]
y_test = df_test[TARGET]

print("\nFeatures:", FEATURES)

# ================================================================
# METRICS
# ================================================================

def evaluate(name, y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"MAE:  {mae:.0f} Kč")
    print(f"RMSE: {rmse:.0f} Kč")
    print(f"R2:   {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}

# ================================================================
# XGBOOST
# ================================================================

print("\n===== XGBoost =====")

xgb_model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

pred_train = xgb_model.predict(X_train)
pred_test = xgb_model.predict(X_test)

metrics_xgb = evaluate("XGB Train", y_train, pred_train)
metrics_xgb_test = evaluate("XGB Test", y_test, pred_test)

# ================================================================
# RANDOM FOREST
# ================================================================

print("\n===== Random Forest =====")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

metrics_rf = evaluate("RF Test", y_test, rf_pred)

# ================================================================
# FEATURE IMPORTANCE
# ================================================================

fi = pd.Series(xgb_model.feature_importances_, index=FEATURES)
fi = fi.sort_values(ascending=False)

print("\nFeature importance:")
for f, v in fi.items():
    print(f"{f:<20} {v:.4f}")

# ================================================================
# SAVE
# ================================================================

xgb_model.save_model("model_xgb.json")

with open("model_rf.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("city_mean.json", "w") as f:
    json.dump(city_mean.to_dict(), f)

print("\n✅ Modely uloženy")
results = {
    "xgboost": metrics_xgb_test,
    "random_forest": metrics_rf,
    "features": FEATURES,
    "train_size": len(df_train),
    "test_size": len(df_test),
    "event_mean_threshold": float(event_mean),
    "global_price_mean_log": float(global_mean),
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Výsledky uloženy do results.json")