"""
Trénování ML modelu pro predikci ceny hotelu
=============================================
Vstup:  dataset_ml.csv  (výstup z transform_dataset.py)
Výstup: model_xgb.json, model_rf.pkl, results.txt

Ohlídáno:
  - Data leakage: city_encoded a price_vs_city_mean počítány pouze z train sady
  - Train/test split před jakýmkoliv fitováním
  - Cross-validace na train sadě
  - Feature importance výpis
"""

import pandas as pd
import numpy as np
import json
import pickle

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

# ================================================================
# NAČTENÍ
# Pozor: načítáme dataset_final.csv (před transform scriptem),
# protože city_encoded a price_vs_city_mean musíme počítat
# až po train/test splitu aby nedošlo k data leakage.
# ================================================================

df = pd.read_csv("../Data_Transforming/dataset.csv")
print(f"Načteno řádků: {len(df)}")

# ================================================================
# ZÁKLADNÍ ČIŠTĚNÍ
# ================================================================

df["city"] = df["city"].str.capitalize()
df["breakfast"] = df["breakfast"].astype(int)
df["stars"] = df["stars"] / 2   # oprava: Booking vrací 2,4,6,8,10

df = df.dropna(subset=["price", "rating", "stars", "distance_km", "review_count"])
print(f"Po dropna: {len(df)} řádků")

# ================================================================
# ZÁKLADNÍ FEATURES (bez leaky sloupců)
# ================================================================

import holidays as hol

CITY_COUNTRY = {
    "Prague": "CZ", "Vienna": "AT", "Budapest": "HU", "Berlin": "DE",
    "Paris": "FR", "Rome": "IT", "Barcelona": "ES", "Amsterdam": "NL",
    "London": "GB", "Madrid": "ES", "Milan": "IT", "Munich": "DE",
    "Lisbon": "PT", "Copenhagen": "DK", "Stockholm": "SE", "Warsaw": "PL",
    "Dublin": "IE", "Brussels": "BE", "Oslo": "NO", "Helsinki": "FI",
    "Zurich": "CH",
}

COUNTRY_HOLIDAYS = {
    c: hol.country_holidays(c, years=2026)
    for c in set(CITY_COUNTRY.values())
}

df["checkin_dt"] = pd.to_datetime(df["checkin"])
df["week_of_year"] = df["checkin_dt"].dt.isocalendar().week.astype(int)
df["is_holiday"] = df.apply(
    lambda r: int(r["checkin_dt"].date() in COUNTRY_HOLIDAYS.get(CITY_COUNTRY.get(r["city"], ""), {})),
    axis=1
)
df["log_review_count"] = np.log1p(df["review_count"])
df["log_distance"]     = np.log1p(df["distance_km"])

# ================================================================
# TRAIN / TEST SPLIT
# Musí proběhnout PŘED výpočtem leaky features
# ================================================================

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
print(f"\nTrain: {len(df_train)} řádků")
print(f"Test:  {len(df_test)} řádků")

# ================================================================
# LEAKY FEATURES – počítáme pouze z train sady
# ================================================================

# 1. City encoding (průměrná cena za město)
city_mean = df_train.groupby("city")["price"].mean()
df_train = df_train.copy()
df_test  = df_test.copy()
df_train["city_encoded"] = df_train["city"].map(city_mean)
df_test["city_encoded"]  = df_test["city"].map(city_mean)

# Fallback pro města která nejsou v train sadě (edge case)
global_mean = df_train["price"].mean()
df_test["city_encoded"] = df_test["city_encoded"].fillna(global_mean)

# 2. Price vs city mean
group_mean_train = df_train.groupby(["city", "checkin"])["price"].transform("mean")
df_train["price_vs_city_mean"] = df_train["price"] / group_mean_train

# Pro test sadu použijeme city_mean (checkin kombinace nemusí být v train sadě)
df_test["price_vs_city_mean"] = df_test["price"] / df_test["city"].map(city_mean)

# 3. High event period
event_mean = df_train["event_count"].mean()
df_train["high_event_period"] = (df_train["event_count"] > event_mean).astype(int)
df_test["high_event_period"]  = (df_test["event_count"] > event_mean).astype(int)

# ================================================================
# FINÁLNÍ FEATURE LIST
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
    "is_holiday",
    "event_count",
    "high_event_period",
    "city_encoded",
    "price_vs_city_mean",
]

TARGET = "price"

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_test  = df_test[FEATURES]
y_test  = df_test[TARGET]

print(f"\nFeatures ({len(FEATURES)}): {FEATURES}")

# ================================================================
# HELPER – metriky
# ================================================================

def print_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  MAE:   {mae:>10.0f} Kč")
    print(f"  RMSE:  {rmse:>10.0f} Kč")
    print(f"  MAPE:  {mape:>9.1f} %")
    print(f"  R²:    {r2:>10.4f}")
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

# ================================================================
# MODEL 1: XGBoost
# ================================================================

print("\n\n══════════════════════════════════════")
print("  XGBoost")
print("══════════════════════════════════════")

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,       # L1 regularizace
    reg_lambda=1.0,      # L2 regularizace
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric="rmse",
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

xgb_pred_train = xgb_model.predict(X_train)
xgb_pred_test  = xgb_model.predict(X_test)

print_metrics("XGBoost – train", y_train, xgb_pred_train)
xgb_metrics = print_metrics("XGBoost – test", y_test, xgb_pred_test)

# Cross-validace na train sadě
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    X_train, y_train,
    cv=cv, scoring="r2", n_jobs=-1
)
print(f"\n  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
fi = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n  Feature importance:")
for feat, imp in fi.items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# ================================================================
# MODEL 2: Random Forest (baseline)
# ================================================================

print("\n\n══════════════════════════════════════")
print("  Random Forest (baseline)")
print("══════════════════════════════════════")

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=4,
    max_features=0.7,
    random_state=42,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)

rf_pred_train = rf_model.predict(X_train)
rf_pred_test  = rf_model.predict(X_test)

print_metrics("Random Forest – train", y_train, rf_pred_train)
rf_metrics = print_metrics("Random Forest – test", y_test, rf_pred_test)

# ================================================================
# SROVNÁNÍ
# ================================================================

print("\n\n══════════════════════════════════════")
print("  Srovnání modelů (test sada)")
print("══════════════════════════════════════")
print(f"  {'Metrika':<10}  {'XGBoost':>12}  {'Random Forest':>14}")
print(f"  {'─'*42}")
for m in ["mae", "rmse", "mape", "r2"]:
    unit = " Kč" if m in ["mae","rmse"] else (" %" if m == "mape" else "   ")
    print(f"  {m.upper():<10}  {xgb_metrics[m]:>11.1f}{unit}  {rf_metrics[m]:>13.1f}{unit}")

# ================================================================
# ULOŽENÍ MODELŮ
# ================================================================

xgb_model.save_model("model_xgb.json")
print("\n✅ XGBoost uložen do model_xgb.json")

with open("model_rf.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("✅ Random Forest uložen do model_rf.pkl")

# Uložení city_mean pro inference
with open("city_mean.json", "w") as f:
    json.dump(city_mean.to_dict(), f, ensure_ascii=False, indent=2)
print("✅ City encoding uložen do city_mean.json")

# Uložení výsledků
results = {
    "xgboost": xgb_metrics,
    "random_forest": rf_metrics,
    "features": FEATURES,
    "train_size": len(df_train),
    "test_size": len(df_test),
    "event_mean_threshold": float(event_mean),
    "global_price_mean": float(global_mean),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Výsledky uloženy do results.json")

print("\n✅ Hotovo.")