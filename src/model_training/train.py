import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

DATA_PATH = "../Data_Transforming/dataset.csv"
MODEL_DIR = "../models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

df = pd.read_csv(DATA_PATH)
print(f"Načteno řádků: {len(df)}")

df["city"] = df["city"].str.lower().str.strip()

df["price_per_night"] = df["price"] / df["stay_length"]

df = df[df["price_per_night"] < 15000].copy()
df = df[df["price_per_night"] > 200].copy()

df["stars"] = df["stars"] / 2
df["has_stars"] = df["stars"].notna().astype(int)
df["stars"] = df["stars"].fillna(0)

df["breakfast"] = df["breakfast"].astype(int)
df["event_count"] = df["event_count"].fillna(0)

df = df.dropna(subset=["price", "rating", "distance_km", "review_count"])

df["checkin_dt"] = pd.to_datetime(df["checkin"])
df["month"] = df["checkin_dt"].dt.month
df["week_of_year"] = df["checkin_dt"].dt.isocalendar().week.astype(int)

df["log_review_count"] = np.log1p(df["review_count"])
df["log_distance"] = np.log1p(df["distance_km"])
df["log_price"] = np.log1p(df["price_per_night"])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

city_mean = df_train.groupby("city")["log_price"].mean()
global_mean = df_train["log_price"].mean()

df_train["city_encoded"] = df_train["city"].map(city_mean)
df_test["city_encoded"] = df_test["city"].map(city_mean).fillna(global_mean)


FEATURES = [
    "rating",
    "stars",
    "has_stars",
    "breakfast",
    "log_distance",
    "log_review_count",
    "month",
    "day_of_week",
    "is_weekend",
    "stay_length",
    "week_of_year",
    "event_count",
    "city_encoded",
]

TARGET = "log_price"

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

def evaluate(name, y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {name} ---")
    print(f"MAE (průměrná chyba): {mae:.0f} Kč za noc")
    print(f"R2 Score:             {r2:.4f}")

    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

xgb_model = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
pred_test_xgb = xgb_model.predict(X_test)
metrics_xgb = evaluate("XGBoost Test", y_test, pred_test_xgb)

fi = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nNejdůležitější faktory:")
print(fi.head(20))

xgb_model.save_model(os.path.join(MODEL_DIR, "model_xgb.json"))
with open(os.path.join(MODEL_DIR, "city_mean.json"), "w") as f:
    json.dump(city_mean.to_dict(), f)

print(f"\nUloženo. MAE: {metrics_xgb['mae']:.0f} Kč")