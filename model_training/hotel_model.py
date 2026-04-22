import json
import numpy as np
import pandas as pd
import xgboost as xgb


class HotelModel:

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

    def __init__(self, model_path: str, city_mean_path: str):
        self._model = xgb.XGBRegressor()
        self._model.load_model(model_path)

        with open(city_mean_path, "r") as f:
            self._city_mean: dict = json.load(f)

        self._event_mean: float | None = None
        self._global_price_mean_log: float | None = None
        self._events: pd.DataFrame | None = None

    @classmethod
    def from_results(cls, model_path: str, city_mean_path: str, results_path: str,
                     events_path: str | None = None) -> "HotelModel":
        instance = cls(model_path, city_mean_path)

        with open(results_path, "r") as f:
            results = json.load(f)

        instance._event_mean = results["event_mean_threshold"]
        instance._global_price_mean_log = results["global_price_mean_log"]

        if events_path:
            df = pd.read_csv(events_path)
            df["city"] = df["city"].str.capitalize()
            df["checkin"] = pd.to_datetime(df["checkin"]).dt.date
            instance._events = df

        return instance

    def get_event_count(self, city: str, checkin) -> int:
        if self._events is None:
            return 0

        if not isinstance(checkin, type(pd.Timestamp("2026-01-01").date())):
            checkin = pd.to_datetime(checkin).date()

        mask = (
            (self._events["city"] == city.capitalize()) &
            (self._events["checkin"] == checkin)
        )
        rows = self._events.loc[mask, "event_count"]

        return int(rows.iloc[0]) if not rows.empty else 0

    def predict(
        self,
        city: str,
        rating: float,
        stars: float,
        breakfast: bool,
        distance_km: float,
        review_count: int,
        month: int,
        day_of_week: int,
        stay_length: int,
        week_of_year: int,
        event_count: int,
    ) -> float:

        log_distance     = np.log1p(distance_km)
        log_review_count = np.log1p(review_count)
        is_weekend       = int(day_of_week >= 5)

        city_encoded = self._city_mean.get(city, self._global_price_mean_log)

        high_event = int(event_count > self._event_mean) if self._event_mean else 0

        features = [[
            rating,
            stars,
            int(breakfast),
            log_distance,
            log_review_count,
            month,
            day_of_week,
            is_weekend,
            stay_length,
            week_of_year,
            event_count,
            high_event,
            city_encoded
        ]]

        pred_log   = self._model.predict(features)[0]
        pred_price = np.expm1(pred_log)

        return float(pred_price)