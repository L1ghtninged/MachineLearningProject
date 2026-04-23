import json
import numpy as np
import pandas as pd
import xgboost as xgb


class HotelModel:
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

    def __init__(self, model_path: str, city_mean_path: str):
        self._model = xgb.XGBRegressor()
        self._model.load_model(model_path)

        with open(city_mean_path, "r") as f:
            data = json.load(f)
            self._city_mean = {str(k).lower(): v for k, v in data.items()}

        self._global_mean_log: float | None = None
        self._events: pd.DataFrame | None = None

    @classmethod
    def from_results(cls, model_path: str, city_mean_path: str, results_path: str,
                     events_path: str | None = None) -> "HotelModel":
        instance = cls(model_path, city_mean_path)

        with open(results_path, "r") as f:
            results = json.load(f)

        instance._global_mean_log = results.get("global_mean_log")

        if events_path:
            df = pd.read_csv(events_path)
            df["city"] = df["city"].str.lower().str.strip()
            df["checkin"] = pd.to_datetime(df["checkin"]).dt.date
            instance._events = df

        return instance

    def get_event_count(self, city: str, checkin) -> int:
        if self._events is None:
            return 0

        if not isinstance(checkin, pd.Timestamp):
            checkin_dt = pd.to_datetime(checkin).date()
        else:
            checkin_dt = checkin.date()

        mask = (
                (self._events["city"] == city.lower().strip()) &
                (self._events["checkin"] == checkin_dt)
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

        city_key = city.lower().strip()

        log_distance = np.log1p(distance_km)
        log_review_count = np.log1p(review_count)
        is_weekend = int(day_of_week >= 4)

        has_stars = 1 if stars > 0 else 0

        city_encoded = self._city_mean.get(city_key, self._global_mean_log)

        city_dist_inter = city_encoded * log_distance

        features = [[
            rating,
            stars,
            has_stars,
            int(breakfast),
            log_distance,
            log_review_count,
            month,
            day_of_week,
            is_weekend,
            stay_length,
            week_of_year,
            event_count,
            city_encoded,
            city_dist_inter
        ]]

        pred_log = self._model.predict(features)[0]
        price_per_night = np.expm1(pred_log)

        total_price = price_per_night * stay_length

        return float(total_price)