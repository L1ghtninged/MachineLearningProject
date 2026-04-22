from hotel_model import HotelModel

model = HotelModel.from_results(model_path="../models/model_xgb.json", city_mean_path="../models/city_mean.json", results_path="../models/results.json")

prediction = model.predict(
    city="Prague",
    rating=8,
    stars=5,
    breakfast=True,
    distance_km=20,
    review_count=100,
    month=11,
    day_of_week=4,
    stay_length=2,
    week_of_year=7,
    event_count=7
)

print(prediction)