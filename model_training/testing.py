from hotel_model import HotelModel

model = HotelModel.from_results(model_path="model_xgb.json", city_mean_path="city_mean.json", results_path="results.json")

model.predict(
    city="Prague",
    rating=
    stars=5,
    breakfast=True,



)