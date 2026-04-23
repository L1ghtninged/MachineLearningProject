import streamlit as st
from datetime import datetime, timedelta
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from src.model_training.hotel_model import HotelModel

BASE = Path(__file__).parent.parent / "models"


@st.cache_resource
def load_hotel_model():
    try:
        return HotelModel.from_results(
            model_path=str(BASE / "model_xgb.json"),
            city_mean_path=str(BASE / "city_mean.json"),
            results_path=str(BASE / "results.json"),
            events_path=str(BASE / "event_counts.csv"),
        )
    except Exception as e:
        st.error(f"Nepodařilo se načíst ML model: {e}")
        return None


model = load_hotel_model()


def get_model_prediction(city, stars, distance, breakfast, checkin_date, stay_length, rating, review_count):
    if model is None:
        return 2500

    month = checkin_date.month
    day_of_week = checkin_date.weekday()
    week_of_year = checkin_date.isocalendar()[1]
    event_count = model.get_event_count(city, checkin_date)

    pred_price = model.predict(
        city=city,
        rating=float(rating),
        stars=float(stars),
        breakfast=breakfast,
        distance_km=float(distance),
        review_count=int(review_count),
        month=month,
        day_of_week=day_of_week,
        stay_length=stay_length,
        week_of_year=week_of_year,
        event_count=event_count,
    )
    return int(pred_price)


st.set_page_config(page_title="Predikce cen hotelů", layout="wide")

st.title("Predikce ceny hotelů")
st.markdown("---")

MESTA = [
    "Prague", "Vienna", "Budapest", "Berlin", "Paris", "Rome",
    "Barcelona", "Amsterdam", "London", "Madrid",
    "Lisbon", "Copenhagen", "Stockholm", "Warsaw",
    "Dublin", "Brussels", "Oslo", "Helsinki", "Zurich",
]

today = datetime.now().date()
MAX_STAY = 5
END_OF_2026 = datetime(2026, 12, 31).date()

tab1, tab2, tab3 = st.tabs(["Základní predikce", "Srovnávač parametrů"])

with tab1:
    st.header("Odhad ceny pro konkrétní pobyt")
    col1, col2 = st.columns([1, 2])
    with col1:
        city = st.selectbox("Město:", MESTA, key="t1_city")
        checkin = st.date_input("Check-in:", value=today + timedelta(days=1), min_value=today, max_value=END_OF_2026,
                                key="t1_cin")
        checkout = st.date_input("Check-out:", value=today + timedelta(days=3), min_value=today + timedelta(days=1),
                                 max_value=END_OF_2026, key="t1_cout")
        stay_length = (checkout - checkin).days

        st.write("---")
        stars = st.slider("Počet hvězdiček:", 0, 5, 3, key="t1_stars")
        rating = st.slider("Hodnocení hotelu (0–10):", 0.0, 10.0, 8.0, 0.1, key="t1_rating")
        reviews = st.number_input("Počet recenzí:", min_value=0, max_value=100000, value=100, key="t1_reviews")
        distance = st.number_input("Vzdálenost od centra (km):", 0.0, 50.0, 2.0, 0.5, key="t1_dist")
        breakfast = st.checkbox("Snídaně v ceně", value=False, key="t1_bf")
        submit_btn = st.button("Spočítat cenu", type="primary", key="t1_btn")

    with col2:
        if submit_btn:
            if stay_length <= 0:
                st.error("Datum odjezdu musí být po datu příjezdu!")
            elif stay_length > MAX_STAY:
                st.error(f"Model podporuje pobyty o délce maximálně {MAX_STAY} nocí.")
            else:
                event_count = model.get_event_count(city, checkin) if model else 0
                celkova_cena = get_model_prediction(city, stars, distance, breakfast,
                                                    checkin, stay_length, rating, reviews)
                cena_za_noc = celkova_cena // stay_length

                st.metric(
                    label=f"Odhadovaná cena za pobyt ({stay_length} nocí)",
                    value=f"{celkova_cena:,} Kč".replace(",", " "),
                )
                st.info(f"Průměrná cena za noc: {cena_za_noc:,} Kč".replace(",", " "))

                if event_count > 0:
                    st.warning(f"V době pobytu se v okolí města koná {event_count} událostí – ceny mohou být vyšší.")
                else:
                    st.success("V době pobytu nejsou evidovány žádné větší události.")

with tab2:
    st.header("Analýza vlivu parametrů")
    st.write("Porovnejte dva různé typy ubytování ve stejném městě a ve stejném termínu.")

    col_top1, col_top2, col_top3 = st.columns(3)
    with col_top1:
        c_city = st.selectbox("Vyberte město pro analýzu:", MESTA, key="t2_city")
    with col_top2:
        c_checkin = st.date_input("Datum příjezdu:", value=today + timedelta(days=7), min_value=today,
                                  max_value=END_OF_2026, key="t2_date")
    with col_top3:
        c_stay_length = st.number_input("Počet nocí:", min_value=1, max_value=MAX_STAY, value=1, key="t2_stay")

    st.markdown("---")

    col_base, col_comp = st.columns(2)

    with col_base:
        st.subheader("1. Scénář (Základní)")
        b_stars = st.slider("Hvězdičky:", 0, 5, 3, key="t2_b_stars")
        b_rating = st.slider("Hodnocení:", 0.0, 10.0, 7.0, 0.1, key="t2_b_rating")
        b_reviews = st.number_input("Recenze:", 0, 100000, 100, key="t2_b_rev")
        b_dist = st.number_input("Vzdálenost (km):", 0.0, 20.0, 2.0, 0.5, key="t2_b_dist")

        cena_b = get_model_prediction(c_city, b_stars, b_dist, False, c_checkin, c_stay_length, b_rating, b_reviews)
        cena_b_per_night = cena_b // c_stay_length
        st.metric("Cena základního scénáře", f"{cena_b_per_night:,} Kč/noc".replace(",", " "))
        if c_stay_length > 1:
            st.caption(f"Celkem za pobyt: {cena_b:,} Kč".replace(",", " "))

    with col_comp:
        st.subheader("2. Scénář (Srovnávací)")
        s_stars = st.slider("Hvězdičky:", 0, 5, 4, key="t2_s_stars")
        s_rating = st.slider("Hodnocení:", 0.0, 10.0, 9.0, 0.1, key="t2_s_rating")
        s_reviews = st.number_input("Recenze:", 0, 100000, 500, key="t2_s_rev")
        s_dist = st.number_input("Vzdálenost (km):", 0.0, 20.0, 1.0, 0.5, key="t2_s_dist")

        cena_s = get_model_prediction(c_city, s_stars, s_dist, False, c_checkin, c_stay_length, s_rating, s_reviews)
        cena_s_per_night = cena_s // c_stay_length

        st.metric(
            "Cena srovnávacího scénáře",
            f"{cena_s_per_night:,} Kč/noc".replace(",", " "),
            delta=f"{cena_s_per_night - cena_b_per_night} Kč",
            delta_color="inverse",
        )
        if c_stay_length > 1:
            st.caption(f"Celkem za pobyt: {cena_s:,} Kč".replace(",", " "))

    st.info(
        f"Shrnutí: Rozdíl v ceně mezi scénáři činí {abs(cena_s_per_night - cena_b_per_night):,} Kč za noc.".replace(
            ",", " "))
