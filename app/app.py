import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model_training.hotel_model import HotelModel

BASE = Path(__file__).parent.parent / "model_training"

@st.cache_resource  # Tato dekorace zajistí, že se model načte jen jednou
def load_hotel_model():
    try:
        # Uprav cesty k souborům podle reality
        return HotelModel.from_results(
            model_path=str(BASE / "model_xgb.json"),
            city_mean_path=str(BASE / "city_mean.json"),
            results_path=str(BASE / "results.json")
        )
    except Exception as e:
        st.error(f"Nepodařilo se načíst ML model: {e}")
        return None


model = load_hotel_model()


# ==============================================================================
# 🛠️ POMOCNÁ FUNKCE PRO PREDIKCI
# ==============================================================================
def get_model_prediction(city, stars, distance, breakfast, checkin_date, stay_length):
    if model is None:
        return 2500  # Fallback cena, pokud model chybí

    # Dopočet parametrů, které model vyžaduje
    month = checkin_date.month
    day_of_week = checkin_date.weekday()
    week_of_year = checkin_date.isocalendar()[1]

    # Fixní hodnoty pro parametry, které uživatel v UI nezadává
    # (Při obhajobě vysvětli, že tyto hodnoty bereš jako "průměrný hotel")
    rating = 8.0
    review_count = 100
    event_count = 0  # Zde by mohlo být API na události

    pred_price = model.predict(
        city=city,
        rating=rating,
        stars=float(stars),
        breakfast=breakfast,
        distance_km=float(distance),
        review_count=review_count,
        month=month,
        day_of_week=day_of_week,
        stay_length=stay_length,
        week_of_year=week_of_year,
        event_count=event_count
    )
    return int(pred_price)


# ==============================================================================
# 🌟 NASTAVENÍ STRÁNKY A UI
# ==============================================================================
st.set_page_config(page_title="Prediktor cen hotelů", page_icon="🏨", layout="wide")

st.title("🏨 Chytrý hotelový asistent")
st.write("Využití strojového učení (XGBoost) pro analýzu a predikci cen ubytování.")
st.markdown("---")

seznam_mest = ["Prague", "Vienna", "Budapest", "Berlin", "Rome", "Paris"]
today = datetime.now().date()

tab1, tab2, tab3 = st.tabs(["🔮 Základní predikce", "⚖️ Srovnávač parametrů", "💡 Chytrá doporučení"])

# --- TAB 1: ZÁKLADNÍ PREDIKCE ---
with tab1:
    st.header("Odhad ceny pro konkrétní pobyt")
    col1, col2 = st.columns([1, 2])
    with col1:
        city = st.selectbox("Město:", seznam_mest, key="t1_city")
        checkin = st.date_input("Check-in:", value=today + timedelta(days=1), key="t1_cin")
        checkout = st.date_input("Check-out:", value=today + timedelta(days=3), key="t1_cout")
        stay_length = (checkout - checkin).days
        stars = st.slider("Počet hvězdiček:", 1, 5, 3, key="t1_stars")
        distance = st.number_input("Vzdálenost od centra (km):", 0.0, 50.0, 2.0, 0.5, key="t1_dist")
        breakfast = st.checkbox("Snídaně v ceně", value=False, key="t1_bf")
        submit_btn = st.button("🔮 Spočítat cenu", type="primary", key="t1_btn")

    with col2:
        if submit_btn:
            if stay_length <= 0:
                st.error("Datum odjezdu musí být po datu příjezdu!")
            else:
                cena_za_noc = get_model_prediction(city, stars, distance, breakfast, checkin, stay_length)
                celkova_cena = cena_za_noc * stay_length
                st.metric(label=f"Odhadovaná cena za pobyt ({stay_length} nocí)",
                          value=f"{celkova_cena:,} Kč".replace(",", " "))
                st.info(f"Průměrná cena za noc: {cena_za_noc:,} Kč".replace(",", " "))

# --- TAB 2: SROVNÁVAČ ---
with tab2:
    st.header("⚖️ Analýza vlivu parametrů")
    c_city = st.selectbox("Vyberte město pro analýzu:", seznam_mest, key="t2_city")
    col_base, col_comp = st.columns(2)
    with col_base:
        st.subheader("1. Scénář (Základní)")
        b_stars = st.slider("Hvězdičky:", 1, 5, 3, key="t2_b_stars")
        b_dist = st.number_input("Vzdálenost (km):", 0.0, 20.0, 2.0, 0.5, key="t2_b_dist")
        cena_b = get_model_prediction(c_city, b_stars, b_dist, False, today, 1)
        st.metric("Cena základního scénáře", f"{cena_b:,} Kč/noc".replace(",", " "))
    with col_comp:
        st.subheader("2. Scénář (Srovnávací)")
        s_stars = st.slider("Hvězdičky:", 1, 5, 4, key="t2_s_stars")
        s_dist = st.number_input("Vzdálenost (km):", 0.0, 20.0, 1.0, 0.5, key="t2_s_dist")
        cena_s = get_model_prediction(c_city, s_stars, s_dist, False, today, 1)
        st.metric("Cena srovnávacího scénáře", f"{cena_s:,} Kč/noc".replace(",", " "), delta=f"{cena_s - cena_b} Kč",
                  delta_color="inverse")

# --- TAB 3: DOPORUČENÍ ---
with tab3:
    st.header("💡 Jak ušetřit")
    c_dop = st.selectbox("Cílové město:", seznam_mest, key="t3_city")
    d_checkin = st.date_input("Plánovaný příjezd:", value=today + timedelta(days=7), key="t3_date")
    if st.button("🔍 Najít úspory", type="primary"):
        cena_aktualni = get_model_prediction(c_dop, 3, 1.0, False, d_checkin, 1)
        st.subheader(f"Aktuální odhad: {cena_aktualni:,} Kč / noc".replace(",", " "))

        # Simulace úspory posunem na pracovní den (pokud je víkend)
        if d_checkin.weekday() >= 4:
            next_tuesday = d_checkin + timedelta(days=(7 - d_checkin.weekday() + 1))
            cena_utery = get_model_prediction(c_dop, 3, 1.0, False, next_tuesday, 1)
            st.warning(
                f"📅 O víkendu je dráž. V úterý {next_tuesday.strftime('%d.%m.')} by byla cena cca {cena_utery:,} Kč (úspora {cena_aktualni - cena_utery} Kč).")

        # Úspora polohou
        cena_daleko = get_model_prediction(c_dop, 3, 5.0, False, d_checkin, 1)
        st.info(
            f"📍 Pokud zvolíte hotel 5 km od centra místo 1 km, ušetříte cca {cena_aktualni - cena_daleko} Kč za noc.")