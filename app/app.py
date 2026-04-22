import streamlit as st
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from model_training.hotel_model import HotelModel

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
    print(city+":"+str(event_count))

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
    "Barcelona", "Amsterdam", "London", "Madrid", "Milan",
    "Munich", "Lisbon", "Copenhagen", "Stockholm", "Warsaw",
    "Dublin", "Brussels", "Oslo", "Helsinki", "Zurich",
]

today = datetime.now().date()

tab1, tab2, tab3 = st.tabs(["Základní predikce", "Srovnávač parametrů", "Chytrá doporučení"])

with tab1:
    st.header("Odhad ceny pro konkrétní pobyt")
    col1, col2 = st.columns([1, 2])
    with col1:
        city = st.selectbox("Město:", MESTA, key="t1_city")
        checkin = st.date_input("Check-in:", value=today + timedelta(days=1), key="t1_cin")
        checkout = st.date_input("Check-out:", value=today + timedelta(days=3), key="t1_cout")
        stay_length = (checkout - checkin).days

        st.write("---")
        stars = st.slider("Počet hvězdiček:", 1, 5, 3, key="t1_stars")
        rating = st.slider("Hodnocení hotelu (0–10):", 0.0, 10.0, 8.0, 0.1, key="t1_rating")
        reviews = st.number_input("Počet recenzí:", min_value=0, max_value=100000, value=100, key="t1_reviews")
        distance = st.number_input("Vzdálenost od centra (km):", 0.0, 50.0, 2.0, 0.5, key="t1_dist")
        breakfast = st.checkbox("Snídaně v ceně", value=False, key="t1_bf")
        submit_btn = st.button("Spočítat cenu", type="primary", key="t1_btn")

    with col2:
        if submit_btn:
            if stay_length <= 0:
                st.error("Datum odjezdu musí být po datu příjezdu!")
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
        c_checkin = st.date_input("Datum příjezdu:", value=today + timedelta(days=7), key="t2_date")
    with col_top3:
        c_stay_length = st.number_input("Počet nocí:", min_value=1, max_value=30, value=1, key="t2_stay")

    st.markdown("---")

    t2_events = model.get_event_count(c_city, c_checkin) if model else 0
    if t2_events > 0:
        st.caption(f"V tomto termínu je v městě {c_city} hlášeno {t2_events} událostí.")

    col_base, col_comp = st.columns(2)

    with col_base:
        st.subheader("1. Scénář (Základní)")
        b_stars = st.slider("Hvězdičky:", 1, 5, 3, key="t2_b_stars")
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
        s_stars = st.slider("Hvězdičky:", 1, 5, 4, key="t2_s_stars")
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
        f"**Shrnutí:** Rozdíl v ceně mezi scénáři činí **{abs(cena_s_per_night - cena_b_per_night):,} Kč** za noc.".replace(
            ",", " "))
with tab3:
    st.header("Jak ušetřit")

    col_t3a, col_t3b = st.columns(2)
    with col_t3a:
        c_dop = st.selectbox("Cílové město:", MESTA, key="t3_city")
        d_checkin = st.date_input("Plánovaný příjezd:", value=today + timedelta(days=7), key="t3_date")
    with col_t3b:
        d_stars = st.slider("Kategorie hotelu (hvězdičky):", 1, 5, 3, key="t3_stars")
        d_stay = st.number_input("Počet nocí:", min_value=1, max_value=14, value=1, key="t3_stay")

    D_RATING = 8.0
    D_REVIEWS = 200
    D_DIST = 1.0

    if st.button("Najít úspory", type="primary"):

        cena_ref = get_model_prediction(c_dop, d_stars, D_DIST, False, d_checkin, d_stay, D_RATING, D_REVIEWS)
        event_count = model.get_event_count(c_dop, d_checkin) if model else 0

        st.subheader(f"Referenční cena: {cena_ref:,} Kč za pobyt".replace(",", " "))
        st.caption(f"{d_stars}★ · {d_stay} noc/í · {D_DIST} km od centra · hodnocení {D_RATING}")
        st.markdown("---")

        ceny_terminy = []
        for delta in range(-7, 8):
            kandidat = d_checkin + timedelta(days=delta)
            if kandidat < today:
                continue
            cena = get_model_prediction(c_dop, d_stars, D_DIST, False, kandidat, d_stay, D_RATING, D_REVIEWS)
            ceny_terminy.append((kandidat, cena))

        if ceny_terminy:
            nejlevnejsi_datum, nejlevnejsi_cena = min(ceny_terminy, key=lambda x: x[1])
            uspora_termin = cena_ref - nejlevnejsi_cena

            if nejlevnejsi_datum != d_checkin and uspora_termin > 0:
                st.success(
                    f"**Nejlevnější termín v okolí:** {nejlevnejsi_datum.strftime('%d.%m.')} "
                    f"({nejlevnejsi_datum.strftime('%A')}) · {nejlevnejsi_cena:,} Kč "
                    f"· ušetříte **{uspora_termin:,} Kč**".replace(",", " ")
                )
            else:
                st.success("Váš termín je v okolí ±7 dní již nejlevnější.")

        if d_checkin.weekday() >= 4:
            dnu_do_utery = (1 - d_checkin.weekday()) % 7 or 7
            next_tuesday = d_checkin + timedelta(days=dnu_do_utery)
            cena_utery = get_model_prediction(c_dop, d_stars, D_DIST, False, next_tuesday, d_stay, D_RATING, D_REVIEWS)
            uspora_den = cena_ref - cena_utery
            if uspora_den > 0:
                st.warning(
                    f"Příjezd o víkendu zdražuje. Nejbližší úterý "
                    f"({next_tuesday.strftime('%d.%m.')}) by vyšlo na {cena_utery:,} Kč "
                    f"· úspora **{uspora_den:,} Kč**".replace(",", " ")
                )

        vzdalenosti = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
        ceny_vzdal = [(d, get_model_prediction(c_dop, d_stars, d, False, d_checkin, d_stay, D_RATING, D_REVIEWS))
                      for d in vzdalenosti]

        cena_centrum = ceny_vzdal[0][1]
        doporucena_vzdal, doporucena_cena = min(
            [(d, c) for d, c in ceny_vzdal if d <= 3.0],
            key=lambda x: x[1]
        )
        uspora_vzdal = cena_centrum - doporucena_cena

        if doporucena_vzdal != 0.3 and uspora_vzdal > 0:
            st.info(
                f"Hotely **{doporucena_vzdal} km** od centra jsou výrazně levnější než přímo v centru "
                f"· úspora cca **{uspora_vzdal:,} Kč** za pobyt.".replace(",", " ")
            )

        ceny_hvezdicky = {
            s: get_model_prediction(c_dop, s, D_DIST, False, d_checkin, d_stay, D_RATING, D_REVIEWS)
            for s in range(1, 6)
        }

        levnejsi = {s: c for s, c in ceny_hvezdicky.items() if s < d_stars}
        if levnejsi:
            nejblizsi_stars = max(levnejsi.keys())
            cena_nizsi = levnejsi[nejblizsi_stars]
            uspora_hvezdicky = cena_ref - cena_nizsi
            if uspora_hvezdicky > 0:
                st.info(
                    f"Snížení kategorie z {d_stars}★ na {nejblizsi_stars}★ ušetří "
                    f"cca **{uspora_hvezdicky:,} Kč** za pobyt.".replace(",", " ")
                )

        if event_count > 0:
            st.warning(
                f"V termínu se v {c_dop} koná {event_count} událostí – "
                f"poptávka po ubytování může být vyšší než obvykle."
            )

        st.markdown("---")
        max_uspora = cena_ref - min(c for _, c in ceny_terminy) if ceny_terminy else 0
        max_uspora = max(max_uspora, uspora_vzdal, 0)
        if max_uspora > 0:
            st.metric("Maximální možná úspora kombinací tipů", f"{max_uspora:,} Kč".replace(",", " "))

