import streamlit as st
from datetime import datetime, timedelta

# ==============================================================================
# 🌟 NASTAVENÍ STRÁNKY
# ==============================================================================
st.set_page_config(
    page_title="Prediktor cen hotelů",
    page_icon="🏨",
    layout="wide"
)


# ==============================================================================
# 🧠 POMOCNÉ FUNKCE (Simulace ML modelu)
# ==============================================================================
def mock_predict_price(city, stars, distance, breakfast, stay_length, is_weekend):
    """
    Simulace ML modelu. Až budeš mít reálný model,
    tuto funkci přepíšeš tak, aby volala tvůj model.
    """
    base_prices = {
        "Prague": 1500, "Vienna": 2200, "Budapest": 1200,
        "Berlin": 2500, "Rome": 2300, "Paris": 3000
    }

    price = base_prices.get(city, 1500)
    price += (stars * 500)  # Hvězdičky zvyšují cenu
    price -= (distance * 30)  # Vzdálenost snižuje cenu
    if breakfast:
        price += 300  # Snídaně zvyšuje cenu
    if is_weekend:
        price *= 1.25  # O víkendu je přirážka 25%

    return int(price * stay_length)


# ==============================================================================
# 🎨 UŽIVATELSKÉ ROZHRANÍ (UI)
# ==============================================================================
st.title("🏨 Chytrý hotelový asistent")
st.write("Využití strojového učení pro analýzu a predikci cen ubytování.")
st.markdown("---")

# Společná data pro celou aplikaci
seznam_mest = ["Prague", "Vienna", "Budapest", "Berlin", "Rome", "Paris"]
today = datetime.now()

# ====== VYTVOŘENÍ ZÁLOŽEK ======
tab1, tab2, tab3 = st.tabs([
    "🔮 Základní predikce",
    "⚖️ Srovnávač měst",
    "💡 Chytrá doporučení"
])

# ==============================================================================
# 🔮 ZÁLOŽKA 1: ZÁKLADNÍ PREDIKCE
# ==============================================================================
with tab1:
    st.header("Odhad ceny pro konkrétní pobyt")
    col1, col2 = st.columns([1, 2])

    with col1:
        city = st.selectbox("Město:", seznam_mest, key="t1_city")
        checkin = st.date_input("Check-in:", value=today + timedelta(days=1), key="t1_cin")
        checkout = st.date_input("Check-out:", value=today + timedelta(days=3), key="t1_cout")

        stay_length = (checkout - checkin).days
        is_weekend = 1 if checkin.weekday() >= 4 or checkout.weekday() >= 4 else 0

        stars = st.slider("Počet hvězdiček:", 1, 5, 3, key="t1_stars")
        distance = st.number_input("Vzdálenost od centra (km):", 0.0, 50.0, 2.0, 0.5, key="t1_dist")
        breakfast = st.checkbox("Snídaně v ceně", value=False, key="t1_bf")

        submit_btn = st.button("🔮 Spočítat cenu", type="primary", key="t1_btn")

    with col2:
        if submit_btn:
            if stay_length <= 0:
                st.error("Datum odjezdu musí být po datu příjezdu!")
            else:
                cena = mock_predict_price(city, stars, distance, breakfast, stay_length, is_weekend)
                st.metric(
                    label=f"Odhadovaná cena za {stay_length} noc(í) v městě {city}",
                    value=f"{cena:,} Kč".replace(",", " ")
                )
                st.success("Predikce byla úspěšně vygenerována modelem.")
        else:
            st.info("Zadejte parametry v levém sloupci a klikněte na tlačítko.")

# ==============================================================================
# ⚖️ ZÁLOŽKA 2: SROVNÁVAČ MĚST
# ==============================================================================
# Úprava v Záložce 2: Srovnávač parametrů
with tab2:
    st.header("⚖️ Analýza vlivu parametrů")
    st.write("Zjistěte, jak konkrétní změna parametru ovlivní výslednou cenu v daném městě.")

    c_city = st.selectbox("Vyberte město pro analýzu:", seznam_mest, key="t2_city")

    col_base, col_comp = st.columns(2)

    with col_base:
        st.subheader("1. Scénář (Základní)")
        b_stars = st.slider("Hvězdičky:", 1, 5, 3, key="t2_b_stars")
        b_dist = st.number_input("Vzdálenost (km):", 0.0, 20.0, 2.0, 0.5, key="t2_b_dist")
        b_bf = st.checkbox("Snídaně", value=False, key="t2_b_bf")

        cena_b = mock_predict_price(c_city, b_stars, b_dist, b_bf, 1, 0)
        st.metric("Cena základního scénáře", f"{cena_b:,} Kč/noc".replace(",", " "))

    with col_comp:
        st.subheader("2. Scénář (Srovnávací)")
        s_stars = st.slider("Hvězdičky:", 1, 5, 4, key="t2_s_stars")
        s_dist = st.number_input("Vzdálenost (km):", 0.0, 20.0, 1.0, 0.5, key="t2_s_dist")
        s_bf = st.checkbox("Snídaně", value=True, key="t2_s_bf")

        cena_s = mock_predict_price(c_city, s_stars, s_dist, s_bf, 1, 0)
        st.metric("Cena srovnávacího scénáře", f"{cena_s:,} Kč/noc".replace(",", " "),
                  delta=f"{cena_s - cena_b} Kč", delta_color="inverse")

    st.info(f"**Rozdíl mezi scénáři:** {abs(cena_s - cena_b):,} Kč za noc.".replace(",", " "))
# ==============================================================================
# 💡 ZÁLOŽKA 3: CHYTRÁ DOPORUČENÍ
# ==============================================================================
# Úprava v Záložce 3: Chytrá doporučení
with tab3:
    st.header("💡 Jak ušetřit v cílové destinaci")
    st.write("Model analyzuje vaše zadání a hledá úspory úpravou termínu nebo polohy.")

    c_dop = st.selectbox("Cílové město:", seznam_mest, key="t3_city")
    d_checkin = st.date_input("Plánovaný příjezd:", value=today + timedelta(days=7), key="t3_date")

    if st.button("🔍 Najít úspory", type="primary"):
        # Základní výpočet (3 hvězdy, 1 km od centra, víkend/týden dle data)
        is_wknd = 1 if d_checkin.weekday() >= 4 else 0
        cena_aktualni = mock_predict_price(c_dop, 3, 1.0, False, 1, is_wknd)

        st.subheader(f"Aktuální odhad: {cena_aktualni:,} Kč / noc".replace(",", " "))
        st.markdown("---")

        col_rec1, col_rec2 = st.columns(2)

        with col_rec1:
            st.write("### 📅 Termín")
            if is_wknd:
                cena_week = mock_predict_price(c_dop, 3, 1.0, False, 1, 0)
                st.warning(
                    f"Cestujete o víkendu. Pokud termín posunete na **střed týdne**, ušetříte cca **{cena_aktualni - cena_week} Kč** za noc.")
            else:
                st.success("Máte vybraný termín mimo víkendovou špičku. Výborně!")

        with col_rec2:
            st.write("### 📍 Poloha")
            # Simulujeme cenu pro 5km od centra
            cena_dist = mock_predict_price(c_dop, 3, 5.0, False, 1, is_wknd)
            uspora_dist = cena_aktualni - cena_dist
            if uspora_dist > 0:
                st.info(
                    f"Pokud zvolíte hotel **5 km od centra** (místo 1 km), můžete ušetřit průměrně **{uspora_dist} Kč** za noc.")