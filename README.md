#  Predikce cen hotelů

Webová aplikace pro predikci cen hotelů v evropských městech. Využívá model strojového učení (XGBoost) natrénovaný na datech z Booking.com a Ticketmaster API.

---

## Požadavky

- Python 3.10 nebo novější

---

## Instalace

**1. Stáhnutí/klonování projektu**
```
git clone https://github.com/L1ghtninged/MachineLearningProject
```

**2. Nainstalování knihoven/závislostí**
```
pip install streamlit xgboost scikit-learn pandas numpy holidays playwright

```
pro streamlit případně:
```
python -m pip install streamlit
```


**3. Pro spuštění booking scraperu**
```
playwright install chromium
```
## Vytrénování modelu
Vytrénujte model v google colabu a pak soubory vložte do složky models
## Spuštění aplikace

```
streamlit run app/app.py
```
nebo
```
python -m streamlit run app/app.py
```

Aplikace se otevře v prohlížeči na adrese `http://localhost:8501`.

---

## Trénování modelu
Jupyter notebook: MachineLearningProject.ipynb


Train script: model_training/train.py
