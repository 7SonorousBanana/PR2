import requests
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Налаштування сторінки
st.set_page_config(page_title="Прогноз опадів", page_icon="", layout="wide")
st.title("Міні-сервіс прогнозування опадів")
st.write("Завантаження щоденних метеоданих з Open-Meteo, навчання ML-моделі та прогноз опадів.")

# Константи
DEFAULT_LAT = 50.45
DEFAULT_LON = 30.52
CSV_FILENAME = "weather_daily.csv"

# Отримання історичних погодних даних (для навчання ML)
def fetch_weather_data(latitude, longitude, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"

    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "wind_speed_10m_max",
        "wind_gusts_10m_max"
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": ",".join(daily_vars),
        "timezone": "auto"
    }
    # HTTP запит до API
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data:
        raise ValueError("У відповіді API немає блоку 'daily'.")
    # перетворюємо отримані дані у DataFrame
    df = pd.DataFrame(data["daily"])

    if df.empty:
        raise ValueError("Отримано порожній датафрейм. Перевір дати або координати.")

    return df
# Отримання прогнозу погоди на 7 днів (Forecast API)
def fetch_forecast_data(latitude, longitude):
    """
    Отримує прогноз погоди на найближчі дні з Open-Meteo
    """

    url = "https://api.open-meteo.com/v1/forecast"

    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "wind_speed_10m_max",
        "wind_gusts_10m_max"
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(daily_vars),
        "forecast_days": 7,
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    data = response.json()

    # прогноз також переводимо у DataFrame
    df = pd.DataFrame(data["daily"])

    return df
# Підготовка даних для машинного навчання
def prepare_features(df):
    work_df = df.copy()
    
    # перетворюємо дату у формат datetime
    work_df["time"] = pd.to_datetime(work_df["time"])
    work_df = work_df.sort_values("time").reset_index(drop=True)
    
    # створення цільової змінної (чи були опади)
    work_df["target"] = (work_df["precipitation_sum"] > 0).astype(int)
    
    # календарні ознаки
    work_df["month"] = work_df["time"].dt.month
    work_df["day"] = work_df["time"].dt.day
    work_df["dayofyear"] = work_df["time"].dt.dayofyear
    work_df["weekday"] = work_df["time"].dt.weekday
    
    # значення попереднього дня
    work_df["temp_mean_lag1"] = work_df["temperature_2m_mean"].shift(1)
    work_df["temp_max_lag1"] = work_df["temperature_2m_max"].shift(1)
    work_df["temp_min_lag1"] = work_df["temperature_2m_min"].shift(1)
    work_df["precip_sum_lag1"] = work_df["precipitation_sum"].shift(1)
    work_df["rain_sum_lag1"] = work_df["rain_sum"].shift(1)
    work_df["wind_max_lag1"] = work_df["wind_speed_10m_max"].shift(1)
    work_df["gust_max_lag1"] = work_df["wind_gusts_10m_max"].shift(1)
    
    # середнє за 3 дні
    work_df["temp_mean_roll3"] = work_df["temperature_2m_mean"].rolling(window=3).mean().shift(1)
    work_df["precip_roll3"] = work_df["precipitation_sum"].rolling(window=3).mean().shift(1)
    work_df["rain_roll3"] = work_df["rain_sum"].rolling(window=3).mean().shift(1)
    
    work_df = work_df.dropna().reset_index(drop=True)

    # список ознак для моделі
    feature_cols = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "rain_sum",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "month",
        "day",
        "dayofyear",
        "weekday",
        "temp_mean_lag1",
        "temp_max_lag1",
        "temp_min_lag1",
        "precip_sum_lag1",
        "rain_sum_lag1",
        "wind_max_lag1",
        "gust_max_lag1",
        "temp_mean_roll3",
        "precip_roll3",
        "rain_roll3"
    ]

    X = work_df[feature_cols]
    y = work_df["target"]

    return work_df, X, y, feature_cols

# Навчання моделі Random Forest
def train_model(X, y):

    # розділення даних 80/20
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    # створення моделі
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # обчислення метрик
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0)
    }

    return model, metrics

# Функція прогнозу для одного рядка
def predict_for_row(model, row_features):
    pred = model.predict(row_features)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_features)[0][1]
    return pred, proba

# Параметри у sidebar
st.sidebar.header("Параметри")

latitude = st.sidebar.number_input("Широта", value=DEFAULT_LAT, format="%.4f")
longitude = st.sidebar.number_input("Довгота", value=DEFAULT_LON, format="%.4f")

start_date = st.sidebar.date_input("Початкова дата", value=pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("Кінцева дата", value=pd.to_datetime("2025-12-31"))

# Session state — зберігання стану програми

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "prepared_df" not in st.session_state:
    st.session_state.prepared_df = None
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "model" not in st.session_state:
    st.session_state.model = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
    
# отримання даних
st.subheader("1. Завантаження або отримання даних")

uploaded_file = st.file_uploader("Або завантаж CSV-файл", type=["csv"])

col1, col2 = st.columns(2)

with col1:
    if st.button("Отримати дані з Open-Meteo"):
        try:
            df = fetch_weather_data(latitude, longitude, start_date, end_date)
            df.to_csv(CSV_FILENAME, index=False, encoding="utf-8-sig")

            st.session_state.raw_df = df
            st.success(f"Дані успішно отримано та збережено у файл: {CSV_FILENAME}")
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Помилка під час завантаження даних: {e}")

with col2:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.raw_df = df
            st.success("CSV-файл успішно завантажено.")
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Помилка під час читання CSV: {e}")

if st.session_state.raw_df is not None:
    st.write(f"Кількість рядків у датасеті: **{len(st.session_state.raw_df)}**")

st.subheader("2. Підготовка даних")

if st.session_state.raw_df is not None:
    try:
        prepared_df, X, y, feature_cols = prepare_features(st.session_state.raw_df)

        st.session_state.prepared_df = prepared_df
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.feature_cols = feature_cols

        st.write("Після підготовки даних:")
        st.write(f"- Кількість рядків: **{len(prepared_df)}**")
        st.write(f"- Кількість ознак: **{len(feature_cols)}**")
        st.write(f"- Кількість днів з опадами: **{int(y.sum())}**")
        st.write(f"- Кількість днів без опадів: **{int((y == 0).sum())}**")

        st.dataframe(prepared_df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Помилка підготовки даних: {e}")

st.subheader("3. Навчання моделі та метрики")

if st.button("Навчити модель"):
    if st.session_state.X is None or st.session_state.y is None:
        st.warning("Спочатку отримай або завантаж дані.")
    else:
        try:
            model, metrics = train_model(
                st.session_state.X,
                st.session_state.y
            )

            st.session_state.model = model
            st.session_state.metrics = metrics

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            c2.metric("Precision", f"{metrics['precision']:.3f}")
            c3.metric("Recall", f"{metrics['recall']:.3f}")
            c4.metric("F1-score", f"{metrics['f1']:.3f}")

            st.write("**Confusion Matrix:**")
            st.write(metrics["confusion_matrix"])

            st.write("**Classification Report:**")
            st.text(metrics["classification_report"])

            st.success("Модель успішно навчена.")
        except Exception as e:
            st.error(f"Помилка навчання моделі: {e}")

st.subheader("4. Прогноз опадів на 7 днів")

if st.session_state.model is not None:

    if st.button("Отримати прогноз на 7 днів"):

        try:

            forecast_df = fetch_forecast_data(latitude, longitude)

            results = []

            for i in range(len(forecast_df)):

                row = forecast_df.iloc[i]
                date = pd.to_datetime(row["time"])

                features = pd.DataFrame([{
                    "temperature_2m_max": row["temperature_2m_max"],
                    "temperature_2m_min": row["temperature_2m_min"],
                    "temperature_2m_mean": row["temperature_2m_mean"],
                    "rain_sum": row["rain_sum"],
                    "wind_speed_10m_max": row["wind_speed_10m_max"],
                    "wind_gusts_10m_max": row["wind_gusts_10m_max"],
                    "month": date.month,
                    "day": date.day,
                    "dayofyear": date.dayofyear,
                    "weekday": date.weekday(),

                    "temp_mean_lag1": row["temperature_2m_mean"],
                    "temp_max_lag1": row["temperature_2m_max"],
                    "temp_min_lag1": row["temperature_2m_min"],
                    "precip_sum_lag1": row["precipitation_sum"],
                    "rain_sum_lag1": row["rain_sum"],
                    "wind_max_lag1": row["wind_speed_10m_max"],
                    "gust_max_lag1": row["wind_gusts_10m_max"],
                    "temp_mean_roll3": row["temperature_2m_mean"],
                    "precip_roll3": row["precipitation_sum"],
                    "rain_roll3": row["rain_sum"]
                }])

                pred, proba = predict_for_row(st.session_state.model, features)

                if pred == 1:
                    text = "Очікуються опади"
                else:
                    text = "Опадів не очікується"

                results.append({
                    "Дата": date.date(),
                    "Прогноз": text,
                    "Ймовірність опадів": f"{proba:.2%}"
                })

            result_df = pd.DataFrame(results)

            st.success("Прогноз на 7 днів:")

            st.dataframe(result_df, use_container_width=True)

        except Exception as e:
            st.error(f"Помилка прогнозу: {e}")

else:
    st.info("Спочатку потрібно навчити модель.")
