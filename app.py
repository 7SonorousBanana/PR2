import requests
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

DEFAULT_LAT = 50.45     # Київ
DEFAULT_LON = 30.52

CSV_FILENAME = "weather_daily.csv"

# Функція завантаження даних

def fetch_weather_data(latitude, longitude, start_date, end_date):
    """
    Завантажує щоденні історичні метеодані з Open-Meteo.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "windspeed_10m_max",
        "windgusts_10m_max"
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": ",".join(daily_vars),
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data:
        raise ValueError("У відповіді API немає блоку 'daily'.")

    df = pd.DataFrame(data["daily"])

    if df.empty:
        raise ValueError("Отримано порожній датафрейм. Перевір дати або координати.")

    return df


# Підготовка даних

def prepare_features(df):
    """
    Підготовка ознак і цільової змінної.
    Ціль: precipitation_sum > 0 => 1, інакше 0
    """
    work_df = df.copy()

    work_df["time"] = pd.to_datetime(work_df["time"])
    work_df = work_df.sort_values("time").reset_index(drop=True)

    # Цільова змінна
    work_df["target"] = (work_df["precipitation_sum"] > 0).astype(int)

    # Календарні ознаки
    work_df["month"] = work_df["time"].dt.month
    work_df["day"] = work_df["time"].dt.day
    work_df["dayofyear"] = work_df["time"].dt.dayofyear
    work_df["weekday"] = work_df["time"].dt.weekday

    # Лагові ознаки (на основі попереднього дня)
    work_df["temp_mean_lag1"] = work_df["temperature_2m_mean"].shift(1)
    work_df["temp_max_lag1"] = work_df["temperature_2m_max"].shift(1)
    work_df["temp_min_lag1"] = work_df["temperature_2m_min"].shift(1)
    work_df["precip_sum_lag1"] = work_df["precipitation_sum"].shift(1)
    work_df["rain_sum_lag1"] = work_df["rain_sum"].shift(1)
    work_df["wind_max_lag1"] = work_df["windspeed_10m_max"].shift(1)
    work_df["gust_max_lag1"] = work_df["windgusts_10m_max"].shift(1)

    # Ознаки ковзного середнього
    work_df["temp_mean_roll3"] = work_df["temperature_2m_mean"].rolling(window=3).mean().shift(1)
    work_df["precip_roll3"] = work_df["precipitation_sum"].rolling(window=3).mean().shift(1)
    work_df["rain_roll3"] = work_df["rain_sum"].rolling(window=3).mean().shift(1)

    # Видаляємо рядки з NaN після shift/rolling
    work_df = work_df.dropna().reset_index(drop=True)

    feature_cols = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "rain_sum",
        "windspeed_10m_max",
        "windgusts_10m_max",
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


# Навчання моделі

def train_model(X, y, model_name="RandomForest"):
    """
    Навчання моделі та повернення метрик.
    """
    # Для часових даних краще не перемішувати
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Недостатньо даних для train/test. Збільш період.")

    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0)
    }

    return model, metrics, X_train, X_test, y_train, y_test, y_pred, y_proba



# Прогноз для вибраного рядка

def predict_for_row(model, row_features):
    pred = model.predict(row_features)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_features)[0][1]
    return pred, proba


# Сайдбар

st.sidebar.header("Параметри")

latitude = st.sidebar.number_input("Широта", value=DEFAULT_LAT, format="%.4f")
longitude = st.sidebar.number_input("Довгота", value=DEFAULT_LON, format="%.4f")

start_date = st.sidebar.date_input("Початкова дата", value=pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("Кінцева дата", value=pd.to_datetime("2025-12-31"))

model_name = st.sidebar.selectbox(
    "Оберіть модель",
    ["RandomForest", "LogisticRegression"]
)

# Стан сесії

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


# Блок 1. Завантаження даних

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


# Блок 2. Підготовка даних

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


# Блок 3. Навчання моделі

st.subheader("3. Навчання моделі та метрики")

if st.button("Навчити модель"):
    if st.session_state.X is None or st.session_state.y is None:
        st.warning("Спочатку отримай або завантаж дані.")
    else:
        try:
            model, metrics, X_train, X_test, y_train, y_test, y_pred, y_proba = train_model(
                st.session_state.X,
                st.session_state.y,
                model_name=model_name
            )

            st.session_state.model = model
            st.session_state.metrics = metrics

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            c2.metric("Precision", f"{metrics['precision']:.3f}")
            c3.metric("Recall", f"{metrics['recall']:.3f}")
            c4.metric("F1-score(Precision/Recall)", f"{metrics['f1']:.3f}")

            st.write("**Confusion Matrix:**")
            st.write(metrics["confusion_matrix"])

            st.write("**Classification Report:**")
            st.text(metrics["classification_report"])

            st.success("Модель успішно навчена.")
        except Exception as e:
            st.error(f"Помилка навчання моделі: {e}")

# Блок 4. Прогноз

st.subheader("4. Прогноз опадів")

if st.session_state.prepared_df is not None and st.session_state.model is not None:
    prepared_df = st.session_state.prepared_df
    feature_cols = st.session_state.feature_cols

    prediction_mode = st.radio(
        "Режим прогнозу",
        ["Для останнього доступного дня в датасеті", "Для конкретного дня з датасету"]
    )

    if prediction_mode == "Для останнього доступного дня в датасеті":
        selected_index = len(prepared_df) - 1
    else:
        date_options = prepared_df["time"].dt.strftime("%Y-%m-%d").tolist()
        selected_date = st.selectbox("Оберіть дату", date_options)
        selected_index = prepared_df[prepared_df["time"].dt.strftime("%Y-%m-%d") == selected_date].index[0]

    if st.button("Зробити прогноз"):
        row = prepared_df.loc[selected_index]
        row_features = pd.DataFrame([row[feature_cols]])

        pred, proba = predict_for_row(st.session_state.model, row_features)

        st.write(f"Дата: **{row['time'].date()}**")

        if pred == 1:
            st.success("Очікуються опади")
        else:
            st.info("Опадів не очікується")

        if proba is not None:
            st.write(f"Ймовірність опадів: **{proba:.2%}**")

        st.write("Ознаки для прогнозу:")
        st.dataframe(row_features, use_container_width=True)

else:
    st.info("Щоб зробити прогноз, спочатку завантаж дані та навчи модель.")
