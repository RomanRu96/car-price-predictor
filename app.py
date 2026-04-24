import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np

# кастомизация страницы

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# --- 1. Загрузка модели и артефактов ---
@st.cache_resource
def load_resources():
    # Твой класс модели
    class CarPredictor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        def forward(self, x):
            return self.net(x)

    # Загружаем веса и препроцессоры
    model = CarPredictor(input_size=7)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    
    preprocessor = joblib.load("preprocessor.joblib")
    price_scaler = joblib.load("price_scaler.joblib")
    
    return model, preprocessor, price_scaler

# Загружаем один раз при старте
model, preprocessor, price_scaler = load_resources()

# --- 2. Интерфейс (UI) ---
st.title("🚗 Оценка стоимости авто")
st.caption("Прогноз цены на основе ML-модели (PyTorch)")

st.sidebar.header("Параметры авто")

# Ввод данных
mileage = st.sidebar.number_input("Пробег (км)", min_value=0, max_value=500000, value=50000, step=1000)
engine_power = st.sidebar.number_input("Мощность (л.с.)", min_value=50, max_value=500, value=150, step=10)
year = st.sidebar.number_input("Год выпуска", min_value=2000, max_value=2026, value=2020, step=1)
brand = st.sidebar.selectbox("Марка", ["Toyota", "BMW", "Mercedes", "Lada"])

if st.button(" Узнать цену"):
    try:
        with st.spinner("...Считаем прогноз..."):
        
            # 1. Собираем данные как DataFrame (как в predict.py)
            input_df = pd.DataFrame([{
                "mileage": mileage,
                "engine_power": engine_power,
                "year": year,
                "brand": brand
            }])

            # 2. Препроцессинг
            processed = preprocessor.transform(input_df)
            tensor_in = torch.FloatTensor(processed)

            # 3. Предсказание
            
            with torch.no_grad():
                pred_norm = model(tensor_in)

            # 4. Денормализация
            price = price_scaler.inverse_transform(pred_norm.numpy().reshape(-1, 1)).item()

            # 5. Вывод
            st.success(f"Прогнозируемая цена: **{price:,.0f} руб**")
            
    except Exception as e:
        st.error(f"Ошибка при расчете: {e}")


# Информация о модели в сайдбар
with st.sidebar:
    st.info("""
        **О модели**
        - Архитектура: MLP, 1 скрытый слой
        - Признаки: пробег, мощность, год, марка 
        - R2 Тест: ~98.7%
        - ! Прогноз на синтетических данных
        """)

# Добавление футера
st.markdown("---")
st.caption("Проект создан в рамках обучения ML | [GitHub](https://github.com/RomanRu96)")
