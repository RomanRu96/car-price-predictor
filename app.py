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

# --- 2. Интерфейс (UI) — Адаптивный для ПК и мобильных ---
st.title("🚗 Оценка стоимости авто")
st.caption("Прогноз цены на основе ML-модели (PyTorch)")

# Заголовок вынесен НАД колонками → сетка не ломается
st.markdown("### 📋 Параметры автомобиля")

col1, col2 = st.columns(2)

with col1:
    mileage = st.number_input("📏 Пробег (км)", min_value=0, max_value=500000, value=50000, step=1000)
    engine_power = st.number_input("⚡ Мощность (л.с.)", min_value=50, max_value=500, value=150, step=10)

with col2:
    year = st.number_input("📅 Год выпуска", min_value=2000, max_value=2026, value=2020, step=1)
    brand = st.selectbox("🏷️ Марка", ["Toyota", "BMW", "Mercedes", "Lada"])

# Отступ перед кнопкой
st.markdown("<br>", unsafe_allow_html=True)

# Кнопка по центру
_, btn_col, _ = st.columns([1, 3, 1])
with btn_col:
    if st.button("💰 Узнать цену", use_container_width=True, type="primary"):
        try:
            with st.spinner("🧮 Считаем прогноз..."):
                input_df = pd.DataFrame([{
                    "mileage": mileage,
                    "engine_power": engine_power,
                    "year": year,
                    "brand": brand
                }])
                processed = preprocessor.transform(input_df)
                tensor_in = torch.FloatTensor(processed)
                with torch.no_grad():
                    pred_norm = model(tensor_in)
                price = price_scaler.inverse_transform(pred_norm.numpy().reshape(-1, 1)).item()

            # Вывод результата
            st.markdown(f"""
                <div style='background-color: #28a745; padding: 15px; border-radius: 8px; text-align: center; margin-top: 10px;'>
                    <h2 style='color: white; margin: 0;'>💰 {price:,.0f} руб</h2>
                    <p style='color: white; margin: 5px 0 0 0;'>Прогнозируемая стоимость</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")

st.markdown("---")

with st.expander("ℹ️ О модели"):
    st.write("""
        - **Архитектура:** MLP, 1 скрытый слой (16 нейронов)
        - **Признаки:** пробег, мощность, год, марка 
        - **R² Test:** ~98.7%
        - ⚠️ Прогноз на синтетических данных
    """)

st.markdown("---")
st.caption("Проект создан в рамках обучения ML | [GitHub](https://github.com/RomanRu96)")

