# predict.py
import torch
import torch.nn as nn
import pandas as pd
import joblib

# === 1. Архитектура модели (должна совпадать с train.py) ===

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


# === 2. Загрузка артефактов ===

preprocessor = joblib.load("preprocessor.joblib")
price_scaler = joblib.load("price_scaler.joblib")


model = CarPredictor(7) # 3 признака + 4 кодировки марки машин 
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

print("\nАртефакты загружены успешно")


# === 3. Функция предсказания ===
def predict_car(mileage: int, engine_power: int, year: int, brand: str) -> float:
    # 1. Создаем DataFrame с той же схемой, что при обучении
    input_df = pd.DataFrame([{
        "mileage": mileage,
        "engine_power": engine_power,
        "year": year,
        "brand": brand 

    }])

    # 2. Пропускаем через препроцессор (сам масштабирует и кодирует)
    processed = preprocessor.transform(input_df)

    # 3. Предсказание
    tensor_in = torch.FloatTensor(processed)
    with torch.no_grad():
        pred_norm = model(tensor_in)

    # 4. Денормализация цены обратно в рубли
    pred_rub = price_scaler.inverse_transform(pred_norm.numpy().reshape(-1, 1)).item()

    return pred_rub




# === 3. Тестирование ===
if __name__ == "__main__":
    print("\n" + "="*55)
    print("\nCAR PRICE PREDICTOR")
    print("\n" + "="*55)

    cases = [
        (40000, 150, 2019, "Toyota"),
        (20000, 250, 2020, "BMW"),
        (100000, 90, 2016, "Lada"),
        (5000, 300, 2021, "Mercedes")
    ]

    for mileage, engine_power, year, brand in cases:
        price = predict_car(mileage, engine_power, year, brand)
        print(f"{brand}, {year}, {engine_power} л.с.,{mileage:,} км --> {price:,.0f} руб")

    print("="*55)

