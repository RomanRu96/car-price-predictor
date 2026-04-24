# train.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib  # Стандарт для сохранения sklearn-объектов
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Фиксируем случайность для воспроизводимости
np.random.seed(42)
torch.manual_seed(42)


# === 1. Загрузка данных в DataFrame ===
data = {
    "brand": ["Toyota", "BMW", "Lada", "Toyota", "Mercedes", "Lada", 
              "BMW", "Toyota", "Mercedes", "Lada", "BMW", "Toyota"],
    "year": [2018, 2020, 2015, 2019, 2021, 2014, 2019, 2017, 2020, 2016, 2021, 2018],
    "mileage": [50000, 20000, 120000, 45000, 15000, 150000, 
                35000, 80000, 25000, 100000, 10000, 60000],
    "engine_power": [150, 250, 90, 150, 300, 90, 
                     250, 150, 300, 90, 250, 150],
    "price": [1500000, 3500000, 400000, 1600000, 4500000, 350000, 
              3200000, 1400000, 4200000, 380000, 3800000, 1550000]
}

df = pd.DataFrame(data)


# === 2. Разделение признаков и целевой переменной ===
numeric_cols = ["mileage", "engine_power", "year"]
categorical_cols = ["brand"]
target_col = ["price"]

X = df[numeric_cols + categorical_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Препроцессинг (ColumnTranformer + Pandas) ===

preprocessor = ColumnTransformer(
    transformers = [
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)  
        # handle_unknown="ignore" это значит что все новые брэнды, которых нет в train будут игнорироваться и браться как [0,0,0,0,0,0] 
        # sparse_output=False - не убирает нули из кодировки, чтобы массив был более читаемым 
    ]

)

# Автоматически: масштабирует числа, кодирует строки, склеивает в матрицу
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)


# === 4. Нормализация цены (Target Scaling) ===
price_scaler = StandardScaler()
y_train_scaled = price_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = price_scaler.transform(y_test.values.reshape(-1, 1)).flatten()


# === 5. Тензоры PyTorch ===
X_train_t = torch.FloatTensor(X_train_proc) 
y_train_t = torch.FloatTensor(y_train_scaled).reshape(-1, 1)
X_test_t = torch.FloatTensor(X_test_proc)
y_test_t = torch.FloatTensor(y_test_scaled).reshape(-1, 1)


# === 6. Модель ===
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

# input_size посчитается автоматически (3 числа + 4 бренда = 7)
model = CarPredictor(X_train_proc.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# === 7. Обучение ===
print("\nОбучение модели...")
for epoch in range(1000):  # Увеличили эпохи для lr=0.001
    preds = model(X_train_t)
    loss = criterion(preds, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Эпоха {epoch}, loss: {loss.item()}")

print(f"\nФинальный Loss: {loss.item():.4f}")


# === 8. Оценка ===
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).numpy().flatten()
    test_pred = model(X_test_t).numpy().flatten()

def calc_r2(pred, true): 
    return 1 - (np.sum((true - pred)**2) / np.sum((true - np.mean(true))**2))

print(f"\nR² Train: {calc_r2(train_pred, y_train_scaled):.2%}")
print(f"R² Test:  {calc_r2(test_pred, y_test_scaled):.2%}")



# === 9. Сохранение артефактов (КРИТИЧЕСКИ ВАЖНО!) ===
torch.save(model.state_dict(), "model.pth")           # Веса модели
joblib.dump(preprocessor, "preprocessor.joblib")      # числовые признаки
joblib.dump(price_scaler, "price_scaler.joblib")      # признаки price

print("\nФайлы успешно сохранены: \n\nmodel.pth \npreprocessor.joblib \nprice_scaler.joblib")
