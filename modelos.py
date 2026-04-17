import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Cargar base
df = pd.read_csv("base_final.csv", encoding="utf-8-sig")

# 2. Definir X e y
X = df[
    [
        "Precipitación acumulada anual (mm/año)",
        "Temperatura media anual (°C)",
        "Humedad relativa media anual (%)",
        "Radiación solar acumulada anual (MJ/m²/año)",
        "altitud_media_m"
    ]
]

y = df["Rendimiento"]

# 3. Eliminar faltantes en X o y
df_modelo = pd.concat([X, y], axis=1).dropna()
X = df_modelo[X.columns]
y = df_modelo["Rendimiento"]

# 4. Separar train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 5. Pipeline
pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", LinearRegression())
])

# 6. Entrenamiento
pipeline_lr.fit(X_train, y_train)

# 7. Predicción
y_pred = pipeline_lr.predict(X_test)

# 8. Métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("========== REGRESIÓN LINEAL CON ESTANDARIZACIÓN ==========")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# 9. Coeficientes
modelo = pipeline_lr.named_steps["modelo"]
coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo.coef_
}).sort_values(by="Coeficiente", ascending=False)

print("\nCoeficientes estandarizados:")
print(coef_df)