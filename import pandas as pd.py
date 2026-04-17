import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# 1. Cargar datos
# =========================
df = pd.read_csv("base_final.csv", encoding="utf-8-sig")

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

data_modelo = pd.concat([X, y], axis=1).dropna()
X = data_modelo.drop(columns=["Rendimiento"])
y = data_modelo["Rendimiento"]

# =========================
# 2. Train/test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# 3. Modelo base
# =========================
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# =========================
# 4. Malla de hiperparámetros
# =========================
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None]
}

# =========================
# 5. Búsqueda
# =========================
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("Mejores parámetros:")
print(grid.best_params_)

mejor_rf = grid.best_estimator_

# =========================
# 6. Predicciones
# =========================
y_pred_train = mejor_rf.predict(X_train)
y_pred_test = mejor_rf.predict(X_test)

# =========================
# 7. Métricas
# =========================
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n========== RANDOM FOREST AJUSTADO ==========")
print("Entrenamiento")
print(f"RMSE: {rmse_train:.4f}")
print(f"MAE : {mae_train:.4f}")
print(f"R²  : {r2_train:.4f}")

print("\nPrueba")
print(f"RMSE: {rmse_test:.4f}")
print(f"MAE : {mae_test:.4f}")
print(f"R²  : {r2_test:.4f}")

# =========================
# 8. Importancia de variables
# =========================
importancias = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": mejor_rf.feature_importances_
}).sort_values(by="Importancia", ascending=False)

print("\nImportancia de variables:")
print(importancias)