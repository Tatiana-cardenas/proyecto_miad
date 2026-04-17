import pandas as pd
import numpy as np
import optuna

from sklearn.model_selection import train_test_split, cross_val_score, KFold
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

# Transformar y
y_log = np.log1p(df_clean["Rendimiento"])
 
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
 
# Entrenar el modelo (RF u otro)
pipeline_lr.fit(X_train, y_train)
y_pred_log = pipeline_lr.predict(X_test)
 
# Revertir la transformación para métricas reales
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)
 
mae  = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2   = r2_score(y_test_real, y_pred_real)
 
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Eliminar faltantes
data_modelo = pd.concat([X, y], axis=1).dropna()
X = data_modelo.drop(columns=["Rendimiento"])
y = data_modelo["Rendimiento"]

# =========================
# 2. Separar train/test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 3. Función objetivo Optuna
# =========================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [None] + list(range(3, 31))),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1
    }

    modelo = RandomForestRegressor(**params)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    rmse_cv = -scores.mean()
    return rmse_cv

# =========================
# 4. Crear y ejecutar estudio
# =========================
study = optuna.create_study(
    direction="minimize",
    study_name="random_forest_optuna"
)

study.optimize(objective, n_trials=50)

# =========================
# 5. Mejores parámetros
# =========================
print("Mejores hiperparámetros encontrados:")
print(study.best_params)

print(f"\nMejor RMSE CV: {study.best_value:.4f}")

# =========================
# 6. Entrenar mejor modelo
# =========================
best_params = study.best_params.copy()
best_params["random_state"] = 42
best_params["n_jobs"] = -1

mejor_rf = RandomForestRegressor(**best_params)
mejor_rf.fit(X_train, y_train)

# =========================
# 7. Predicciones
# =========================
y_pred_train = mejor_rf.predict(X_train)
y_pred_test = mejor_rf.predict(X_test)

# =========================
# 8. Métricas
# =========================
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n========== RANDOM FOREST AJUSTADO CON OPTUNA ==========")
print("Entrenamiento")
print(f"RMSE: {rmse_train:.4f}")
print(f"MAE : {mae_train:.4f}")
print(f"R²  : {r2_train:.4f}")

print("\nPrueba")
print(f"RMSE: {rmse_test:.4f}")
print(f"MAE : {mae_test:.4f}")
print(f"R²  : {r2_test:.4f}")

# =========================
# 9. Importancia de variables
# =========================
importancias = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": mejor_rf.feature_importances_
}).sort_values(by="Importancia", ascending=False)

print("\nImportancia de variables:")
print(importancias)

# =========================
# 10. Guardar resultados de Optuna
# =========================
resultados_optuna = study.trials_dataframe()
resultados_optuna.to_csv("resultados_optuna_random_forest.csv", index=False, encoding="utf-8-sig")

print("\nSe guardó el archivo: resultados_optuna_random_forest.csv")