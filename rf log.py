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
 
FEATURES = [
    "Precipitación acumulada anual (mm/año)",
    "Temperatura media anual (°C)",
    "Humedad relativa media anual (%)",
    "Radiación solar acumulada anual (MJ/m²/año)",
    "altitud_media_m"
]
 
X = df[FEATURES]
y = df["Rendimiento"]
 
# =========================
# 2. Limpiar y transformar
# =========================
data_modelo = pd.concat([X, y], axis=1).dropna()
X = data_modelo[FEATURES]
y = data_modelo["Rendimiento"]
 
# 🔴 Eliminar outliers extremos (rendimiento > percentil 99)
p99 = y.quantile(0.99)
print(f"Percentil 99 del Rendimiento: {p99:.4f}")
mask = y <= p99
X = X[mask]
y = y[mask]
print(f"Filas después de filtrar outliers: {len(y)}")
 
# ✅ Transformación logarítmica del target
y_log = np.log1p(y)
 
# =========================
# 3. Separar train/test
# =========================
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)
 
# =========================
# 4. Función objetivo Optuna
# =========================
def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth":         trial.suggest_categorical("max_depth", [None] + list(range(3, 31))),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1
    }
 
    modelo = RandomForestRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
 
    scores = cross_val_score(
        modelo,
        X_train,
        y_train_log,          # ← entrena en escala log
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
 
    return -scores.mean()
 
# =========================
# 5. Crear y ejecutar estudio
# =========================
optuna.logging.set_verbosity(optuna.logging.WARNING)
 
study = optuna.create_study(direction="minimize", study_name="rf_log_optuna")
study.optimize(objective, n_trials=50)
 
print("Mejores hiperparámetros:")
print(study.best_params)
print(f"Mejor RMSE CV (escala log): {study.best_value:.4f}")
 
# =========================
# 6. Entrenar mejor modelo
# =========================
best_params = study.best_params.copy()
best_params["random_state"] = 42
best_params["n_jobs"] = -1
 
mejor_rf = RandomForestRegressor(**best_params)
mejor_rf.fit(X_train, y_train_log)
 
# =========================
# 7. Predicciones y revertir log
# =========================
y_pred_train_log = mejor_rf.predict(X_train)
y_pred_test_log  = mejor_rf.predict(X_test)
 
# ✅ Revertir transformación → escala original
y_pred_train = np.expm1(y_pred_train_log)
y_pred_test  = np.expm1(y_pred_test_log)
y_train_real = np.expm1(y_train_log)
y_test_real  = np.expm1(y_test_log)
 
# =========================
# 8. Métricas en escala original
# =========================
def metricas(y_true, y_pred, nombre):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{nombre}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")
 
print("\n========== RANDOM FOREST + LOG-TARGET + OPTUNA ==========")
metricas(y_train_real, y_pred_train, "Entrenamiento")
metricas(y_test_real,  y_pred_test,  "Prueba")
 
# =========================
# 9. Importancia de variables
# =========================
importancias = pd.DataFrame({
    "Variable":    FEATURES,
    "Importancia": mejor_rf.feature_importances_
}).sort_values(by="Importancia", ascending=False)
 
print("\nImportancia de variables:")
print(importancias)
 
# =========================
# 10. Guardar resultados
# =========================
resultados_optuna = study.trials_dataframe()
resultados_optuna.to_csv("resultados_optuna_rf_log.csv", index=False, encoding="utf-8-sig")
print("\nGuardado: resultados_optuna_rf_log.csv")