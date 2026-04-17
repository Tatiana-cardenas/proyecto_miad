import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# =========================================================
# 0. Carpeta de salida para gráficos
# =========================================================
CARPETA_GRAFICOS = "graficos_modelos"
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

# =========================================================
# 1. Cargar base
# =========================================================
df = pd.read_csv("base_final.csv", encoding="utf-8-sig")

# =========================================================
# 2. Variables
# =========================================================
variables_predictoras = [
    "Precipitación acumulada anual (mm/año)",
    "Temperatura media anual (°C)",
    "Máximo de la temperatura media mensual (°C)",
    "Mínimo de la temperatura media mensual (°C)",
    "Humedad relativa media anual (%)",
    "Radiación solar acumulada anual (MJ/m²/año)",
    "Humedad volumétrica media anual del suelo capa 1 (m³/m³)",
    "Humedad volumétrica media anual del suelo capa 2 (m³/m³)",
    "Evaporación potencial acumulada anual (mm/año)",
    "altitud_media_m"
]

variable_objetivo = "Rendimiento"

faltantes = [col for col in variables_predictoras + [variable_objetivo] if col not in df.columns]
if faltantes:
    raise ValueError(f"Faltan estas columnas en la base: {faltantes}")

# =========================================================
# 3. Limpieza
# =========================================================
df_modelo = df[variables_predictoras + [variable_objetivo]].dropna().copy()

X = df_modelo[variables_predictoras].copy()
y_original = df_modelo[variable_objetivo].copy()

# transformación logarítmica
y = np.log1p(y_original)

print("Dimensión final de la base:", df_modelo.shape)

# =========================================================
# 4. Train / test
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# 5. Funciones
# =========================================================
def limpiar_nombre_archivo(texto):
    reemplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N",
        "²": "2", "³": "3",
        "/": "_", "\\": "_", ":": "_", "*": "_", "?": "_",
        '"': "_", "<": "_", ">": "_", "|": "_", " ": "_",
        "(": "", ")": "", "%": "pct"
    }
    for a, b in reemplazos.items():
        texto = texto.replace(a, b)
    return texto

def metricas_escala_original(y_real_log, y_pred_log):
    y_real = np.expm1(y_real_log)
    y_pred = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    return rmse, mae, r2, y_real, y_pred

def imprimir_metricas(nombre, y_train_log, y_pred_train_log, y_test_log, y_pred_test_log):
    rmse_train, mae_train, r2_train, _, _ = metricas_escala_original(y_train_log, y_pred_train_log)
    rmse_test, mae_test, r2_test, y_test_real, y_test_pred = metricas_escala_original(y_test_log, y_pred_test_log)

    print(f"\n========== {nombre.upper()} ==========")
    print("Entrenamiento")
    print(f"RMSE: {rmse_train:.4f}")
    print(f"MAE : {mae_train:.4f}")
    print(f"R²  : {r2_train:.4f}")
    print("\nPrueba")
    print(f"RMSE: {rmse_test:.4f}")
    print(f"MAE : {mae_test:.4f}")
    print(f"R²  : {r2_test:.4f}")

    return rmse_train, mae_train, r2_train, rmse_test, mae_test, r2_test, y_test_real, y_test_pred

def graficar_real_vs_pred(y_real, y_pred, titulo):
    nombre_archivo = limpiar_nombre_archivo(titulo) + ".png"
    ruta_salida = os.path.join(CARPETA_GRAFICOS, nombre_archivo)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_real, y_pred, alpha=0.7)
    min_val = min(np.min(y_real), np.min(y_pred))
    max_val = max(np.max(y_real), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico guardado en: {ruta_salida}")

def mostrar_importancias(modelo, nombre, variables):
    if hasattr(modelo, "feature_importances_"):
        imp = pd.DataFrame({
            "Variable": variables,
            "Importancia": modelo.feature_importances_
        }).sort_values("Importancia", ascending=False)

        print(f"\nImportancia de variables - {nombre}")
        print(imp)

        nombre_archivo = limpiar_nombre_archivo(f"Importancia de variables - {nombre}") + ".png"
        ruta_salida = os.path.join(CARPETA_GRAFICOS, nombre_archivo)

        plt.figure(figsize=(8, 5))
        plt.barh(imp["Variable"], imp["Importancia"])
        plt.gca().invert_yaxis()
        plt.title(f"Importancia de variables - {nombre}")
        plt.tight_layout()
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Gráfico guardado en: {ruta_salida}")

# scorer para CV en escala log
rmse_scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    greater_is_better=False
)

def evaluar_cv(modelo, X, y, nombre):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        modelo,
        X, y,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": rmse_scorer
        },
        n_jobs=-1
    )

    resumen = {
        "Modelo": nombre,
        "CV_R2_mean": np.mean(scores["test_r2"]),
        "CV_R2_std": np.std(scores["test_r2"]),
        "CV_MAE_mean_log": -np.mean(scores["test_mae"]),
        "CV_RMSE_mean_log": -np.mean(scores["test_rmse"])
    }
    return resumen

# =========================================================
# 6. Modelos base
# =========================================================
resultados = []
resultados_cv = []

# ---------------------------------------------------------
# 6.1 Regresión lineal
# ---------------------------------------------------------
modelo_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", LinearRegression())
])

modelo_lr.fit(X_train, y_train)

y_pred_train_lr = modelo_lr.predict(X_train)
y_pred_test_lr = modelo_lr.predict(X_test)

res_lr = imprimir_metricas("Regresion Lineal log", y_train, y_pred_train_lr, y_test, y_pred_test_lr)
rmse_train, mae_train, r2_train, rmse_test, mae_test, r2_test, y_real_plot, y_pred_plot = res_lr

resultados.append({
    "Modelo": "Regresion Lineal log",
    "RMSE_train": rmse_train,
    "MAE_train": mae_train,
    "R2_train": r2_train,
    "RMSE_test": rmse_test,
    "MAE_test": mae_test,
    "R2_test": r2_test
})

resultados_cv.append(evaluar_cv(modelo_lr, X, y, "Regresion Lineal log"))
graficar_real_vs_pred(y_real_plot, y_pred_plot, "Regresion Lineal log - Real vs Predicho")

# ---------------------------------------------------------
# 6.2 Gradient Boosting con búsqueda de hiperparámetros
# ---------------------------------------------------------
gb_base = GradientBoostingRegressor(random_state=42)

param_dist_gb = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "max_features": ["sqrt", "log2", None]
}

random_search_gb = RandomizedSearchCV(
    estimator=gb_base,
    param_distributions=param_dist_gb,
    n_iter=20,
    scoring="r2",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search_gb.fit(X_train, y_train)

mejor_gb = random_search_gb.best_estimator_
print("\nMejores parámetros Gradient Boosting:")
print(random_search_gb.best_params_)

y_pred_train_gb = mejor_gb.predict(X_train)
y_pred_test_gb = mejor_gb.predict(X_test)

res_gb = imprimir_metricas(
    "Gradient Boosting Tuned log",
    y_train, y_pred_train_gb,
    y_test, y_pred_test_gb
)
rmse_train, mae_train, r2_train, rmse_test, mae_test, r2_test, y_real_plot, y_pred_plot = res_gb

resultados.append({
    "Modelo": "Gradient Boosting Tuned log",
    "RMSE_train": rmse_train,
    "MAE_train": mae_train,
    "R2_train": r2_train,
    "RMSE_test": rmse_test,
    "MAE_test": mae_test,
    "R2_test": r2_test
})

resultados_cv.append(evaluar_cv(mejor_gb, X, y, "Gradient Boosting Tuned log"))
mostrar_importancias(mejor_gb, "Gradient Boosting Tuned log", X.columns)
graficar_real_vs_pred(y_real_plot, y_pred_plot, "Gradient Boosting Tuned log - Real vs Predicho")

# =========================================================
# 7. Tuning Random Forest
# =========================================================
rf_base = RandomForestRegressor(random_state=42)

param_dist_rf = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 8, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

random_search_rf = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist_rf,
    n_iter=20,
    scoring="r2",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search_rf.fit(X_train, y_train)

mejor_rf = random_search_rf.best_estimator_
print("\nMejores parámetros Random Forest:")
print(random_search_rf.best_params_)

y_pred_train_rf = mejor_rf.predict(X_train)
y_pred_test_rf = mejor_rf.predict(X_test)

res_rf = imprimir_metricas("Random Forest Tuned log", y_train, y_pred_train_rf, y_test, y_pred_test_rf)
rmse_train, mae_train, r2_train, rmse_test, mae_test, r2_test, y_real_plot, y_pred_plot = res_rf

resultados.append({
    "Modelo": "Random Forest Tuned log",
    "RMSE_train": rmse_train,
    "MAE_train": mae_train,
    "R2_train": r2_train,
    "RMSE_test": rmse_test,
    "MAE_test": mae_test,
    "R2_test": r2_test
})

resultados_cv.append(evaluar_cv(mejor_rf, X, y, "Random Forest Tuned log"))
mostrar_importancias(mejor_rf, "Random Forest Tuned log", X.columns)
graficar_real_vs_pred(y_real_plot, y_pred_plot, "Random Forest Tuned log - Real vs Predicho")

# =========================================================
# 8. Tuning XGBoost
# =========================================================
try:
    from xgboost import XGBRegressor

    xgb_base = XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )

    param_dist_xgb = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5]
    }

    random_search_xgb = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist_xgb,
        n_iter=20,
        scoring="r2",
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search_xgb.fit(X_train, y_train)

    mejor_xgb = random_search_xgb.best_estimator_
    print("\nMejores parámetros XGBoost:")
    print(random_search_xgb.best_params_)

    y_pred_train_xgb = mejor_xgb.predict(X_train)
    y_pred_test_xgb = mejor_xgb.predict(X_test)

    res_xgb = imprimir_metricas("XGBoost Tuned log", y_train, y_pred_train_xgb, y_test, y_pred_test_xgb)
    rmse_train, mae_train, r2_train, rmse_test, mae_test, r2_test, y_real_plot, y_pred_plot = res_xgb

    resultados.append({
        "Modelo": "XGBoost Tuned log",
        "RMSE_train": rmse_train,
        "MAE_train": mae_train,
        "R2_train": r2_train,
        "RMSE_test": rmse_test,
        "MAE_test": mae_test,
        "R2_test": r2_test
    })

    resultados_cv.append(evaluar_cv(mejor_xgb, X, y, "XGBoost Tuned log"))
    mostrar_importancias(mejor_xgb, "XGBoost Tuned log", X.columns)
    graficar_real_vs_pred(y_real_plot, y_pred_plot, "XGBoost Tuned log - Real vs Predicho")

except ImportError:
    print("\nNo tienes instalado xgboost. Instálalo con:")
    print("pip install xgboost")

# =========================================================
# 9. Tablas finales
# =========================================================
df_resultados = pd.DataFrame(resultados).sort_values(by="R2_test", ascending=False)
df_cv = pd.DataFrame(resultados_cv).sort_values(by="CV_R2_mean", ascending=False)

print("\n========== RESULTADOS TRAIN / TEST ==========")
print(df_resultados)

print("\n========== RESULTADOS VALIDACIÓN CRUZADA ==========")
print(df_cv)

df_resultados.to_csv("resultados_modelos_log.csv", index=False, encoding="utf-8-sig")
df_cv.to_csv("resultados_cv_modelos_log.csv", index=False, encoding="utf-8-sig")

# =========================================================
# 10. Gráficos comparativos
# =========================================================
plt.figure(figsize=(10, 5))
plt.bar(df_resultados["Modelo"], df_resultados["R2_test"])
plt.title("Comparación de modelos por R2 en prueba")
plt.ylabel("R2 test")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "comparacion_modelos_r2_test.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(df_resultados["Modelo"], df_resultados["RMSE_test"])
plt.title("Comparación de modelos por RMSE en prueba")
plt.ylabel("RMSE test")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "comparacion_modelos_rmse_test.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(df_cv["Modelo"], df_cv["CV_R2_mean"])
plt.title("Comparación de modelos por R2 promedio en validación cruzada")
plt.ylabel("CV R2 mean")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "comparacion_modelos_cv_r2_mean.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"\nTodos los gráficos fueron guardados en la carpeta: {CARPETA_GRAFICOS}")