import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
 
# =========================================================
# 0. Configuración de salida
# =========================================================
CARPETA_GRAFICOS = "graficos_modelos"
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
 
# =========================================================
# 1. Cargar base
# =========================================================
# Ajusta esta ruta si corres en Colab:
# df = pd.read_csv("/content/drive/MyDrive/proyecto_miad/base_final.csv", encoding="utf-8-sig")
df = pd.read_csv("base_final.csv", encoding="utf-8-sig")
print(f"Base cargada: {df.shape}")
 
# =========================================================
# 2. Limpieza de datos
# =========================================================
# Eliminar Rendimiento == 0 (registros erróneos o sin cosecha)
df = df[df["Rendimiento"] > 0].copy()
print(f"Tras eliminar Rendimiento=0: {df.shape}")
 
# Eliminar outliers extremos (>P99.5) — distorsionan el entrenamiento
p995 = df["Rendimiento"].quantile(0.995)
df = df[df["Rendimiento"] <= p995].copy()
print(f"Tras eliminar outliers extremos (>P99.5={p995:.2f}): {df.shape}")
 
# =========================================================
# 3. Feature Engineering
# =========================================================
# Variable objetivo
variable_objetivo = "Rendimiento"
 
# Variables climáticas originales
vars_climaticas = [
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
 
# ---- Nuevas features derivadas ----
 
# Amplitud térmica: diferencia entre máx y mín temperatura
df["amplitud_termica"] = (
    df["Máximo de la temperatura media mensual (°C)"] -
    df["Mínimo de la temperatura media mensual (°C)"]
)
 
# Índice de aridez simplificado: precipitación / evaporación potencial
df["indice_aridez"] = (
    df["Precipitación acumulada anual (mm/año)"] /
    (df["Evaporación potencial acumulada anual (mm/año)"] + 1e-6)
)
 
# Media de humedad de suelo (ambas capas)
df["humedad_suelo_media"] = (
    df["Humedad volumétrica media anual del suelo capa 1 (m³/m³)"] +
    df["Humedad volumétrica media anual del suelo capa 2 (m³/m³)"]
) / 2
 
# Tendencia temporal normalizada (para capturar cambios tecnológicos/climáticos)
df["año_norm"] = (df["Año"] - df["Año"].min()) / (df["Año"].max() - df["Año"].min())
 
# Logaritmo del área para reducir escala
df["log_area_cosechada"] = np.log1p(df["Área Cosechada"])
df["log_area_sembrada"] = np.log1p(df["Área Sembrada"])
 
# Eficiencia de uso del área
df["ratio_cosecha_siembra"] = df["Área Cosechada"] / (df["Área Sembrada"] + 1e-6)
 
# ---- Encoging de municipio/departamento (Target Encoding) ----
# Se calculará dentro del pipeline para evitar data leakage
 
vars_numericas = vars_climaticas + [
    "amplitud_termica",
    "indice_aridez",
    "humedad_suelo_media",
    "año_norm",
    "log_area_cosechada",
    "log_area_sembrada",
    "ratio_cosecha_siembra"
]
 
vars_categoricas = ["Departamento"]  # Municipio tiene 570 niveles; usamos Departamento
 
todas_las_vars = vars_numericas + vars_categoricas
 
# Verificar columnas
faltantes = [c for c in todas_las_vars + [variable_objetivo] if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas: {faltantes}")
 
df_modelo = df[todas_las_vars + [variable_objetivo]].dropna().copy()
print(f"Dimensión final para modelado: {df_modelo.shape}")
 
X = df_modelo[todas_las_vars].copy()
y_original = df_modelo[variable_objetivo].copy()
 
# Transformación logarítmica de la variable objetivo
y = np.log1p(y_original)
 
print(f"\nDistribución de Rendimiento (original):")
print(y_original.describe())
print(f"Skewness original: {y_original.skew():.2f}")
print(f"Skewness log1p: {y.skew():.2f}")
 
# =========================================================
# 4. Train / test (split temporal para evitar data leakage)
# =========================================================
# Separar por año (más realista que split aleatorio en datos de panel)
anio_corte = df_modelo.index.map(lambda i: df.loc[i, "Año"] if i in df.index else None)
 
# Reconstruir año en df_modelo para el split
df_modelo["Año"] = df.loc[df_modelo.index, "Año"].values
anio_corte_val = int(df_modelo["Año"].quantile(0.80))
print(f"\nSplit temporal: entrenamiento hasta {anio_corte_val}, prueba desde {anio_corte_val+1}")
 
mask_train = df_modelo["Año"] <= anio_corte_val
mask_test  = df_modelo["Año"] >  anio_corte_val
 
X_train = X[mask_train]
X_test  = X[mask_test]
y_train = y[mask_train]
y_test  = y[mask_test]
 
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
 
# =========================================================
# 5. Preprocesador (ColumnTransformer)
# =========================================================
# Target Encoding para Departamento (evita data leakage al ir dentro del pipeline)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), vars_numericas),
        ("cat", TargetEncoder(random_state=42), vars_categoricas),
    ],
    remainder="drop"
)
 
# =========================================================
# 6. Funciones auxiliares
# =========================================================
def limpiar_nombre_archivo(texto):
    reemplazos = {
        "á": "a","é": "e","í": "i","ó": "o","ú": "u",
        "Á": "A","É": "E","Í": "I","Ó": "O","Ú": "U",
        "ñ": "n","Ñ": "N","²": "2","³": "3",
        "/": "_","\\": "_",":": "_","*": "_","?": "_",
        '"': "_","<": "_",">": "_","|": "_"," ": "_",
        "(": "",")"  : "","%": "pct"
    }
    for a, b in reemplazos.items():
        texto = texto.replace(a, b)
    return texto
 
def metricas_escala_original(y_real_log, y_pred_log):
    y_real = np.expm1(y_real_log)
    y_pred = np.expm1(y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae  = mean_absolute_error(y_real, y_pred)
    r2   = r2_score(y_real, y_pred)
    return rmse, mae, r2, y_real, y_pred
 
def imprimir_metricas(nombre, y_train_log, y_pred_train_log, y_test_log, y_pred_test_log):
    rmse_tr, mae_tr, r2_tr, _, _       = metricas_escala_original(y_train_log, y_pred_train_log)
    rmse_te, mae_te, r2_te, y_re, y_pr = metricas_escala_original(y_test_log, y_pred_test_log)
 
    print(f"\n{'='*50}")
    print(f" {nombre.upper()}")
    print(f"{'='*50}")
    print(f"{'':20s} {'TRAIN':>10s} {'TEST':>10s}")
    print(f"{'RMSE':20s} {rmse_tr:>10.4f} {rmse_te:>10.4f}")
    print(f"{'MAE':20s} {mae_tr:>10.4f} {mae_te:>10.4f}")
    print(f"{'R²':20s} {r2_tr:>10.4f} {r2_te:>10.4f}")
 
    return rmse_tr, mae_tr, r2_tr, rmse_te, mae_te, r2_te, y_re, y_pr
 
def graficar_real_vs_pred(y_real, y_pred, titulo):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    # Scatter real vs predicho
    axes[0].scatter(y_real, y_pred, alpha=0.4, s=10)
    mn = min(y_real.min(), y_pred.min())
    mx = max(y_real.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "--", color="red", lw=1.5)
    axes[0].set_xlabel("Valor real")
    axes[0].set_ylabel("Valor predicho")
    axes[0].set_title(f"{titulo}\nReal vs Predicho")
 
    # Residuos
    residuos = y_pred - y_real
    axes[1].scatter(y_pred, residuos, alpha=0.4, s=10)
    axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Valor predicho")
    axes[1].set_ylabel("Residuo")
    axes[1].set_title("Residuos vs Predicho")
 
    plt.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, limpiar_nombre_archivo(titulo) + ".png")
    plt.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Gráfico guardado: {ruta}")
 
def mostrar_importancias(modelo_final, nombre, vars_num, vars_cat):
    # Extraer el estimador del pipeline
    estimator = modelo_final.named_steps["modelo"]
    if not hasattr(estimator, "feature_importances_"):
        return
 
    # Nombres de columnas tras el ColumnTransformer
    nombres = vars_num + vars_cat
    importancias = estimator.feature_importances_
 
    imp = pd.DataFrame({"Variable": nombres[:len(importancias)],
                         "Importancia": importancias})\
            .sort_values("Importancia", ascending=False)
 
    print(f"\nImportancia de variables — {nombre}")
    print(imp.to_string(index=False))
 
    ruta = os.path.join(CARPETA_GRAFICOS,
                        limpiar_nombre_archivo(f"Importancia_{nombre}") + ".png")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp["Variable"][:15], imp["Importancia"][:15])
    ax.invert_yaxis()
    ax.set_title(f"Top 15 variables — {nombre}")
    plt.tight_layout()
    plt.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Gráfico importancias: {ruta}")
 
rmse_scorer = make_scorer(
    lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
    greater_is_better=False
)
 
def evaluar_cv(modelo, X, y, nombre):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(modelo, X, y, cv=cv,
                            scoring={"r2": "r2",
                                     "mae": "neg_mean_absolute_error",
                                     "rmse": rmse_scorer},
                            n_jobs=-1)
    return {
        "Modelo": nombre,
        "CV_R2_mean": np.mean(scores["test_r2"]),
        "CV_R2_std":  np.std(scores["test_r2"]),
        "CV_MAE_mean": -np.mean(scores["test_mae"]),
        "CV_RMSE_mean": -np.mean(scores["test_rmse"])
    }
 
# =========================================================
# 7. Modelos
# =========================================================
resultados    = []
resultados_cv = []
 
# ---------------------------------------------------------
# 7.1 Ridge (reemplaza Regresión Lineal simple — más robusta)
# ---------------------------------------------------------
print("\n>>> Entrenando Ridge Regression...")
 
modelo_ridge = Pipeline([
    ("prep",   preprocessor),
    ("modelo", Ridge(alpha=10.0))
])
modelo_ridge.fit(X_train, y_train)
 
yp_tr = modelo_ridge.predict(X_train)
yp_te = modelo_ridge.predict(X_test)
 
res = imprimir_metricas("Ridge Regression", y_train, yp_tr, y_test, yp_te)
rmse_tr, mae_tr, r2_tr, rmse_te, mae_te, r2_te, y_re, y_pr = res
 
resultados.append({"Modelo": "Ridge Regression",
                   "RMSE_train": rmse_tr, "MAE_train": mae_tr, "R2_train": r2_tr,
                   "RMSE_test":  rmse_te, "MAE_test":  mae_te, "R2_test":  r2_te})
resultados_cv.append(evaluar_cv(modelo_ridge, X, y, "Ridge Regression"))
graficar_real_vs_pred(y_re, y_pr, "Ridge Regression - Real vs Predicho")
 
# ---------------------------------------------------------
# 7.2 Random Forest con hiperparámetros
# ---------------------------------------------------------
print("\n>>> Buscando hiperparámetros para Random Forest...")
 
param_dist_rf = {
    "modelo__n_estimators":    [200, 400, 600],
    "modelo__max_depth":       [None, 8, 12, 20],
    "modelo__min_samples_split": [2, 5, 10],
    "modelo__min_samples_leaf":  [1, 2, 4],
    "modelo__max_features":    ["sqrt", "log2", 0.5]
}
 
pipe_rf = Pipeline([
    ("prep",   preprocessor),
    ("modelo", RandomForestRegressor(random_state=42, n_jobs=-1))
])
 
search_rf = RandomizedSearchCV(
    estimator=pipe_rf,
    param_distributions=param_dist_rf,
    n_iter=20, scoring="r2", cv=5,
    verbose=1, random_state=42, n_jobs=-1
)
search_rf.fit(X_train, y_train)
mejor_rf = search_rf.best_estimator_
 
print("Mejores parámetros RF:", search_rf.best_params_)
 
yp_tr = mejor_rf.predict(X_train)
yp_te = mejor_rf.predict(X_test)
 
res = imprimir_metricas("Random Forest Tuned", y_train, yp_tr, y_test, yp_te)
rmse_tr, mae_tr, r2_tr, rmse_te, mae_te, r2_te, y_re, y_pr = res
 
resultados.append({"Modelo": "Random Forest Tuned",
                   "RMSE_train": rmse_tr, "MAE_train": mae_tr, "R2_train": r2_tr,
                   "RMSE_test":  rmse_te, "MAE_test":  mae_te, "R2_test":  r2_te})
resultados_cv.append(evaluar_cv(mejor_rf, X, y, "Random Forest Tuned"))
mostrar_importancias(mejor_rf, "Random Forest Tuned", vars_numericas, vars_categoricas)
graficar_real_vs_pred(y_re, y_pr, "Random Forest Tuned - Real vs Predicho")
 
# ---------------------------------------------------------
# 7.3 Gradient Boosting con hiperparámetros
# ---------------------------------------------------------
print("\n>>> Buscando hiperparámetros para Gradient Boosting...")
 
param_dist_gb = {
    "modelo__n_estimators":    [200, 300, 500],
    "modelo__learning_rate":   [0.01, 0.03, 0.05, 0.1],
    "modelo__max_depth":       [2, 3, 4],
    "modelo__min_samples_split": [2, 5, 10],
    "modelo__min_samples_leaf":  [1, 2, 4],
    "modelo__subsample":       [0.7, 0.8, 0.9],
    "modelo__max_features":    ["sqrt", "log2", None]
}
 
pipe_gb = Pipeline([
    ("prep",   preprocessor),
    ("modelo", GradientBoostingRegressor(random_state=42))
])
 
search_gb = RandomizedSearchCV(
    estimator=pipe_gb,
    param_distributions=param_dist_gb,
    n_iter=20, scoring="r2", cv=5,
    verbose=1, random_state=42, n_jobs=-1
)
search_gb.fit(X_train, y_train)
mejor_gb = search_gb.best_estimator_
 
print("Mejores parámetros GB:", search_gb.best_params_)
 
yp_tr = mejor_gb.predict(X_train)
yp_te = mejor_gb.predict(X_test)
 
res = imprimir_metricas("Gradient Boosting Tuned", y_train, yp_tr, y_test, yp_te)
rmse_tr, mae_tr, r2_tr, rmse_te, mae_te, r2_te, y_re, y_pr = res
 
resultados.append({"Modelo": "Gradient Boosting Tuned",
                   "RMSE_train": rmse_tr, "MAE_train": mae_tr, "R2_train": r2_tr,
                   "RMSE_test":  rmse_te, "MAE_test":  mae_te, "R2_test":  r2_te})
resultados_cv.append(evaluar_cv(mejor_gb, X, y, "Gradient Boosting Tuned"))
mostrar_importancias(mejor_gb, "Gradient Boosting Tuned", vars_numericas, vars_categoricas)
graficar_real_vs_pred(y_re, y_pr, "Gradient Boosting Tuned - Real vs Predicho")
 
# ---------------------------------------------------------
# 7.4 XGBoost (si está instalado)
# ---------------------------------------------------------
try:
    from xgboost import XGBRegressor
 
    print("\n>>> Buscando hiperparámetros para XGBoost...")
 
    param_dist_xgb = {
        "modelo__n_estimators":   [200, 400, 600],
        "modelo__max_depth":      [3, 4, 5, 6],
        "modelo__learning_rate":  [0.01, 0.03, 0.05, 0.1],
        "modelo__subsample":      [0.7, 0.8, 0.9],
        "modelo__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "modelo__min_child_weight": [1, 3, 5],
        "modelo__reg_alpha":      [0, 0.1, 1.0],
        "modelo__reg_lambda":     [1.0, 5.0, 10.0]
    }
 
    pipe_xgb = Pipeline([
        ("prep",   preprocessor),
        ("modelo", XGBRegressor(objective="reg:squarederror", random_state=42,
                                 n_jobs=-1, verbosity=0))
    ])
 
    search_xgb = RandomizedSearchCV(
        estimator=pipe_xgb,
        param_distributions=param_dist_xgb,
        n_iter=20, scoring="r2", cv=5,
        verbose=1, random_state=42, n_jobs=-1
    )
    search_xgb.fit(X_train, y_train)
    mejor_xgb = search_xgb.best_estimator_
 
    print("Mejores parámetros XGBoost:", search_xgb.best_params_)
 
    yp_tr = mejor_xgb.predict(X_train)
    yp_te = mejor_xgb.predict(X_test)
 
    res = imprimir_metricas("XGBoost Tuned", y_train, yp_tr, y_test, yp_te)
    rmse_tr, mae_tr, r2_tr, rmse_te, mae_te, r2_te, y_re, y_pr = res
 
    resultados.append({"Modelo": "XGBoost Tuned",
                       "RMSE_train": rmse_tr, "MAE_train": mae_tr, "R2_train": r2_tr,
                       "RMSE_test":  rmse_te, "MAE_test":  mae_te, "R2_test":  r2_te})
    resultados_cv.append(evaluar_cv(mejor_xgb, X, y, "XGBoost Tuned"))
    mostrar_importancias(mejor_xgb, "XGBoost Tuned", vars_numericas, vars_categoricas)
    graficar_real_vs_pred(y_re, y_pr, "XGBoost Tuned - Real vs Predicho")
 
except ImportError:
    print("\nXGBoost no instalado. Instalar con: pip install xgboost")
 
# =========================================================
# 8. Tablas finales y gráficos comparativos
# =========================================================
df_res = pd.DataFrame(resultados).sort_values("R2_test", ascending=False)
df_cv  = pd.DataFrame(resultados_cv).sort_values("CV_R2_mean", ascending=False)
 
print("\n" + "="*60)
print("RESULTADOS TRAIN / TEST")
print("="*60)
print(df_res.to_string(index=False))
 
print("\n" + "="*60)
print("VALIDACIÓN CRUZADA")
print("="*60)
print(df_cv.to_string(index=False))
 
df_res.to_csv("resultados_modelos_mejorado.csv", index=False, encoding="utf-8-sig")
df_cv.to_csv("resultados_cv_mejorado.csv",       index=False, encoding="utf-8-sig")
 
# Gráficos comparativos
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
 
axes[0].bar(df_res["Modelo"], df_res["R2_test"])
axes[0].set_title("R² en prueba")
axes[0].set_ylabel("R²")
axes[0].tick_params(axis="x", rotation=25)
 
axes[1].bar(df_res["Modelo"], df_res["RMSE_test"])
axes[1].set_title("RMSE en prueba")
axes[1].set_ylabel("RMSE")
axes[1].tick_params(axis="x", rotation=25)
 
axes[2].bar(df_cv["Modelo"], df_cv["CV_R2_mean"])
axes[2].set_title("CV R² promedio")
axes[2].set_ylabel("CV R²")
axes[2].tick_params(axis="x", rotation=25)
 
plt.tight_layout()
ruta_comp = os.path.join(CARPETA_GRAFICOS, "comparacion_modelos.png")
plt.savefig(ruta_comp, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nGráficos guardados en: {CARPETA_GRAFICOS}/")