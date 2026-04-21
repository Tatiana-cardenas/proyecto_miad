# 1. Carga de librerías

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")


# 2. Configuración general

ruta_archivo = "base_final_narino_cundinamarca.csv"

carpeta_graficos = "graficos_modelos"
os.makedirs(carpeta_graficos, exist_ok=True)

carpeta_ols = os.path.join(carpeta_graficos, "ols_escenarios")
os.makedirs(carpeta_ols, exist_ok=True)

carpeta_despliegue = "despliegue_tablero"
carpeta_modelo = os.path.join(carpeta_despliegue, "modelo")
os.makedirs(carpeta_modelo, exist_ok=True)

random_state = 42
n_splits = 5
target = "Rendimiento"
columna_tiempo = "Año"


# 3. Carga y validación de la base

df = pd.read_csv(ruta_archivo, encoding="utf-8-sig")
print("Base cargada:", df.shape)

if target not in df.columns:
    raise ValueError(f"La variable objetivo '{target}' no existe en la base.")

if columna_tiempo not in df.columns:
    raise ValueError(f"La columna temporal '{columna_tiempo}' no existe en la base.")

df = df.dropna(subset=[target, columna_tiempo]).copy()
df[columna_tiempo] = pd.to_numeric(df[columna_tiempo], errors="coerce")
df = df.dropna(subset=[columna_tiempo]).copy()
df[columna_tiempo] = df[columna_tiempo].astype(int)


# 4. Definición de variables

variables_climaticas = [
    "Precipitación acumulada anual (mm/año)",
    "Temperatura media anual (°C)",
    "Máximo de la temperatura media mensual (°C)",
    "Mínimo de la temperatura media mensual (°C)",
    "Humedad relativa media anual (%)",
    "Radiación solar acumulada anual (MJ/m²/año)"
]

variables_geo = [
    "altitud_media_m"
]

variables_spi = [
    "SPI1_floracion",
    "SPI1_llenado"
]

variables_lags = [
    "Rendimiento_lag1"
]

variables_categoricas = [
    "Departamento",
    "Municipio"
]

variables_climaticas = [col for col in variables_climaticas if col in df.columns]
variables_geo = [col for col in variables_geo if col in df.columns]
variables_spi = [col for col in variables_spi if col in df.columns]
variables_lags = [col for col in variables_lags if col in df.columns]
variables_categoricas = [col for col in variables_categoricas if col in df.columns]


# 5. Escenarios

escenarios_variables = {
    "clima": variables_climaticas,
    "clima_geo": variables_climaticas + variables_geo,
    "clima_spi": variables_climaticas + variables_spi,
    "clima_geo_spi": variables_climaticas + variables_geo + variables_spi,
    "clima_geo_lags": variables_climaticas + variables_geo + variables_lags,
    "clima_geo_cat": variables_climaticas + variables_geo + variables_categoricas,
    "completo": variables_climaticas + variables_geo + variables_spi + variables_lags + variables_categoricas
}

escenarios_variables = {
    nombre: [col for col in columnas if col in df.columns]
    for nombre, columnas in escenarios_variables.items()
}

escenarios_variables = {
    nombre: columnas for nombre, columnas in escenarios_variables.items()
    if len(columnas) > 0
}

print("\nEscenarios:")
for nombre, columnas in escenarios_variables.items():
    print(f"{nombre}: {len(columnas)} variables")


# 6. Métricas

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

scoring = {
    "rmse": make_scorer(rmse, greater_is_better=False),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "r2": make_scorer(r2_score)
}


# 7. Funciones auxiliares

def construir_preprocesador(X):
    columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
    columnas_categoricas = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformador_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformador_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocesador = ColumnTransformer(transformers=[
        ("num", transformador_numerico, columnas_numericas),
        ("cat", transformador_categorico, columnas_categoricas)
    ])

    return preprocesador


def evaluar_modelo_cv(nombre_modelo, pipeline, X, y, cv):
    resultados = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    return {
        "modelo": nombre_modelo,
        "rmse_cv_mean": -resultados["test_rmse"].mean(),
        "mae_cv_mean": -resultados["test_mae"].mean(),
        "r2_cv_mean": resultados["test_r2"].mean(),
        "rmse_train_cv_mean": -resultados["train_rmse"].mean(),
        "mae_train_cv_mean": -resultados["train_mae"].mean(),
        "r2_train_cv_mean": resultados["train_r2"].mean()
    }


def evaluar_en_test(modelo, X_train, X_test, y_train, y_test):
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    return {
        "rmse_train": rmse(y_train, y_pred_train),
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "r2_train": r2_score(y_train, y_pred_train),
        "rmse_test": rmse(y_test, y_pred_test),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "r2_test": r2_score(y_test, y_pred_test)
    }, y_pred_test


def preparar_datos_ols(X_train, X_test, y_train, y_test):
    X_train_ols = pd.get_dummies(X_train, drop_first=True)
    X_test_ols = pd.get_dummies(X_test, drop_first=True)

    X_train_ols, X_test_ols = X_train_ols.align(X_test_ols, join="left", axis=1, fill_value=0)

    X_train_ols = X_train_ols.apply(pd.to_numeric, errors="coerce").astype(float)
    X_test_ols = X_test_ols.apply(pd.to_numeric, errors="coerce").astype(float)
    y_train_ols = pd.to_numeric(y_train, errors="coerce").astype(float)
    y_test_ols = pd.to_numeric(y_test, errors="coerce").astype(float)

    base_train = pd.concat([X_train_ols, y_train_ols.rename(target)], axis=1).dropna().copy()
    if base_train.empty:
        return None, None, None, None

    X_train_ols = base_train.drop(columns=[target])
    y_train_ols = base_train[target]

    X_test_ols = X_test_ols.loc[y_test_ols.index].copy()
    base_test = pd.concat([X_test_ols, y_test_ols.rename(target)], axis=1).dropna().copy()
    if base_test.empty:
        return None, None, None, None

    X_test_ols = base_test.drop(columns=[target])
    y_test_ols = base_test[target]

    X_train_ols, X_test_ols = X_train_ols.align(X_test_ols, join="left", axis=1, fill_value=0)

    return X_train_ols, X_test_ols, y_train_ols, y_test_ols


def ajustar_evaluar_ols(X_train, X_test, y_train, y_test):
    datos_ols = preparar_datos_ols(X_train, X_test, y_train, y_test)

    if datos_ols[0] is None:
        return None

    X_train_ols, X_test_ols, y_train_ols, y_test_ols = datos_ols

    if X_train_ols.shape[1] == 0:
        return None

    X_train_const = sm.add_constant(X_train_ols, has_constant="add").astype(float)
    X_test_const = sm.add_constant(X_test_ols, has_constant="add").astype(float)

    X_train_const, X_test_const = X_train_const.align(X_test_const, join="left", axis=1, fill_value=0)

    modelo_ols = sm.OLS(y_train_ols, X_train_const).fit()

    y_pred_train = modelo_ols.predict(X_train_const)
    y_pred_test = modelo_ols.predict(X_test_const)

    return {
        "modelo_obj": modelo_ols,
        "X_train": X_train_ols,
        "X_test": X_test_ols,
        "y_train": y_train_ols,
        "y_test": y_test_ols,
        "y_pred_test": y_pred_test,
        "rmse_train": rmse(y_train_ols, y_pred_train),
        "mae_train": mean_absolute_error(y_train_ols, y_pred_train),
        "r2_train": r2_score(y_train_ols, y_pred_train),
        "rmse_test": rmse(y_test_ols, y_pred_test),
        "mae_test": mean_absolute_error(y_test_ols, y_pred_test),
        "r2_test": r2_score(y_test_ols, y_pred_test),
        "r2_ajustado_train": modelo_ols.rsquared_adj,
        "aic": modelo_ols.aic,
        "bic": modelo_ols.bic
    }


def graficar_real_vs_pred(y_true, y_pred, titulo, carpeta_salida, nombre_archivo):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)

    minimo = min(float(np.min(y_true)), float(np.min(y_pred)))
    maximo = max(float(np.max(y_true)), float(np.max(y_pred)))

    plt.plot([minimo, maximo], [minimo, maximo], linestyle="--")
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title(titulo)
    plt.tight_layout()

    ruta = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    print("Gráfico guardado en:", ruta)
    plt.close()


def guardar_tabla_como_imagen(df_tabla, carpeta_salida, nombre_archivo, titulo):
    fig, ax = plt.subplots(figsize=(14, max(3, len(df_tabla) * 0.6 + 1.5)))
    ax.axis("off")

    tabla_df = df_tabla.copy()

    for col in tabla_df.columns:
        if pd.api.types.is_numeric_dtype(tabla_df[col]):
            tabla_df[col] = tabla_df[col].round(4)

    tabla = ax.table(
        cellText=tabla_df.values,
        colLabels=tabla_df.columns,
        cellLoc="center",
        loc="center"
    )

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1.2, 1.4)

    for (fila, col), celda in tabla.get_celld().items():
        if fila == 0:
            celda.set_text_props(weight="bold")

    plt.title(titulo, fontsize=12, pad=15)
    plt.tight_layout()

    ruta = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    print("Tabla guardada en:", ruta)
    plt.close()


def obtener_modelos():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "Lasso": Lasso(random_state=random_state, max_iter=10000),
        "RandomForest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state)
    }


def obtener_espacios_busqueda():
    return {
        "Ridge": {
            "modelo__alpha": np.logspace(-3, 2, 30)
        },
        "Lasso": {
            "modelo__alpha": np.logspace(-5, 1, 40)
        },
        "RandomForest": {
            "modelo__n_estimators": [100, 200, 300, 500],
            "modelo__max_depth": [None, 5, 10, 15, 20],
            "modelo__min_samples_split": [2, 5, 10, 20],
            "modelo__min_samples_leaf": [1, 2, 4, 8],
            "modelo__max_features": ["sqrt", "log2", None]
        },
        "GradientBoosting": {
            "modelo__n_estimators": [100, 200, 300, 500],
            "modelo__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "modelo__max_depth": [2, 3, 4, 5],
            "modelo__min_samples_split": [2, 5, 10],
            "modelo__min_samples_leaf": [1, 2, 4],
            "modelo__subsample": [0.7, 0.8, 0.9, 1.0]
        }
    }


def construir_tabla_resumen_mejor_por_modelo(df_resultados):
    tabla = df_resultados[[
        "modelo", "escenario", "ajustado",
        "rmse_train", "mae_train", "r2_train",
        "rmse_test", "mae_test", "r2_test"
    ]].copy()

    tabla = tabla.sort_values(
        by=["r2_test", "rmse_test"],
        ascending=[False, True]
    ).head(5).reset_index(drop=True)

    tabla.columns = [
        "Modelo", "Escenario", "Ajustado",
        "RMSE_train", "MAE_train", "R2_train",
        "RMSE_test", "MAE_test", "R2_test"
    ]

    return tabla


# 8. Entrenamiento y evaluación

resultados_cv = []
resultados_test_modelos = []
resultados_ols_resumen = []

modelos = obtener_modelos()
espacios_busqueda = obtener_espacios_busqueda()

for nombre_escenario, variables in escenarios_variables.items():
    print("\nEscenario:", nombre_escenario)

    columnas_modelo = variables + [target]
    if columna_tiempo not in columnas_modelo:
        columnas_modelo.append(columna_tiempo)

    df_modelo = df[columnas_modelo].copy()
    df_modelo = df_modelo.sort_values(columna_tiempo).reset_index(drop=True)

    X = df_modelo[variables]
    y = df_modelo[target]

    anio_corte = int(df_modelo[columna_tiempo].quantile(0.80))

    mask_train = df_modelo[columna_tiempo] <= anio_corte
    mask_test = df_modelo[columna_tiempo] > anio_corte

    X_train = X.loc[mask_train].copy()
    X_test = X.loc[mask_test].copy()
    y_train = y.loc[mask_train].copy()
    y_test = y.loc[mask_test].copy()

    print("Train:", X_train.shape[0], "| Test:", X_test.shape[0])

    if X_train.empty or X_test.empty:
        print("No se pudo construir train y test para este escenario.")
        continue

    print("Probando OLS")
    try:
        salida_ols = ajustar_evaluar_ols(X_train, X_test, y_train, y_test)

        if salida_ols is not None:
            resultados_test_modelos.append({
                "escenario": nombre_escenario,
                "modelo": "OLS",
                "ajustado": "No",
                "variables": variables.copy(),
                "best_params": None,
                "anio_corte": anio_corte,
                "X_train": salida_ols["X_train"].copy(),
                "X_test": salida_ols["X_test"].copy(),
                "y_train": salida_ols["y_train"].copy(),
                "y_test": salida_ols["y_test"].copy(),
                "modelo_obj": salida_ols["modelo_obj"],
                "tipo_modelo": "OLS",
                "r2_ajustado_train": salida_ols["r2_ajustado_train"],
                "aic": salida_ols["aic"],
                "bic": salida_ols["bic"],
                "rmse_train": salida_ols["rmse_train"],
                "mae_train": salida_ols["mae_train"],
                "r2_train": salida_ols["r2_train"],
                "rmse_test": salida_ols["rmse_test"],
                "mae_test": salida_ols["mae_test"],
                "r2_test": salida_ols["r2_test"]
            })

            resultados_ols_resumen.append({
                "Escenario": nombre_escenario,
                "R2_train": salida_ols["r2_train"],
                "R2_ajustado_train": salida_ols["r2_ajustado_train"],
                "RMSE_test": salida_ols["rmse_test"],
                "MAE_test": salida_ols["mae_test"],
                "R2_test": salida_ols["r2_test"],
                "AIC": salida_ols["aic"],
                "BIC": salida_ols["bic"]
            })

    except Exception as e:
        print("Error en OLS:", e)

    if X_train.shape[0] <= n_splits:
        print("No hay suficientes datos para TimeSeriesSplit.")
        continue

    cv = TimeSeriesSplit(n_splits=n_splits)
    preprocesador = construir_preprocesador(X_train)

    for nombre_modelo, modelo in modelos.items():
        print("Probando", nombre_modelo)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocesador),
            ("modelo", modelo)
        ])

        try:
            resumen_cv = evaluar_modelo_cv(nombre_modelo, pipeline, X_train, y_train, cv)
            resumen_cv["escenario"] = nombre_escenario
            resumen_cv["ajustado"] = "No"
            resumen_cv["best_params"] = None
            resumen_cv["anio_corte"] = anio_corte
            resultados_cv.append(resumen_cv)
        except Exception as e:
            print("Error en validación cruzada:", e)
            continue

        if nombre_modelo in espacios_busqueda:
            try:
                busqueda = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=espacios_busqueda[nombre_modelo],
                    n_iter=20,
                    scoring="neg_root_mean_squared_error",
                    cv=cv,
                    random_state=random_state,
                    n_jobs=-1
                )

                busqueda.fit(X_train, y_train)
                mejor_pipeline = busqueda.best_estimator_

                resultados_test, y_pred_test = evaluar_en_test(
                    mejor_pipeline, X_train, X_test, y_train, y_test
                )

                resultados_test_modelos.append({
                    "escenario": nombre_escenario,
                    "modelo": nombre_modelo,
                    "ajustado": "Sí",
                    "variables": variables.copy(),
                    "best_params": busqueda.best_params_,
                    "anio_corte": anio_corte,
                    "X_train": X_train.copy(),
                    "X_test": X_test.copy(),
                    "y_train": y_train.copy(),
                    "y_test": y_test.copy(),
                    "modelo_obj": mejor_pipeline,
                    "tipo_modelo": "ML",
                    "rmse_train": resultados_test["rmse_train"],
                    "mae_train": resultados_test["mae_train"],
                    "r2_train": resultados_test["r2_train"],
                    "rmse_test": resultados_test["rmse_test"],
                    "mae_test": resultados_test["mae_test"],
                    "r2_test": resultados_test["r2_test"]
                })

            except Exception as e:
                print("Error en ajuste de hiperparámetros:", e)

        else:
            try:
                pipeline.fit(X_train, y_train)

                resultados_test, y_pred_test = evaluar_en_test(
                    pipeline, X_train, X_test, y_train, y_test
                )

                resultados_test_modelos.append({
                    "escenario": nombre_escenario,
                    "modelo": nombre_modelo,
                    "ajustado": "No",
                    "variables": variables.copy(),
                    "best_params": None,
                    "anio_corte": anio_corte,
                    "X_train": X_train.copy(),
                    "X_test": X_test.copy(),
                    "y_train": y_train.copy(),
                    "y_test": y_test.copy(),
                    "modelo_obj": pipeline,
                    "tipo_modelo": "ML",
                    "rmse_train": resultados_test["rmse_train"],
                    "mae_train": resultados_test["mae_train"],
                    "r2_train": resultados_test["r2_train"],
                    "rmse_test": resultados_test["rmse_test"],
                    "mae_test": resultados_test["mae_test"],
                    "r2_test": resultados_test["r2_test"]
                })

            except Exception as e:
                print("Error entrenando modelo:", e)


# 9. Consolidación de resultados

columnas_salida = [
    "escenario",
    "modelo",
    "ajustado",
    "variables",
    "best_params",
    "anio_corte",
    "rmse_train",
    "mae_train",
    "r2_train",
    "rmse_test",
    "mae_test",
    "r2_test",
    "tipo_modelo",
    "r2_ajustado_train",
    "aic",
    "bic"
]

df_resultados_test = pd.DataFrame([
    {k: v for k, v in registro.items() if k in columnas_salida}
    for registro in resultados_test_modelos
])

df_resultados_test = df_resultados_test.sort_values(
    by=["r2_test", "rmse_test"],
    ascending=[False, True]
).reset_index(drop=True)


# 10. Tabla resumen por modelo

tabla_resumen = construir_tabla_resumen_mejor_por_modelo(df_resultados_test)

print("\nCOMPARACIÓN MODELOS")
print(tabla_resumen.round(6).to_string(index=False))

guardar_tabla_como_imagen(
    df_tabla=tabla_resumen,
    carpeta_salida=carpeta_graficos,
    nombre_archivo="comparacion_modelos.png",
    titulo="Comparación modelos"
)

# 11. Tabla completa

tabla_completa = df_resultados_test[[
    "modelo",
    "escenario",
    "ajustado",
    "rmse_train",
    "mae_train",
    "r2_train",
    "rmse_test",
    "mae_test",
    "r2_test"
]].copy()

tabla_completa.columns = [
    "Modelo",
    "Escenario",
    "Ajustado",
    "RMSE_train",
    "MAE_train",
    "R2_train",
    "RMSE_test",
    "MAE_test",
    "R2_test"
]

print("\nTABLA COMPLETA DE MODELOS")
print(tabla_completa.round(6).to_string(index=False))

guardar_tabla_como_imagen(
    df_tabla=tabla_completa,
    carpeta_salida=carpeta_graficos,
    nombre_archivo="tabla_completa_modelos_ols_ml_narino_cundinamarca.png",
    titulo="Tabla completa de modelos OLS y ML"
)


# 12. Tabla OLS por escenario

if len(resultados_ols_resumen) > 0:
    df_resumen_ols = pd.DataFrame(resultados_ols_resumen)
    df_resumen_ols = df_resumen_ols.sort_values(
        by=["R2_test", "RMSE_test"],
        ascending=[False, True]
    ).reset_index(drop=True)

    print("\nTABLA OLS POR ESCENARIO")
    print(df_resumen_ols.round(6).to_string(index=False))

    guardar_tabla_como_imagen(
        df_tabla=df_resumen_ols,
        carpeta_salida=carpeta_ols,
        nombre_archivo="tabla_comparativa_escenarios_ols_narino_cundinamarca.png",
        titulo="Tabla comparativa de escenarios OLS"
    )


# 13. Mejor modelo

mejor_fila = df_resultados_test.iloc[0]

print("\nMEJOR MODELO")
print(mejor_fila)


# 14. Recuperar el mejor modelo

mejor_objeto = None

for registro in resultados_test_modelos:
    if (
        registro["escenario"] == mejor_fila["escenario"] and
        registro["modelo"] == mejor_fila["modelo"] and
        registro["ajustado"] == mejor_fila["ajustado"]
    ):
        mejor_objeto = registro
        break

if mejor_objeto is None:
    raise ValueError("No se pudo recuperar el mejor modelo.")

modelo_final = mejor_objeto["modelo_obj"]
X_train = mejor_objeto["X_train"]
X_test = mejor_objeto["X_test"]
y_train = mejor_objeto["y_train"]
y_test = mejor_objeto["y_test"]

if mejor_fila["modelo"] == "OLS":
    X_test_pred = sm.add_constant(X_test, has_constant="add").astype(float)
    y_pred_test = modelo_final.predict(X_test_pred)
else:
    y_pred_test = modelo_final.predict(X_test)


# 15. Gráfico real vs predicho

graficar_real_vs_pred(
    y_true=y_test,
    y_pred=y_pred_test,
    titulo=f"Real vs Predicho - {mejor_fila['modelo']} - {mejor_fila['escenario']}",
    carpeta_salida=carpeta_graficos,
    nombre_archivo="real_vs_predicho_narino_cundinamarca.png"
)


# 16. Mostrar parámetros del mejor modelo

print("\nPARÁMETROS DEL MEJOR MODELO")

if mejor_fila["modelo"] == "OLS":
    print("El mejor modelo fue OLS.")
    print("OLS no usa hiperparámetros como los modelos de machine learning.")
    print("Variables usadas:")
    print(mejor_objeto["X_train"].columns.tolist())
else:
    print("Modelo:", mejor_fila["modelo"])
    print("Escenario:", mejor_fila["escenario"])
    print("Ajustado:", mejor_fila["ajustado"])

    if mejor_objeto["best_params"] is not None:
        for clave, valor in mejor_objeto["best_params"].items():
            print(f"{clave}: {valor}")
    else:
        print("No hubo ajuste de hiperparámetros.")
        print(modelo_final.named_steps["modelo"].get_params())


# 17. Exportar el mejor modelo

ruta_modelo_joblib = os.path.join(carpeta_modelo, "modelo_narino_cundinamarca.joblib")
joblib.dump(modelo_final, ruta_modelo_joblib)

columnas_modelo = X_train.columns.tolist()
ruta_columnas = os.path.join(carpeta_modelo, "columnas_modelo_narino_cundinamarca.json")

with open(ruta_columnas, "w", encoding="utf-8") as f:
    json.dump(columnas_modelo, f, ensure_ascii=False, indent=4)

info_modelo = {
    "modelo": mejor_fila["modelo"],
    "escenario": mejor_fila["escenario"],
    "ajustado": mejor_fila["ajustado"],
    "anio_corte": int(mejor_fila["anio_corte"]),
    "rmse_train": float(mejor_fila["rmse_train"]),
    "mae_train": float(mejor_fila["mae_train"]),
    "r2_train": float(mejor_fila["r2_train"]),
    "rmse_test": float(mejor_fila["rmse_test"]),
    "mae_test": float(mejor_fila["mae_test"]),
    "r2_test": float(mejor_fila["r2_test"])
}

if "r2_ajustado_train" in mejor_fila.index and pd.notnull(mejor_fila["r2_ajustado_train"]):
    info_modelo["r2_ajustado_train"] = float(mejor_fila["r2_ajustado_train"])

if "aic" in mejor_fila.index and pd.notnull(mejor_fila["aic"]):
    info_modelo["aic"] = float(mejor_fila["aic"])

if "bic" in mejor_fila.index and pd.notnull(mejor_fila["bic"]):
    info_modelo["bic"] = float(mejor_fila["bic"])

ruta_info = os.path.join(carpeta_modelo, "info_modelo_narino_cundinamarca.json")

with open(ruta_info, "w", encoding="utf-8") as f:
    json.dump(info_modelo, f, ensure_ascii=False, indent=4)

print("\nArchivos exportados correctamente.")