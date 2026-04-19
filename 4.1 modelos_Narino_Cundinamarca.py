# 1. Carga de librerías

import os
import json
import warnings
import numpy as np
import pandas as pd
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
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# 2. Configuración general

ruta_archivo = "base_final_narino_cundinamarca.csv"
carpeta_graficos = "graficos_modelos"
os.makedirs(carpeta_graficos, exist_ok=True)

random_state = 42
n_splits = 5
target = "Rendimiento"
columna_tiempo = "Año"


# 3. Carga y validación de la base

df = pd.read_csv(ruta_archivo, encoding="utf-8-sig")
print("Base cargada:", df.shape)
print(df.head())

if target not in df.columns:
    raise ValueError(f"La variable objetivo '{target}' no existe en la base.")

if columna_tiempo not in df.columns:
    raise ValueError(f"La columna temporal '{columna_tiempo}' no existe en la base.")

df = df.dropna(subset=[target, columna_tiempo]).copy()
df[columna_tiempo] = pd.to_numeric(df[columna_tiempo], errors="coerce")
df = df.dropna(subset=[columna_tiempo]).copy()
df[columna_tiempo] = df[columna_tiempo].astype(int)


# 4. Definición de grupos de variables

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

variables_SPI = [
    "SPI3_dic",
    "SPI6_dic",
    "SPI12_dic"
]

variables_lags = [
    "Rendimiento_lag1",
    "Rendimiento_rolling3"
]

variables_categoricas = [
    "Departamento",
    "Municipio"
]

variables_climaticas = [col for col in variables_climaticas if col in df.columns]
variables_geo = [col for col in variables_geo if col in df.columns]
variables_SPI = [col for col in variables_SPI if col in df.columns]
variables_lags = [col for col in variables_lags if col in df.columns]
variables_categoricas = [col for col in variables_categoricas if col in df.columns]


# 5. Construcción de escenarios

escenarios_variables = {
    "clima": variables_climaticas,
    "clima_geo": variables_climaticas + variables_geo,
    "clima_spi": variables_climaticas + variables_SPI,
    "clima_geo_spi": variables_climaticas + variables_geo + variables_SPI,
    "clima_geo_lags": variables_climaticas + variables_geo + variables_lags,
    "clima_geo_cat": variables_climaticas + variables_geo + variables_categoricas,
    "completo": variables_climaticas + variables_geo + variables_SPI + variables_lags + variables_categoricas
}

escenarios_variables = {
    k: [col for col in v if col in df.columns]
    for k, v in escenarios_variables.items()
}

escenarios_variables = {
    k: v for k, v in escenarios_variables.items() if len(v) > 0
}

print("\nEscenarios de variables:")
for nombre, vars_ in escenarios_variables.items():
    print(f"- {nombre}: {len(vars_)} variables")


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

    transformer_num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformer_cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", transformer_num, columnas_numericas),
        ("cat", transformer_cat, columnas_categoricas)
    ])

    return preprocessor, columnas_numericas, columnas_categoricas


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

    resumen = {
        "modelo": nombre_modelo,
        "rmse_cv_mean": -resultados["test_rmse"].mean(),
        "rmse_cv_std": resultados["test_rmse"].std(),
        "mae_cv_mean": -resultados["test_mae"].mean(),
        "r2_cv_mean": resultados["test_r2"].mean(),
        "r2_cv_std": resultados["test_r2"].std(),
        "rmse_train_mean": -resultados["train_rmse"].mean(),
        "mae_train_mean": -resultados["train_mae"].mean(),
        "r2_train_mean": resultados["train_r2"].mean()
    }

    return resumen


def evaluar_en_test(modelo_ajustado, X_train, X_test, y_train, y_test):
    y_pred_train = modelo_ajustado.predict(X_train)
    y_pred_test = modelo_ajustado.predict(X_test)

    resultados = {
        "rmse_train": rmse(y_train, y_pred_train),
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "r2_train": r2_score(y_train, y_pred_train),
        "rmse_test": rmse(y_test, y_pred_test),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "r2_test": r2_score(y_test, y_pred_test)
    }

    return resultados, y_pred_test


def graficar_real_vs_pred(y_true, y_pred, titulo, carpeta_salida, nombre_archivo):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)

    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title(titulo)
    plt.tight_layout()

    ruta_guardado = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    print(f"Gráfico guardado en: {ruta_guardado}")

    plt.close()


def calcular_importancia_permutacion(modelo, X_test, y_test):
    try:
        resultado = permutation_importance(
            modelo,
            X_test,
            y_test,
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1
        )

        importancias = pd.DataFrame({
            "Variable": X_test.columns,
            "Importancia_perm_mean": resultado.importances_mean,
            "Importancia_perm_std": resultado.importances_std
        })

        importancias = (
            importancias.groupby("Variable", as_index=False)[
                ["Importancia_perm_mean", "Importancia_perm_std"]
            ]
            .mean()
            .sort_values("Importancia_perm_mean", ascending=False)
            .reset_index(drop=True)
        )

        return importancias

    except Exception as e:
        print("No fue posible calcular la importancia por permutación:", e)
        return None


def obtener_modelos_base():
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


def construir_tabla_resumen_mejor_por_modelo(df_resultados_test):
    df_aux = df_resultados_test.copy()

    idx_mejores = (
        df_aux
        .sort_values(by=["modelo", "r2_test", "rmse_test"], ascending=[True, False, True])
        .groupby("modelo")
        .head(1)
        .index
    )

    tabla = df_aux.loc[idx_mejores, [
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

    tabla = tabla.rename(columns={
        "modelo": "Modelo",
        "escenario": "Escenario",
        "ajustado": "Ajustado",
        "rmse_train": "RMSE_train",
        "mae_train": "MAE_train",
        "r2_train": "R2_train",
        "rmse_test": "RMSE_test",
        "mae_test": "MAE_test",
        "r2_test": "R2_test"
    })

    tabla = tabla.sort_values(by=["R2_test", "RMSE_test"], ascending=[False, True]).reset_index(drop=True)

    return tabla


# 8. Entrenamiento y evaluación de escenarios

resultados_cv = []
resultados_test_modelos = []

modelos_base = obtener_modelos_base()
espacios_busqueda = obtener_espacios_busqueda()

for nombre_escenario, variables in escenarios_variables.items():
    print("\n" + "-" * 70)
    print(f"Escenario: {nombre_escenario}")
    print("-" * 70)

    variables = [v for v in variables if v in df.columns]

    if len(variables) == 0:
        print("Este escenario no tiene variables disponibles.")
        continue

    columnas_modelo = variables + [target]
    if columna_tiempo not in columnas_modelo:
        columnas_modelo.append(columna_tiempo)

    df_modelo = df[columnas_modelo].copy()
    df_modelo = df_modelo.sort_values(columna_tiempo).reset_index(drop=True)

    X = df_modelo[variables]
    y = df_modelo[target]

    anio_corte = int(df_modelo[columna_tiempo].quantile(0.80))
    print(f"Split temporal: train <= {anio_corte}, test > {anio_corte}")

    mask_train = df_modelo[columna_tiempo] <= anio_corte
    mask_test = df_modelo[columna_tiempo] > anio_corte

    X_train = X.loc[mask_train].copy()
    X_test = X.loc[mask_test].copy()
    y_train = y.loc[mask_train].copy()
    y_test = y.loc[mask_test].copy()

    print(f"Observaciones train: {X_train.shape[0]}")
    print(f"Observaciones test: {X_test.shape[0]}")

    if X_train.shape[0] == 0:
        print("El conjunto de entrenamiento quedó vacío.")
        continue

    if X_test.shape[0] == 0:
        print("El conjunto de prueba quedó vacío.")
        continue

    if X_train.shape[0] <= n_splits:
        print("No hay suficientes observaciones para TimeSeriesSplit.")
        continue

    cv = TimeSeriesSplit(n_splits=n_splits)
    preprocessor, cols_num, cols_cat = construir_preprocesador(X_train)

    print("Variables numéricas:", cols_num)
    print("Variables categóricas:", cols_cat)

    for nombre_modelo, modelo in modelos_base.items():
        print(f"\nProbando modelo: {nombre_modelo}")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("modelo", modelo)
        ])

        try:
            resumen_cv = evaluar_modelo_cv(
                nombre_modelo=nombre_modelo,
                pipeline=pipeline,
                X=X_train,
                y=y_train,
                cv=cv
            )

            resumen_cv["escenario"] = nombre_escenario
            resumen_cv["modelo_base"] = nombre_modelo
            resumen_cv["ajustado"] = "No"
            resumen_cv["best_params"] = None
            resumen_cv["anio_corte"] = anio_corte
            resultados_cv.append(resumen_cv)

        except Exception as e:
            print(f"Error en validación cruzada para {nombre_modelo}: {e}")
            continue

        if nombre_modelo in espacios_busqueda:
            print(f"Ajustando hiperparámetros para {nombre_modelo}")

            try:
                busqueda = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=espacios_busqueda[nombre_modelo],
                    n_iter=20,
                    scoring="neg_root_mean_squared_error",
                    cv=cv,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=0
                )

                busqueda.fit(X_train, y_train)
                mejor_pipeline = busqueda.best_estimator_

                resumen_ajustado = evaluar_modelo_cv(
                    nombre_modelo=nombre_modelo,
                    pipeline=mejor_pipeline,
                    X=X_train,
                    y=y_train,
                    cv=cv
                )

                resumen_ajustado["escenario"] = nombre_escenario
                resumen_ajustado["modelo_base"] = nombre_modelo
                resumen_ajustado["ajustado"] = "Sí"
                resumen_ajustado["best_params"] = json.dumps(busqueda.best_params_, ensure_ascii=False)
                resumen_ajustado["anio_corte"] = anio_corte
                resultados_cv.append(resumen_ajustado)

                resultados_test, y_pred_test = evaluar_en_test(
                    mejor_pipeline, X_train, X_test, y_train, y_test
                )

                registro = {
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
                    **resultados_test
                }

                resultados_test_modelos.append(registro)

            except Exception as e:
                print(f"Error en tuning para {nombre_modelo}: {e}")

        else:
            try:
                pipeline.fit(X_train, y_train)

                resultados_test, y_pred_test = evaluar_en_test(
                    pipeline, X_train, X_test, y_train, y_test
                )

                registro = {
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
                    **resultados_test
                }

                resultados_test_modelos.append(registro)

            except Exception as e:
                print(f"Error entrenando {nombre_modelo}: {e}")


# 9. Consolidación de resultados

df_resultados_cv = pd.DataFrame(resultados_cv)

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
    "r2_test"
]

df_resultados_test = pd.DataFrame([
    {k: v for k, v in registro.items() if k in columnas_salida}
    for registro in resultados_test_modelos
])

df_resultados_cv = df_resultados_cv.sort_values(
    by=["r2_cv_mean", "rmse_cv_mean"],
    ascending=[False, True]
).reset_index(drop=True)

df_resultados_test = df_resultados_test.sort_values(
    by=["r2_test", "rmse_test"],
    ascending=[False, True]
).reset_index(drop=True)


# 10. Tabla resumen

tabla_resumen = construir_tabla_resumen_mejor_por_modelo(df_resultados_test)

print("\nRESULTADOS TRAIN / TEST")
print("=" * 100)
print(tabla_resumen.round(6).to_string(index=False))


# 11. Selección del mejor modelo global

mejor_fila = df_resultados_test.sort_values(
    by=["r2_test", "rmse_test"],
    ascending=[False, True]
).iloc[0]

print("\nMejor modelo seleccionado:")
print(mejor_fila)


# 12. Mejor modelo

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

y_pred_test = modelo_final.predict(X_test)


# 13. Importancia de las variables del mejor modelo

print("\nImportancia por permutación del mejor modelo:")
importancias = calcular_importancia_permutacion(
    modelo=modelo_final,
    X_test=X_test,
    y_test=y_test
)

if importancias is not None:
    print(importancias.head(10).round(6).to_string(index=False))


# 14. Gráfico real vs predicho del mejor modelo

graficar_real_vs_pred(
    y_true=y_test,
    y_pred=y_pred_test,
    titulo=f"Real vs Predicho - {mejor_fila['modelo']} - {mejor_fila['escenario']}",
    carpeta_salida=carpeta_graficos,
    nombre_archivo="Real vs Predicho_nariño_cundinamarca.png"
)