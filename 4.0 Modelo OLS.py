import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# 1. Configuración

carpeta_salida = "graficos_modelos"
carpeta_ols = os.path.join(carpeta_salida, "ols_escenarios")
carpeta_graficos = os.path.join(carpeta_ols, "graficos")

os.makedirs(carpeta_ols, exist_ok=True)
os.makedirs(carpeta_graficos, exist_ok=True)

ruta_archivo = "base_final.csv"
target = "Rendimiento"
columna_tiempo = "Año"

# 2. Carga y validación de la base

df = pd.read_csv(ruta_archivo, encoding="utf-8-sig")
print(f"Base cargada: {df.shape[0]} filas x {df.shape[1]} columnas")

if target not in df.columns:
    raise ValueError(f"La variable objetivo '{target}' no existe en la base.")

if columna_tiempo not in df.columns:
    raise ValueError(f"La columna temporal '{columna_tiempo}' no existe en la base.")

# 3. Definición de variables y escenarios

variables_climaticas = [
    "Precipitación acumulada anual (mm/año)",
    "Temperatura media anual (°C)",
    "Máximo de la temperatura media mensual (°C)",
    "Mínimo de la temperatura media mensual (°C)",
    "Humedad relativa media anual (%)",
    "Radiación solar acumulada anual (MJ/m²/año)"
]

variables_geo = ["altitud_media_m"]
variables_spi = ["SPI3_dic", "SPI6_dic", "SPI12_dic"]
variables_lags = ["Rendimiento_lag1", "Rendimiento_rolling3"]
variables_categoricas = ["Departamento", "Municipio"]

def filtrar_variables(lista, columnas_df):
    return [col for col in lista if col in columnas_df]

variables_climaticas = filtrar_variables(variables_climaticas, df.columns)
variables_geo = filtrar_variables(variables_geo, df.columns)
variables_spi = filtrar_variables(variables_spi, df.columns)
variables_lags = filtrar_variables(variables_lags, df.columns)
variables_categoricas = filtrar_variables(variables_categoricas, df.columns)

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
    nombre: filtrar_variables(vars_escenario, df.columns)
    for nombre, vars_escenario in escenarios_variables.items()
    if len(filtrar_variables(vars_escenario, df.columns)) > 0
}

print("\nEscenarios definidos:")
print(list(escenarios_variables.keys()))

# 4. Función para guardar la tabla comparativa como imagen

def guardar_tabla_comparativa_como_imagen(df_tabla, carpeta):
    filas, columnas = df_tabla.shape
    alto = max(3, filas * 0.5 + 1)
    ancho = max(10, columnas * 1.5)

    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.axis("off")

    tabla = ax.table(
        cellText=df_tabla.values,
        colLabels=df_tabla.columns,
        loc="center",
        cellLoc="center"
    )

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1.1, 1.3)

    for (fila, col), celda in tabla.get_celld().items():
        if fila == 0:
            celda.set_text_props(weight="bold")
            celda.set_facecolor("#D9EAF7")

    plt.title("Tabla comparativa de escenarios OLS", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, "tabla_comparativa_escenarios.png"), dpi=150, bbox_inches="tight")
    plt.close()

# 5. Ajuste de modelos OLS por escenario

resultados_ols_resumen = []

for nombre_escenario, variables in escenarios_variables.items():
    print(f"\nEscenario: {nombre_escenario}")
    print(f"Variables incluidas: {len(variables)}")

    columnas_modelo = variables + [target, columna_tiempo]
    df_esc = df[columnas_modelo].copy().sort_values(columna_tiempo).reset_index(drop=True)

    X_esc = pd.get_dummies(df_esc[variables], drop_first=True)

    for col in X_esc.columns:
        if X_esc[col].dtype == "bool":
            X_esc[col] = X_esc[col].astype(int)

    X_esc = X_esc.apply(pd.to_numeric, errors="coerce").astype(float)
    y_esc = pd.to_numeric(df_esc[target], errors="coerce").astype(float)

    base_ols = pd.concat([X_esc, y_esc.rename(target)], axis=1).dropna().copy()

    if base_ols.shape[0] == 0:
        print("No hay datos suficientes después de eliminar nulos")
        continue

    X_esc = base_ols.drop(columns=[target])
    y_esc = base_ols[target]

    if X_esc.shape[1] == 0:
        print("No hay variables disponibles para este escenario")
        continue

    cols_no_num = X_esc.select_dtypes(exclude=["number"]).columns.tolist()
    if cols_no_num:
        print(f"Se omitió el escenario por columnas no numéricas: {cols_no_num}")
        continue

    X_con_constante = sm.add_constant(X_esc, has_constant="add").astype(float)

    try:
        modelo_ols = sm.OLS(y_esc, X_con_constante).fit()
    except Exception as e:
        print(f"Error ajustando el modelo OLS: {e}")
        continue

    conf_int = modelo_ols.conf_int()
    coef_df = pd.DataFrame({
        "Variable": modelo_ols.params.index,
        "Coeficiente": modelo_ols.params.values.round(6),
        "Std_Error": modelo_ols.bse.values.round(6),
        "t_value": modelo_ols.tvalues.values.round(4),
        "p_value": modelo_ols.pvalues.values.round(4),
        "Significativo_5pct": (modelo_ols.pvalues.values < 0.05).astype(int),
        "IC_2.5%": conf_int[0].values.round(6),
        "IC_97.5%": conf_int[1].values.round(6)
    }).sort_values("p_value", ascending=True).reset_index(drop=True)

    n_sig = coef_df[coef_df["Significativo_5pct"] == 1].shape[0]

    resultados_ols_resumen.append({
        "Escenario": nombre_escenario,
        "N_obs": int(modelo_ols.nobs),
        "N_variables": int(X_esc.shape[1]),
        "R2": round(modelo_ols.rsquared, 4),
        "R2_ajustado": round(modelo_ols.rsquared_adj, 4),
        "AIC": round(modelo_ols.aic, 2),
        "BIC": round(modelo_ols.bic, 2),
        "F_pvalue": round(float(modelo_ols.f_pvalue), 6) if modelo_ols.f_pvalue is not None else np.nan,
        "Variables_sig_5pct": n_sig
    })

# 6. Tabla comparativa de escenarios

if resultados_ols_resumen:
    df_resumen = pd.DataFrame(resultados_ols_resumen)
    df_resumen = df_resumen.sort_values(by=["R2_ajustado", "AIC"], ascending=[False, True]).reset_index(drop=True)

    print("\nTabla comparativa de escenarios:")
    print(df_resumen.to_string(index=False))

    guardar_tabla_comparativa_como_imagen(df_resumen, carpeta_graficos)

print(f"\nImagen guardada en: {carpeta_graficos}")