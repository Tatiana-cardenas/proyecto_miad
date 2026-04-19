import pandas as pd
import unicodedata
import re
import os

# 1. Cargar los datos
BASE_DIR = "."

ruta_rend = os.path.join(BASE_DIR, "data_rendimientos", "rendimiento_municipios_2007_2024.csv")
ruta_clima = os.path.join(BASE_DIR, "data_clima", "clima_anual_municipio.csv")
ruta_altitud = os.path.join(BASE_DIR, "data_altitud", "altitud_municipios.csv")

df_rendimientos = pd.read_csv(ruta_rend)
df_clima = pd.read_csv(ruta_clima)
df_altitud = pd.read_csv(ruta_altitud)

# 2. Función de limpieza
def limpiar_municipio(valor):
    if pd.isna(valor):
        return None

    valor = str(valor).strip().upper()

    valor = "".join(
        c for c in unicodedata.normalize("NFD", valor)
        if unicodedata.category(c) != "Mn"
    )

    valor = re.sub(r"[^A-Z0-9 ]", " ", valor)
    valor = re.sub(r"\s+", " ", valor).strip()

    return valor

# 3. Limpiar columnas de departamento y municipio
df_rendimientos["Departamento"] = df_rendimientos["Departamento"].apply(limpiar_municipio)
df_rendimientos["Municipio"] = df_rendimientos["Municipio"].apply(limpiar_municipio)

df_clima["departamento"] = df_clima["departamento"].apply(limpiar_municipio)
df_clima["municipio"] = df_clima["municipio"].apply(limpiar_municipio)

df_altitud["departamento"] = df_altitud["departamento"].apply(limpiar_municipio)
df_altitud["municipio"] = df_altitud["municipio"].apply(limpiar_municipio)

# 4. Renombrar columnas para unificar
df_clima = df_clima.rename(columns={
    "anio": "Año",
    "departamento": "Departamento",
    "municipio": "Municipio",
})

df_altitud = df_altitud.rename(columns={
    "departamento": "Departamento",
    "municipio": "Municipio",
})

# 5. Selección de columnas
df_rendimientos_final = df_rendimientos[[
    "Departamento",
    "Municipio",
    "Año",
    "Área Cosechada",
    "Área Sembrada",
    "Producción",
    "Rendimiento"
]].copy()

df_clima_final = df_clima[[
    "Departamento",
    "Municipio",
    "Año",
    "Precipitación acumulada anual (mm/año)",
    "Temperatura media anual (°C)",
    "Máximo de la temperatura media mensual (°C)",
    "Mínimo de la temperatura media mensual (°C)",
    "Humedad relativa media anual (%)",
    "Radiación solar acumulada anual (MJ/m²/año)",
    "Humedad volumétrica media anual del suelo capa 1 (m³/m³)",
    "Humedad volumétrica media anual del suelo capa 2 (m³/m³)",
    "Evaporación potencial acumulada anual (mm/año)",
    "SPI3_mean_anual",
    "SPI3_min_anual",
    "SPI3_meses_bajo_m1",
    "SPI6_mean_anual",
    "SPI6_min_anual",
    "SPI6_meses_bajo_m1",
    "SPI12_mean_anual",
    "SPI12_min_anual",
    "SPI12_meses_bajo_m1",
    "SPI3_dic",
    "SPI6_dic",
    "SPI12_dic"
]].copy()

df_altitud_final = df_altitud[[
    "Departamento",
    "Municipio",
    "altitud_media_m"
]].copy()

# 6. Homologación de municipios
homologacion = {
    "DON MATIAS": "DONMATIAS",
    "CARMEN DE VIBORAL": "EL CARMEN DE VIBORAL",
    "PUEBLO RICO": "PUEBLORRICO",
    "SAN VICENTE": "SAN VICENTE FERRER",
    "SANTAFE DE ANTIOQUIA": "SANTA FE DE ANTIOQUIA",
    "MONTANITA": "LA MONTANITA",
    "PIENDAMO": "PIENDAMO TUNIA",
    "CARMEN DEL ATRATO": "EL CARMEN DE ATRATO",
    "HATO NUEVO": "HATONUEVO",
    "LA URIBE": "URIBE",
    "VISTA HERMOSA": "VISTAHERMOSA",
    "DOS QUEBRADAS": "DOSQUEBRADAS",
    "ARMERO GUAYABAL": "ARMERO",
    "CAROLINA": "CAROLINA DEL PRINCIPE"
}

df_rendimientos_final["Municipio"] = df_rendimientos_final["Municipio"].replace(homologacion)
df_clima_final["Municipio"] = df_clima_final["Municipio"].replace(homologacion)
df_altitud_final["Municipio"] = df_altitud_final["Municipio"].replace(homologacion)

# 7. Unir rendimientos, clima y altitud
df_base_final = df_rendimientos_final.merge(
    df_clima_final,
    on=["Departamento", "Municipio", "Año"],
    how="left"
).merge(
    df_altitud_final,
    on=["Departamento", "Municipio"],
    how="left"
)

print(df_base_final.shape)

# 8. Revisar faltantes por columna
faltantes = df_base_final.isna().sum().sort_values(ascending=False)
print("Datos faltantes por columna:")
print(faltantes)

# 9. Eliminar filas con faltantes en variables climáticas y altitud
df_base_final = df_base_final.dropna(subset=[
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
])

print(df_base_final.isna().sum())
print(df_base_final.shape)

# 10. Limpieza de rendimiento
df = df_base_final.copy()

df = df[df["Rendimiento"] > 0].copy()

p995 = df["Rendimiento"].quantile(0.995)
df = df[df["Rendimiento"] <= p995].copy()
print(f"Tras limpieza de rendimientos: {df.shape}")

# 11. Variables de rezago
df = df.sort_values(["Departamento", "Municipio", "Año"]).reset_index(drop=True)

grupo = df.groupby(["Departamento", "Municipio"])["Rendimiento"]

df["Rendimiento_lag1"] = grupo.shift(1)
df["Rendimiento_lag2"] = grupo.shift(2)
df["Rendimiento_lag3"] = grupo.shift(3)

df["Rendimiento_rolling3"] = (
    df.groupby(["Departamento", "Municipio"])["Rendimiento"]
      .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)

media_mun = (
    df.groupby(["Departamento", "Municipio"])["Rendimiento"]
      .transform(lambda x: x.shift(1).expanding().mean())
)

df["Rendimiento_vs_media_mun"] = df["Rendimiento_lag1"] - media_mun

df = df.dropna(subset=["Rendimiento_lag1"]).copy()

print(f"Tras limpieza Rendimiento_lag1: {df.shape}")

# 12. Exportar base final
df.to_csv("base_final.csv", index=False, encoding="utf-8-sig")

# 13. Filtrar y exportar base final por departamentos de NARINO y CUNDINAMARCA
df_filtrado = df[
    df["Departamento"].isin(["NARINO", "CUNDINAMARCA"])
].copy()

df_filtrado.to_csv("base_final_narino_cundinamarca.csv", index=False, encoding="utf-8-sig")


print(f"Tras filtrar departamentos de Narino y Cundinamarca: {df_filtrado.shape}")

