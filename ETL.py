
import pandas as pd
import unicodedata
import re
import os

# Cargar los datos
BASE_DIR = '.'

ruta_rend = os.path.join(BASE_DIR, 'data_rendimientos', 'rendimiento_municipios_2007_2024.csv')
ruta_clima = os.path.join(BASE_DIR, 'data_clima', 'clima_anual_municipio.csv')
ruta_altitud = os.path.join(BASE_DIR, 'data_altitud', 'altitud_municipios.csv')

df_rendimientos = pd.read_csv(ruta_rend)
df_clima = pd.read_csv(ruta_clima)
df_altitud = pd.read_csv(ruta_altitud)


# función limpieza
def limpiar_municipio(valor):
    if pd.isna(valor):
        return None
    valor = str(valor).strip().upper()
    
    # Quitar tildes
    valor = ''.join(
        c for c in unicodedata.normalize('NFD', valor)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Quitar caracteres especiales, dejando letras, números y espacios
    valor = re.sub(r'[^A-Z0-9 ]', ' ', valor)
    
    # Quitar espacios múltiples
    valor = re.sub(r'\s+', ' ', valor).strip()
    
    return valor

# Limpiar columna departamento y municipio
df_rendimientos["Departamento"] = df_rendimientos["Departamento"].apply(limpiar_municipio)
df_rendimientos["Municipio"] = df_rendimientos["Municipio"].apply(limpiar_municipio)
df_clima["departamento"] = df_clima["departamento"].apply(limpiar_municipio)
df_clima["municipio"] = df_clima["municipio"].apply(limpiar_municipio)
df_altitud["departamento"] = df_altitud["departamento"].apply(limpiar_municipio)
df_altitud["municipio"] = df_altitud["municipio"].apply(limpiar_municipio)

#  Renombrar columnas para unificar
df_clima = df_clima.rename(columns={
    "anio": "Año",
    "departamento": "Departamento",
    "municipio": "Municipio",

})

df_altitud = df_altitud.rename(columns={
    "departamento": "Departamento",
    "municipio": "Municipio",

})

# Selección de columnas
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
    "Evaporación potencial acumulada anual (mm/año)"
]].copy()

df_altitud_final = df_altitud[[
    "Departamento",
    "Municipio",
    "altitud_media_m"
]].copy()


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


# Unir df rendimientos, clima y altitud
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

# Mostrar cantidad de datos faltantes por columna
faltantes = df_base_final.isna().sum().sort_values(ascending=False)
print("Datos faltantes por columna:")
print(faltantes)

# Eliminar filas que tengan faltantes en las variables clave
df_base_final = df_base_final.dropna(subset=[
    "Precipitación acumulada anual (mm/año)",
    "Temperatura media anual (°C)",
    "Máximo de la temperatura media mensual (°C)",
    "Mínimo de la temperatura media mensual (°C)",
    "Humedad relativa media anual (%)",
    "Radiación solar acumulada anual (MJ/m²/año)",
    "Humedad volumétrica media anual del suelo capa 1 (m³/m³)",
    "Humedad volumétrica media anual del suelo capa 2 (m³/m³)",
    "Evaporación potencial acumulada anual (mm/año)"
])

# Verificar
print(df_base_final.isna().sum())
print(df_base_final.shape)

df_base_final.to_csv("base_final.csv", index=False, encoding="utf-8-sig")