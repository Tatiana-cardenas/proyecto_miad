
import pandas as pd
import unicodedata
import re

# Cargar los datos
df_rendimientos_cundinamarca = pd.read_csv('Rendimientos Cundinamarca 2007-2024.csv')
df_rendimientos_narino = pd.read_csv('Rendimientos Nariño 2007-2024.csv')
df_clima=pd.read_csv('clima_anual_municipio.csv')

# Unir los dos dataframes de rendimientos
df_rendimientos = pd.concat(
    [df_rendimientos_cundinamarca, df_rendimientos_narino],
    ignore_index=True
)

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

print(df_rendimientos.head())
print(df_clima.head())

# Limpiar columna departamento y municipio
df_rendimientos["Departamento"] = df_rendimientos["Departamento"].apply(limpiar_municipio)
df_rendimientos["Municipio"] = df_rendimientos["Municipio"].apply(limpiar_municipio)
df_clima["departamento"] = df_clima["departamento"].apply(limpiar_municipio)
df_clima["municipio"] = df_clima["municipio"].apply(limpiar_municipio)

#  Renombrar columnas para unificar
df_clima = df_clima.rename(columns={
    "anio": "Año",
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
    "Humedad relativa media anual (%)",
    "Radiación solar acumulada anual (MJ/m²/año)"
]].copy()


homologacion = {
    "EL TABLON": "EL TABLON DE GOMEZ",
    "TUMACO": "SAN ANDRES DE TUMACO"
}

df_rendimientos_final["Municipio"] = df_rendimientos["Municipio"].replace(homologacion)


print(
    sorted(set(df_rendimientos_final["Municipio"]) - set(df_clima_final["Municipio"]))
)

# Unir df rendimientos y clima
df_base_final = df_rendimientos_final.merge(
    df_clima_final,
    on=["Departamento", "Municipio", "Año"],
    how="left"
)

# Columnas climáticas
cols_clima = [
    "Precipitación acumulada anual (mm/año)",
    "Temperatura media anual (°C)",
    "Humedad relativa media anual (%)",
    "Radiación solar acumulada anual (MJ/m²/año)"
]

# Filas sin información climática
sin_clima = df_base_final[df_base_final[cols_clima].isna().all(axis=1)].copy()

# Contar combinaciones únicas departamento-municipio-año sin clima
faltantes_unicos = (
    sin_clima[["Departamento", "Municipio"]]
    .drop_duplicates()
    .sort_values(["Departamento", "Municipio"])
)

print("Cantidad de combinaciones Departamento Municipio sin información climática:")
print(faltantes_unicos.shape[0])

df_base_final = df_base_final[~df_base_final[cols_clima].isna().all(axis=1)]

# Filtrar años 2015-2023
df_base_final = df_base_final[
    (df_base_final["Año"] >= 2015) & (df_base_final["Año"] <= 2023)
]

# Calcular producción total por municipio dentro de cada departamento
top5_productores = (
    df_base_final
    .groupby(["Departamento", "Municipio"], as_index=False)["Producción"]
    .sum()
)

# Quedarse con los 5 mayores productores por departamento
top5_productores = (
    top5_productores
    .sort_values(["Departamento", "Producción"], ascending=[True, False])
    .groupby("Departamento")
    .head(5)
)

# Filtrar la base original solo con esos municipios
df_base_final_top5 = df_base_final.merge(
    top5_productores[["Departamento", "Municipio"]],
    on=["Departamento", "Municipio"],
    how="inner"
)

print(top5_productores)

# Exportar base final a CSV
df_base_final.to_csv("base_final.csv", index=False, encoding="utf-8-sig")