import calendar
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping

# =========================
# Rutas
# =========================
NC_PATH_CLIMA = r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_era5_land\era5_land_mensual_2007_2024.nc"
NC_PATH_SUELO = r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_era5_land\era5_land_suelo_2007_2024.nc"
SHP_PATH = r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_municipios\MGN_MPIO_POLITICO.shp"
OUT_CSV = r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_clima\clima_anual_municipio.csv"

# =========================
# Leer municipios
# =========================
gdf = gpd.read_file(SHP_PATH)

DEPTO = "DPTO_CNMBR"
MPIO = "MPIO_CNMBR"
COD = "MPIO_CCDGO"

gdf = gdf.to_crs("EPSG:4326")

print("Total municipios:", len(gdf))
print(gdf[[DEPTO, MPIO, COD]].head())

# =========================
# Leer NETCDFs
# =========================
ds_clima = xr.open_dataset(NC_PATH_CLIMA)
ds_suelo = xr.open_dataset(NC_PATH_SUELO)

print("Variables clima:", list(ds_clima.data_vars))
print("Variables suelo:", list(ds_suelo.data_vars))

# =========================
# Renombrar dimensiones espaciales
# =========================
def ajustar_coords(ds):
    rename_dict = {}
    if "longitude" in ds.coords:
        rename_dict["longitude"] = "x"
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "y"
    if rename_dict:
        ds = ds.rename(rename_dict)

    ds = ds.rio.write_crs("EPSG:4326")
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return ds

ds_clima = ajustar_coords(ds_clima)
ds_suelo = ajustar_coords(ds_suelo)

# =========================
# Unir datasets
# =========================
ds = xr.merge([ds_clima, ds_suelo], compat="override")

print("Variables combinadas:")
print(list(ds.data_vars))

# =========================
# Función para detectar nombres
# =========================
def encontrar_variable(ds, candidatos, nombre_amigable):
    for var in candidatos:
        if var in ds.data_vars:
            print(f"{nombre_amigable}: usando '{var}'")
            return var
    raise ValueError(
        f"No se encontró la variable para {nombre_amigable}. "
        f"Candidatas probadas: {candidatos}. "
        f"Variables disponibles: {list(ds.data_vars)}"
    )

tp_var = encontrar_variable(ds, ["tp", "total_precipitation"], "Precipitación")
t2m_var = encontrar_variable(ds, ["t2m", "2m_temperature"], "Temperatura 2m")
d2m_var = encontrar_variable(ds, ["d2m", "2m_dewpoint_temperature"], "Punto de rocío 2m")
ssrd_var = encontrar_variable(ds, ["ssrd", "surface_solar_radiation_downwards"], "Radiación solar")
swvl1_var = encontrar_variable(ds, ["swvl1", "volumetric_soil_water_layer_1"], "Humedad suelo capa 1")
swvl2_var = encontrar_variable(ds, ["swvl2", "volumetric_soil_water_layer_2"], "Humedad suelo capa 2")
pev_var = encontrar_variable(ds, ["pev", "potential_evaporation"], "Evaporación potencial")

# =========================
# Identificar tiempo
# =========================
if "time" in ds.coords:
    time_name = "time"
elif "valid_time" in ds.coords:
    time_name = "valid_time"
elif "date" in ds.coords:
    time_name = "date"
else:
    raise ValueError(f"No se encontró coordenada temporal. Coordenadas disponibles: {list(ds.coords)}")

print("Coordenada temporal detectada:", time_name)

# =========================
# Funciones auxiliares
# =========================
def kelvin_a_celsius(x):
    return x - 273.15

def humedad_relativa(t_c, td_c):
    a = 17.625
    b = 243.04
    es_t = np.exp((a * t_c) / (b + t_c))
    es_td = np.exp((a * td_c) / (b + td_c))
    rh = 100.0 * (es_td / es_t)
    return np.clip(rh, 0, 100)

def dias_del_mes(t):
    return calendar.monthrange(int(t.year), int(t.month))[1]

# =========================
# Conversiones
# =========================
ds["t2m_c"] = kelvin_a_celsius(ds[t2m_var])
ds["d2m_c"] = kelvin_a_celsius(ds[d2m_var])

ds["rh"] = xr.apply_ufunc(
    humedad_relativa,
    ds["t2m_c"],
    ds["d2m_c"],
    vectorize=True,
)

dias = xr.DataArray(
    [dias_del_mes(pd.Timestamp(t)) for t in pd.to_datetime(ds[time_name].values)],
    dims=[time_name],
    coords={time_name: ds[time_name].values},
)

# Precipitación mensual en mm/mes
ds["precip_mm_mes"] = ds[tp_var] * 1000.0 * dias

# Radiación mensual en MJ/m²/mes
ds["rad_mes"] = (ds[ssrd_var] * dias) / 1_000_000

# Evaporación potencial en mm/mes
ds["pev_mm_mes"] = np.abs(ds[pev_var]) * 1000.0 * dias

# =========================
# Promedio por municipio
# =========================
resultados = []

for _, row in gdf.iterrows():
    geom = [mapping(row.geometry)]

    try:
        clip_ds = ds.rio.clip(geom, gdf.crs, drop=True)

        if clip_ds.sizes.get("x", 0) == 0 or clip_ds.sizes.get("y", 0) == 0:
            print(f"Sin celdas para {row[MPIO]}")
            continue

        df_mpio = pd.DataFrame({
            "time": pd.to_datetime(clip_ds[time_name].values),
            "precip_mensual": clip_ds["precip_mm_mes"].mean(dim=("y", "x"), skipna=True).values,
            "temperatura_mensual": clip_ds["t2m_c"].mean(dim=("y", "x"), skipna=True).values,
            "humedad_relativa_mensual": clip_ds["rh"].mean(dim=("y", "x"), skipna=True).values,
            "radiacion_mensual": clip_ds["rad_mes"].mean(dim=("y", "x"), skipna=True).values,
            "humedad_suelo_capa_1_mensual": clip_ds[swvl1_var].mean(dim=("y", "x"), skipna=True).values,
            "humedad_suelo_capa_2_mensual": clip_ds[swvl2_var].mean(dim=("y", "x"), skipna=True).values,
            "evaporacion_potencial_mensual": clip_ds["pev_mm_mes"].mean(dim=("y", "x"), skipna=True).values,
        })

        df_mpio["departamento"] = row[DEPTO]
        df_mpio["municipio"] = row[MPIO]
        df_mpio["cod_mpio"] = row[COD]
        df_mpio["anio"] = df_mpio["time"].dt.year

        resultados.append(df_mpio)
        print(f"Procesado: {row[MPIO]}")

    except Exception as e:
        print(f"Error en {row[MPIO]}: {e}")

if len(resultados) == 0:
    raise ValueError("No se generaron resultados. Revisa el recorte espacial y las rutas.")

df_mensual = pd.concat(resultados, ignore_index=True)

# =========================
# Agregación anual
# =========================
df_anual = (
    df_mensual
    .groupby(["departamento", "municipio", "cod_mpio", "anio"], as_index=False)
    .agg(
        precip_mensual_sum=("precip_mensual", "sum"),
        temperatura_mensual_mean=("temperatura_mensual", "mean"),
        temperatura_mensual_max=("temperatura_mensual", "max"),
        temperatura_mensual_min=("temperatura_mensual", "min"),
        humedad_relativa_mensual_mean=("humedad_relativa_mensual", "mean"),
        radiacion_mensual_sum=("radiacion_mensual", "sum"),
        humedad_suelo_capa_1_mensual_mean=("humedad_suelo_capa_1_mensual", "mean"),
        humedad_suelo_capa_2_mensual_mean=("humedad_suelo_capa_2_mensual", "mean"),
        evaporacion_potencial_mensual_sum=("evaporacion_potencial_mensual", "sum"),
    )
    .rename(columns={
        "precip_mensual_sum": "Precipitación acumulada anual (mm/año)",
        "temperatura_mensual_mean": "Temperatura media anual (°C)",
        "temperatura_mensual_max": "Máximo de la temperatura media mensual (°C)",
        "temperatura_mensual_min": "Mínimo de la temperatura media mensual (°C)",
        "humedad_relativa_mensual_mean": "Humedad relativa media anual (%)",
        "radiacion_mensual_sum": "Radiación solar acumulada anual (MJ/m²/año)",
        "humedad_suelo_capa_1_mensual_mean": "Humedad volumétrica media anual del suelo capa 1 (m³/m³)",
        "humedad_suelo_capa_2_mensual_mean": "Humedad volumétrica media anual del suelo capa 2 (m³/m³)",
        "evaporacion_potencial_mensual_sum": "Evaporación potencial acumulada anual (mm/año)",
    })
)

df_anual.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"Archivo exportado en: {OUT_CSV}")
print(df_anual.head())
print(df_anual.columns.tolist())

