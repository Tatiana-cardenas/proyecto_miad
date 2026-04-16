import calendar
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping

# Rutas
NC_PATH = r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_era5_land\era5_land_mensual_2007_2024.nc"
SHP_PATH = r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_municipios\MGN_MPIO_POLITICO.shp"
OUT_CSV = r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_clima\clima_anual_municipio.csv"

# Leer municipios
gdf = gpd.read_file(SHP_PATH)

DEPTO = "DPTO_CNMBR"
MPIO = "MPIO_CNMBR"
COD = "MPIO_CCDGO"

departamentos = [
    "ANTIOQUIA",
    "NORTE DE SANTANDER",
    "META",
    "HUILA",
    "CUNDINAMARCA",
    "CESAR",
    "SANTANDER",
    "CALDAS",
    "CASANARE",
    "NARIÑO",
    "VALLE DEL CAUCA",
    "CAUCA",
    "BOYACA",
    "TOLIMA",
    "RISARALDA",
    "MAGDALENA",
    "QUINDIO",
    "LA GUAJIRA",
    "CAQUETA",
    "CHOCO",
    "SUCRE",
    "GUAVIARE",
    "CORDOBA",
    "PUTUMAYO",
    "BOLIVAR",
    "ARAUCA"
]

gdf = gdf.to_crs("EPSG:4326")
gdf = gdf[gdf[DEPTO].str.strip().str.upper().isin(departamentos)].copy()

print("Municipios filtrados:", len(gdf))
print(gdf[[DEPTO, MPIO, COD]].head())

# Leer NETCDF
ds = xr.open_dataset(NC_PATH)

print("Variables del netcdf:")
print(list(ds.data_vars))

print("Coordenadas del netcdf:")
print(list(ds.coords))
print(ds)

# Renombrar dimensiones/coordenadas espaciales
rename_dict = {}
if "longitude" in ds.coords:
    rename_dict["longitude"] = "x"
if "latitude" in ds.coords:
    rename_dict["latitude"] = "y"

if rename_dict:
    ds = ds.rename(rename_dict)

# Definir CRS y dimensiones espaciales
ds = ds.rio.write_crs("EPSG:4326")
ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")

print("Coordenadas después del ajuste:")
print(list(ds.coords))
print(ds)

# Identificar variables
tp_var = "tp"
t2m_var = "t2m"
d2m_var = "d2m"
ssrd_var = "ssrd"

# Identificar tiempo
if "time" in ds.coords:
    time_name = "time"
elif "valid_time" in ds.coords:
    time_name = "valid_time"
elif "date" in ds.coords:
    time_name = "date"
else:
    raise ValueError(f"No se encontró coordenada temporal. Coordenadas disponibles: {list(ds.coords)}")

print("Coordenada temporal detectada:", time_name)

# Funciones
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

# Conversiones
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

# Precipitación mensual en mm
ds["precip_mm_mes"] = ds[tp_var] * 1000.0 * dias

# Radiación mensual
ds["rad_mes"] = (ds[ssrd_var] * dias) / 1_000_000

# Promedio por municipio
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

# Agregación anual
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
    )
    .rename(columns={
        "precip_mensual_sum": "Precipitación acumulada anual (mm/año)",
        "temperatura_mensual_mean": "Temperatura media anual (°C)",
        "temperatura_mensual_max": "Máximo de la temperatura media mensual (°C)",
        "temperatura_mensual_min": "Mínimo de la temperatura media mensual (°C)",
        "humedad_relativa_mensual_mean": "Humedad relativa media anual (%)",
        "radiacion_mensual_sum": "Radiación solar acumulada anual (MJ/m²/año)",
    })
)

# Exportar CSV
df_anual.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"Archivo exportado en: {OUT_CSV}")
print(df_anual.head())