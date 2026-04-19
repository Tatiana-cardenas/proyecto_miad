import calendar
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
from scipy.stats import gamma, norm, rankdata
from shapely.geometry import mapping

# 1. Rutas
NC_PATH_CLIMA = r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_era5_land\era5_land_mensual_2007_2024.nc"
NC_PATH_SUELO = r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_era5_land\era5_land_suelo_2007_2024.nc"
SHP_PATH = r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_municipios\MGN_MPIO_POLITICO.shp"

OUT_CSV_ANUAL = r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_clima\clima_anual_municipio.csv"

# 2. Leer municipios
gdf = gpd.read_file(SHP_PATH)

DEPTO = "DPTO_CNMBR"
MPIO = "MPIO_CNMBR"
COD = "MPIO_CCDGO"

gdf = gdf.to_crs("EPSG:4326")

# 3. Leer NETCDFs
ds_clima = xr.open_dataset(NC_PATH_CLIMA)
ds_suelo = xr.open_dataset(NC_PATH_SUELO)

# 4. Ajustar coordenadas espaciales
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

def nombre_coord_tiempo(ds):
    if "time" in ds.coords:
        return "time"
    elif "valid_time" in ds.coords:
        return "valid_time"
    elif "date" in ds.coords:
        return "date"
    else:
        raise ValueError(f"No se encontró coordenada temporal. Coordenadas disponibles: {list(ds.coords)}")

def normalizar_tiempo_a_mes(ds):
    tname = nombre_coord_tiempo(ds)

    fechas = pd.to_datetime(ds[tname].values).to_period("M").to_timestamp()
    ds = ds.assign_coords({tname: fechas})

    if pd.Index(fechas).duplicated().any():
        ds = ds.groupby(tname).mean()

    if tname != "time":
        ds = ds.rename({tname: "time"})

    return ds

ds_clima = ajustar_coords(ds_clima)
ds_suelo = ajustar_coords(ds_suelo)

ds_clima = normalizar_tiempo_a_mes(ds_clima)
ds_suelo = normalizar_tiempo_a_mes(ds_suelo)

# 5. Unir datasets
ds = xr.merge([ds_clima, ds_suelo], compat="override")

# 6. Función para detectar nombres
def encontrar_variable(ds, candidatos, nombre_amigable):
    for var in candidatos:
        if var in ds.data_vars:
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

time_name = "time"

# 7. Funciones auxiliares
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

# 8. Conversiones
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

ds["precip_mm_mes"] = ds[tp_var] * 1000.0 * dias
ds["rad_mes"] = (ds[ssrd_var] * dias) / 1_000_000
ds["pev_mm_mes"] = np.abs(ds[pev_var]) * 1000.0 * dias

# 9. Promedio por municipio
resultados = []

for _, row in gdf.iterrows():
    geom = [mapping(row.geometry)]

    try:
        clip_ds = ds.rio.clip(geom, gdf.crs, drop=True)

        if clip_ds.sizes.get("x", 0) == 0 or clip_ds.sizes.get("y", 0) == 0:
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
        df_mpio["mes"] = df_mpio["time"].dt.month

        resultados.append(df_mpio)

    except Exception:
        continue

if len(resultados) == 0:
    raise ValueError("No se generaron resultados. Revisa el recorte espacial y las rutas.")

df_mensual = pd.concat(resultados, ignore_index=True)

# 10. Consolidar una sola fila por municipio-mes
df_mensual = (
    df_mensual
    .groupby(["departamento", "municipio", "cod_mpio", "time", "anio", "mes"], as_index=False)
    .agg(
        precip_mensual=("precip_mensual", "mean"),
        temperatura_mensual=("temperatura_mensual", "mean"),
        humedad_relativa_mensual=("humedad_relativa_mensual", "mean"),
        radiacion_mensual=("radiacion_mensual", "mean"),
        humedad_suelo_capa_1_mensual=("humedad_suelo_capa_1_mensual", "mean"),
        humedad_suelo_capa_2_mensual=("humedad_suelo_capa_2_mensual", "mean"),
        evaporacion_potencial_mensual=("evaporacion_potencial_mensual", "mean"),
    )
    .sort_values(["departamento", "municipio", "time"])
    .reset_index(drop=True)
)

# 11. Limpieza numérica
cols_numericas = [
    "precip_mensual",
    "temperatura_mensual",
    "humedad_relativa_mensual",
    "radiacion_mensual",
    "humedad_suelo_capa_1_mensual",
    "humedad_suelo_capa_2_mensual",
    "evaporacion_potencial_mensual"
]

for col in cols_numericas:
    df_mensual[col] = pd.to_numeric(df_mensual[col], errors="coerce")

df_mensual["precip_mensual"] = df_mensual["precip_mensual"].clip(lower=0)

# 12. SPI robusto
def spi_referencia(fecha, precipitacion):
    return pd.DataFrame({
        "fecha": pd.to_datetime(fecha),
        "precipitacion": pd.to_numeric(precipitacion, errors="coerce")
    }).dropna()

def _spi_empirico(x):
    x = pd.Series(x)
    out = pd.Series(np.nan, index=x.index)

    valid = x.dropna()
    if len(valid) < 5:
        return out

    ranks = rankdata(valid, method="average")
    probs = (ranks - 0.44) / (len(valid) + 0.12)
    probs = np.clip(probs, 1e-8, 1 - 1e-8)

    out.loc[valid.index] = norm.ppf(probs)
    return out

def _ajustar_spi_una_escala(df, escala, referencia_mask=None, distribucion="Gamma"):
    df = df.copy().sort_values("fecha")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["precipitacion"] = pd.to_numeric(df["precipitacion"], errors="coerce").clip(lower=0)

    col_acum = f"precip_acum_{escala}"
    df[col_acum] = df["precipitacion"].rolling(window=escala, min_periods=escala).sum()
    df["spi"] = np.nan

    if referencia_mask is None:
        referencia_mask = pd.Series(True, index=df.index)

    datos_ref = df.loc[referencia_mask, col_acum].dropna()

    if len(datos_ref) < 10:
        df["spi"] = _spi_empirico(df[col_acum])
        out = df[["fecha"]].copy()
        out["escala"] = escala
        out["spi"] = df["spi"]
        return out

    if distribucion.lower() != "gamma":
        raise ValueError("Solo se soporta distribucion='Gamma'.")

    q = (datos_ref == 0).mean()
    datos_pos = datos_ref[datos_ref > 0]

    if len(datos_pos) < 8 or datos_pos.nunique() < 5:
        df["spi"] = _spi_empirico(df[col_acum])
        out = df[["fecha"]].copy()
        out["escala"] = escala
        out["spi"] = df["spi"]
        return out

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape, loc, scale = gamma.fit(datos_pos, floc=0)

        if not np.isfinite(shape) or not np.isfinite(scale) or shape <= 0 or scale <= 0:
            df["spi"] = _spi_empirico(df[col_acum])
        else:
            vals = df[col_acum]
            spi_vals = []

            for x in vals:
                if pd.isna(x):
                    spi_vals.append(np.nan)
                elif x <= 0:
                    h = q if q > 0 else 1e-8
                    h = np.clip(h, 1e-8, 1 - 1e-8)
                    spi_vals.append(norm.ppf(h))
                else:
                    g_x = gamma.cdf(x, a=shape, loc=0, scale=scale)
                    h = q + (1 - q) * g_x
                    h = np.clip(h, 1e-8, 1 - 1e-8)
                    spi_vals.append(norm.ppf(h))

            df["spi"] = spi_vals

    except Exception:
        df["spi"] = _spi_empirico(df[col_acum])

    out = df[["fecha"]].copy()
    out["escala"] = escala
    out["spi"] = df["spi"]
    return out

def spi_indice(fecha, precipitacion, escalas, referencia=None, distribucion="Gamma"):
    df = pd.DataFrame({
        "fecha": pd.to_datetime(fecha),
        "precipitacion": pd.to_numeric(precipitacion, errors="coerce")
    }).sort_values("fecha").reset_index(drop=True)

    if referencia is None:
        referencia_mask = pd.Series(True, index=df.index)
    elif isinstance(referencia, pd.DataFrame):
        ref_df = referencia.copy()
        ref_df["fecha"] = pd.to_datetime(ref_df["fecha"])
        referencia_mask = df["fecha"].isin(set(ref_df["fecha"]))
    else:
        referencia = pd.Series(referencia)
        if len(referencia) != len(df):
            raise ValueError("Si referencia es vector, debe tener la misma longitud que fecha.")
        referencia_mask = referencia.astype(bool).reset_index(drop=True)

    resultados_spi = []
    for escala in escalas:
        resultados_spi.append(
            _ajustar_spi_una_escala(
                df=df,
                escala=int(escala),
                referencia_mask=referencia_mask,
                distribucion=distribucion
            )
        )

    return pd.concat(resultados_spi, ignore_index=True)

def calcular_spi_por_municipio(grupo):
    grupo = grupo.sort_values("time").copy()

    referencia = None

    res = spi_indice(
        fecha=grupo["time"],
        precipitacion=grupo["precip_mensual"],
        escalas=[3, 6, 12],
        referencia=referencia,
        distribucion="Gamma"
    )

    res = res.pivot(index="fecha", columns="escala", values="spi").reset_index()
    res = res.rename(columns={
        3: "SPI_3",
        6: "SPI_6",
        12: "SPI_12"
    })

    grupo["precip_acum_3"] = grupo["precip_mensual"].rolling(window=3, min_periods=3).sum()
    grupo["precip_acum_6"] = grupo["precip_mensual"].rolling(window=6, min_periods=6).sum()
    grupo["precip_acum_12"] = grupo["precip_mensual"].rolling(window=12, min_periods=12).sum()

    grupo = grupo.merge(res, left_on="time", right_on="fecha", how="left")
    grupo = grupo.drop(columns=["fecha"])
    return grupo

# 13. Calcular SPI
df_mensual = (
    df_mensual
    .groupby(["departamento", "municipio"], group_keys=False)
    .apply(calcular_spi_por_municipio)
    .reset_index(drop=True)
)

# 14. Resumen anual SPI
df_spi_anual = (
    df_mensual
    .groupby(["departamento", "municipio", "cod_mpio", "anio"], as_index=False)
    .agg(
        SPI3_mean_anual=("SPI_3", "mean"),
        SPI3_min_anual=("SPI_3", "min"),
        SPI3_meses_bajo_m1=("SPI_3", lambda x: (x < -1).sum()),
        SPI6_mean_anual=("SPI_6", "mean"),
        SPI6_min_anual=("SPI_6", "min"),
        SPI6_meses_bajo_m1=("SPI_6", lambda x: (x < -1).sum()),
        SPI12_mean_anual=("SPI_12", "mean"),
        SPI12_min_anual=("SPI_12", "min"),
        SPI12_meses_bajo_m1=("SPI_12", lambda x: (x < -1).sum()),
    )
)

df_spi_dic = (
    df_mensual[df_mensual["mes"] == 12][
        ["departamento", "municipio", "cod_mpio", "anio", "SPI_3", "SPI_6", "SPI_12"]
    ]
    .rename(columns={
        "SPI_3": "SPI3_dic",
        "SPI_6": "SPI6_dic",
        "SPI_12": "SPI12_dic"
    })
)

# 15. Agregación anual clima
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

# 16. Unir clima anual + SPI anual
df_anual = df_anual.merge(
    df_spi_anual,
    on=["departamento", "municipio", "cod_mpio", "anio"],
    how="left"
)

df_anual = df_anual.merge(
    df_spi_dic,
    on=["departamento", "municipio", "cod_mpio", "anio"],
    how="left"
)

# 17. Exportar
df_anual.to_csv(OUT_CSV_ANUAL, index=False, encoding="utf-8-sig")