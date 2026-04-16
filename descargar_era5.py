import cdsapi
import os

# Variables mensuales precipitación acumulada, temperatura media, humedad relativa y radiacion
client = cdsapi.Client()

dataset = "reanalysis-era5-land-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "2m_dewpoint_temperature",
        "2m_temperature",
        "surface_solar_radiation_downwards",
        "total_precipitation"
    ],
    "year": [
        "2007", "2008", "2009",
        "2010", "2011", "2012",
        "2013", "2014", "2015",
        "2016", "2017", "2018",
        "2019", "2020", "2021",
        "2022", "2023", "2024"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [6.5, -79.5, -2.0, -72.5]
}

carpeta_salida = os.path.join("datos_era5_land")
os.makedirs(carpeta_salida, exist_ok=True)

archivo_salida = os.path.join(carpeta_salida, "era5_land_mensual_2007_2024.nc")

print(f"Descargando archivo en: {archivo_salida}")
client.retrieve(dataset, request, archivo_salida)
print("Descarga finalizada.")
