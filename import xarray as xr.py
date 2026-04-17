import xarray as xr
ds = xr.open_dataset(r"C:\Users\tatia\OneDrive\Escritorio\proyecto_miad\data_era5_land\era5_land_suelo_2007_2024.nc")
print(list(ds.data_vars))