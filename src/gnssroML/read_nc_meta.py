import xarray as xr

fn='/media/datastore/mirror/nasa_spire/gnss-r/gnss-r/L1B/grzRfl/latest/2021/01/05/spire_gnss-r_L1B_grzRfl_v06.01_2021-01-05T05-06-35_FM102_G24_antFRO.nc'
fn = '/media/datastore/mirror/spwxdp/test/spire/level1b/scnPhs/2023.143/scnPhs_2023.143.169.23.02.R19.SC001_0001.0001_nc'
ds= xr.open_dataset(fn, decode_times=False)