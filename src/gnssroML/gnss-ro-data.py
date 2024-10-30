import os
import pickle
import xarray as xr
import pandas as pd

from pathlib import Path

dir_path='/media/datastore/mirror/nasa_spire/gnss-r/gnss-r/L1B/grzRfl/latest/2022'
dir_path='/media/datastore/mirror/spwxdp/test/spire/level1b/scnPhs'
#scnPhs_2023.119.163.23.02.G18.SC001_0001.0001_nc
dir_path='/media/datastore/mirror/spwxdp/repro4/spire/level1b/scnPhs/2023.119'

file_list=[]
data_list=[]
keep_list=[]
for path, subdirs, files in os.walk(dir_path):
    files = [ fi for fi in files if fi.endswith("FRO.nc") ]
    for name in files:
        fp=os.path.join(path, name)
        #print(path)
        file_list+=[fp]
        path_vals=Path(path).parts
        keep={'year':path_vals[-3], 'month':path_vals[-2], 'day':path_vals[-1], 'sv':name[-13:-10]}
        keep_list+=[keep]


df=pd.DataFrame(keep_list)
print(df.head())
print(len(df))