import os
import pickle
import xarray as xr
import pandas as pd
import datetime

from pathlib import Path

def load_leo(fn):
    ds = xr.open_dataset(fn, decode_times=False)
    return ds
def meta_dict(ds):
    day_str=str(int(ds.attrs['year']))+'-'+str(int(ds.attrs['month']))+'-'+str(int(ds.attrs['day']))
    hour_str=str(int(ds.attrs['hour']))+':'+str(int(ds.attrs['minute']))+':'+str(ds.attrs['second'])
    date_str=day_str+" "+hour_str
    file_dt=pd.to_datetime(date_str)
    occ_start_dt=file_dt+datetime.timedelta(seconds=float(ds.time.min()))
    occ_end_dt=file_dt+datetime.timedelta(seconds=float(ds.time.max()))
    row_dict={"start":occ_start_dt, "end":occ_end_dt, "spire_id":int(ds.attrs['leoId']), 
            "con":ds.attrs['conId'], "sat":'%02d' %ds.attrs['occsatId']}
    return row_dict

'''
dir_path='/media/datastore/mirror/spwxdp/repro4/spire/level1b/scnPhs/2023.119'
file='scnPhs_2023.119.163.23.02.G18.SC001_0001.0001_nc'
ncf=dir_path+'/'+file
ds=load_leo(ncf)
row_dict=meta_dict(ds)
print(row_dict)
'''
file_list=[]
dict_list=[]
dir_path='/media/datastore/mirror/spwxdp/repro4/spire/level1b/scnPhs'
for path, subdirs, files in os.walk(dir_path):
    files = [ fi for fi in files if fi.endswith("nc") ]
    for name in files:
        fp=os.path.join(path, name)
        file_list+=[fp]
        
        ds=load_leo(fp)
        row_dict=meta_dict(ds)
        dict_list+=[row_dict]
df=pd.DataFrame(dict_list)
print(df.head(), len(df))
df.to_pickle("data/spire_distr_df.pkl")

print(len(file_list))


