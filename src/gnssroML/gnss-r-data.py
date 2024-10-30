# Program to determine GNSS-R data pip install pandas
import os
import pickle
import xarray as xr
import pandas as pd

from pathlib import Path

dir_path='/media/datastore/mirror/nasa_spire/gnss-r/gnss-r/L1B/grzRfl/latest/2022'

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
'''
        with xr.open_dataset(fp) as ds:
            data_list+=[ds.attrs]

df = pd.DataFrame(data_list)
df['filename']=file_list
print(df.head())
print(len(df))
'''

#with open('file_list.pkl', 'wb') as f:
#    pickle.dump(file_list, f)