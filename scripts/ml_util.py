import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 
from sklearn.decomposition import PCA
import argparse
import pandas as pd

import sys  
sys.path.insert(1, '/home/stdi2687/gnss-leo-data/scripts')

from feature_extract_util import *

'''
ie = 1 = e region ionospheric scintillation
if = 2 = f region is
n =  3 = NO disturbance
o =  4 = oscillator anomoly
r1 =  5 = L1 rfi
r2 = 6 = L2 rfi
r = 7 = L1 + L2 RFI
a = 8 = suspected tracking/processing artifact
l = 9 = low SNR
d = 10 = unknown disturbance
11 = severe scint
'''
def class_map_binary(y):
    y_new=np.empty(len(y))
    y_new[y==1]=1
    y_new[y==2]=1
    y_new[y==3]=0
    y_new[y==4]=0
    y_new[y==5]=0
    y_new[y==6]=0
    y_new[y==7]=0
    y_new[y==8]=0
    y_new[y==9]=0
    y_new[y==10]=0
    return y_new

def class_map_multi(y):
    y_new=np.empty(len(y))
    y_new[y==1]=0
    y_new[y==2]=0
    y_new[y==3]=1
    y_new[y==4]=3
    y_new[y==5]=2
    y_new[y==6]=2
    y_new[y==7]=2
    y_new[y==8]=3
    y_new[y==9]=1
    y_new[y==10]=3
    y_new[y==11]=4
    return y_new
labels=['Scint','Quiet','RFI','Artifact','LowSNR','Unknown'] #'oa',
labels=['Scint','Quiet','RFI','Artifact','Unknown'] #'oa',
labels=['Scint','Quiet','RFI','Artifact/Unknown','Severe Scint', 'Comms Link']

def get_meta_drop(df):
    times=df['time'].values
    lats=df["lat_m"].values
    occ_hts=df['occheight_m'].values
    X=df.loc[:, df.columns != 'y_'].drop(columns=['time', 'lat_m','occheight_m']).values
    feature_names=df.loc[:, df.columns != 'y_'].drop(columns=['time', 'lat_m','occheight_m']).columns
    return X, feature_names, times, lats, occ_hts

def df_2_Xy(df):
    #drop_cols=['time2', 'rfi_max','lat_m', 'lon_m', 'elevation_m', 'occheight_m','slip_L1', 'slip_L2',]
    #drop_cols=[ 'rfi_max', 'spl1_2.0', 'spl1_2.3', 'spl1_2.7', 'spl1_3.0', 'spl1_3.3', 'spl1_3.7', 'spl1_4.0', 'spl1_4.3', 'spl1_4.7', 'spl1_5.0', 'spl1_5.3', 'spl1_5.7', 'spl1_6.0', 'spl1_6.3', 'spl1_6.7', 'spl1_7.0', 'spl1_7.3', 'spl1_7.7', 'spl1_8.0', 'spl1_8.3', 'spl2_2.0', 'spl2_2.3', 'spl2_2.7', 'spl2_3.0', 'spl2_3.3', 'spl2_3.7', 'spl2_4.0', 'spl2_4.3', 'spl2_4.7', 'spl2_5.0', 'spl2_5.3', 'spl2_5.7', 'spl2_6.0', 'spl2_6.3', 'spl2_6.7', 'spl2_7.0', 'spl2_7.3', 'spl2_7.7', 'spl2_8.0', 'spl2_8.3', 'time2', 'lat_m', 'lon_m', 'elevation_m', 'occheight_m', 'slip_L1', 'slip_L2']
    drop_cols=[ 'spl1_2.0', 'spl1_2.3', 'spl1_2.7', 'spl1_3.0', 'spl1_3.3', 'spl1_3.7', 'spl1_4.0', 'spl1_4.3', 'spl1_4.7', 'spl1_5.0', 'spl1_5.3', 'spl1_5.7', 'spl1_6.0', 'spl1_6.3', 'spl1_6.7', 'spl1_7.0', 'spl1_7.3', 'spl1_7.7', 'spl1_8.0', 'spl1_8.3', 'spl2_2.0', 'spl2_2.3', 'spl2_2.7', 'spl2_3.0', 'spl2_3.3', 'spl2_3.7', 'spl2_4.0', 'spl2_4.3', 'spl2_4.7', 'spl2_5.0', 'spl2_5.3', 'spl2_5.7', 'spl2_6.0', 'spl2_6.3', 'spl2_6.7', 'spl2_7.0', 'spl2_7.3', 'spl2_7.7', 'spl2_8.0', 'spl2_8.3', 'time2', 'elevation_m', 'slip_L1', 'slip_L2']
    #drop_cols=[ 'rfi_max','time2', 'elevation_m','slip_L1', 'slip_L2']
    df=df.drop(columns=drop_cols)

    #Drop low SNR?
    #if "y_" in df:
    #    df = df[df["y_"] != 9]

    # Drop nans, inf, etc
    df=df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    '''
    fs_dict={"times":df['time'].values, "lats":df["lat_m"].values,"lons":df["lon_m"].values,
             "occ_hts":df['occheight_m'].values,
             'xLeo_m':df['xLeo_m'].values, 'yLeo_m':df['yLeo_m'].values,'zLeo_m':df['zLeo_m'].values,
             'xGnss_m':df['xGnss_m'].values, 'yGnss_m':df['yGnss_m'].values,'zGnss_m':df['zGnss_m'].values,}
    '''
    fs_meta_cols=['time', 'lat_m','lon_m','occheight_m','sample']
    # for later version where I add extra meta
    if 'xLeo_m' in df.columns:
        fs_meta_cols=['time', 'lat_m','lon_m','occheight_m', 'xLeo_m', 'yLeo_m', 'zLeo_m','xGnss_m', 'yGnss_m', 'zGnss_m','sample']
    keys=fs_meta_cols
    vals=[df[f].values for f in fs_meta_cols]
    fs_dict=dict(zip(keys,vals))
    #X=df.loc[:, df.columns != 'y_'].drop(columns=['time', 'lat_m','lon_m','occheight_m']).values
    #feature_names=df.loc[:, df.columns != 'y_'].drop(columns=['time', 'lat_m','occheight_m']).columns
    X=df.loc[:, df.columns != 'y_'].drop(columns=fs_meta_cols).values
    feature_names=df.loc[:, df.columns != 'y_'].drop(columns=fs_meta_cols).columns
    if "y_" in df:
        y_og=df["y_"]
        y_=class_map_multi(y_og)
        #y_=y_og
    else:
        y_=np.empty(X.shape[1])*np.nan

    return X, y_, feature_names, fs_dict