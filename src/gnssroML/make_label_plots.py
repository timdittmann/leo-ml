'''
script to extract features and generate figures for quick labeling
'''

import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 
from sklearn.decomposition import PCA
import argparse
import pandas as pd

from feature_extract_util import *

year=2023
doy=144

file_list=os.listdir('/media/datastore/mirror/spwxdp/repro4/spire/level1b/scnPhs/%s.%03d/'%(year,doy))
for file_n in file_list:
    try:
        sample=file_n[7:-8]
    
        fn='/media/datastore/mirror/spwxdp/repro4/spire/level1b/scnPhs/%s/scnPhs_%s.0001_nc'%(sample[:8],sample)
        lv1=load_leo(fn)
        fn='/media/datastore/mirror/spwxdp/repro4/spire/level2/scnLv2/%s/scnLv2_%s.0001_nc'%(sample[:8],sample)
        lv2=load_leo(fn)

        #plot_leo(sample, sp, lv2)
        fdf=extract_fs(lv1,lv2)
        fdf.to_pickle('data/feature_sets/%s.pkl' %sample)
        plot_leo_feat(sample, lv1,lv2,fdf)
    except Exception as ex:
        print(ex)
        print(file_n)