
import os
import pandas as pd

from feature_extract_util import *
from ml_util import *

parent_directory = '/home/stdi2687/gnss-leo-data/data/feature_sets_all'
all_files_f = os.listdir(parent_directory)


for filez in all_files_f[:]:
    try:
        existing=pd.read_pickle('../data/feature_sets_all/%s' %filez)
        sample=filez[:-4]
        fn='/media/datastore/mirror/spwxdp/repro4/spire/level1b/scnPhs/%s/scnPhs_%s.0001_nc'%(sample[:8],sample)
        lv1=load_leo(fn)
        fn='/media/datastore/mirror/spwxdp/repro4/spire/level2/scnLv2/%s/scnLv2_%s.0001_nc'%(sample[:8],sample)
        lv2=load_leo(fn)
        fdf=extract_fs_addition(lv1, lv2, sample)
        merged_Frame = pd.merge(existing,fdf, on = 'time', how='outer')
        merged_Frame.to_pickle('../data/feature_sets_all_v2/%s.pkl' %sample)
    except Exception as ex:
        print(ex)
        print(filez)