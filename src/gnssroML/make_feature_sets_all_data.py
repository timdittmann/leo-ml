'''
script to extract features for all data (6 months SPIRE)
'''


import os

from feature_extract_util import *

def list_all_files(parent_directory):
    file_list = []
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            file_list.append(file)
    return file_list

parent_directory = '/media/datastore/mirror/spwxdp/repro4/spire/level1b/scnPhs'
all_files = list_all_files(parent_directory)

print(len(all_files))

for file_n in all_files:
    try:
        sample=file_n[7:-8]
    
        fn='/media/datastore/mirror/spwxdp/repro4/spire/level1b/scnPhs/%s/scnPhs_%s.0001_nc'%(sample[:8],sample)
        lv1=load_leo(fn)
        fn='/media/datastore/mirror/spwxdp/repro4/spire/level2/scnLv2/%s/scnLv2_%s.0001_nc'%(sample[:8],sample)
        lv2=load_leo(fn)

        #plot_leo(sample, sp, lv2)
        fdf=extract_fs(lv1,lv2)
        fdf.to_pickle('../data/feature_sets_all/%s.pkl' %sample)
        
    except Exception as ex:
        print(ex)
        print(file_n)
