
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from matplotlib.patches import Rectangle

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

def labeldf_to_fsdf(cat_file):
    label_df=pd.read_pickle(cat_file)
    label_df_t=label_df[label_df["labeled?"]==True]
    fdf_li=[]
    for i,row in enumerate(label_df_t.Filename):
        sample=row
        if cat_file=='../data/converted_labels.pkl':
            sample=row[7:-8]

        fn="../data/data/feature_sets/%s.pkl" %sample
        try:
            fdf=pd.read_pickle(fn)
            if i==0:
                print(fdf.shape)
            if fdf.shape[1]==64:
                fdf['sample']=len(fdf)*[sample]
                fdf_li+=[fdf]
            else:
                print(fn, fdf.shape[1])
            #print(i)
        except Exception as e:
            print(e)
            pass
    fdf=pd.concat(fdf_li, axis=0, ignore_index=True)
    return fdf

def class_map_RFI(y):
    '''
    class map to change RFI during downlinks to new class (5)
    '''
    y_new=y
    y_new[y==2]=5
    return y_new

def sample_2_DFandXy(sample, ml_model):
    feature_pkl='../data/data/feature_sets/%s.pkl' %sample
    fn='/media/datastore/mirror/spwxdp/repro4/spire/level1b/scnPhs/%s/scnPhs_%s.0001_nc'%(sample[:8],sample)
    lv1=load_leo(fn)
    fn='/media/datastore/mirror/spwxdp/repro4/spire/level2/scnLv2/%s/scnLv2_%s.0001_nc'%(sample[:8],sample)
    lv2=load_leo(fn)

    fdf=pd.read_pickle(feature_pkl)
    fdf['sample']=len(fdf)*[sample]

    X, y, feature_names, fs_dict=df_2_Xy(fdf)
    #--- test model
    y_pred = ml_model.predict(X)
    print(y_pred)
    #print(y)
    # map downlink labels to Comms link
    label_df=pd.read_pickle('../data/converted_labels_comms.pkl')
    if sample in label_df.Filename.values:
        y=class_map_RFI(y)
        
    y_true=y
    return lv1, lv2, y_pred, y_true, fs_dict

def plot_leo_ml_multi(sample, ds, lv2, y_pred, y_true, times, labels, plot_f=False):

    fig,ax=plt.subplots(6, figsize=(6.5,6),sharex=True)
    ax[0].set_title(sample)
    init_time=(ds['caL1Snr'].time+ds.startTime)[0]
    for freq,snr,color in zip([1,2],['caL1Snr','pL2Snr'],['#1b9e77','#d95f02']): 
        ax[0].plot(ds[snr].time+ds.startTime-init_time, ds[snr].values, alpha=.5, color=color, label='L%s'%freq)
        ax[0].set_ylabel('SNR' "\n" '(v/v)')

        ax[1].plot(lv2['s4_L%s'%freq].time+lv2.startTime-init_time, lv2['s4_L%s'%freq].values, color=color, label='L%s'%freq)
        ax[1].set_ylabel('S4')
        
        
        ax[4].plot(ds['occheight'].time+ds.startTime-init_time, ds['occheight'].values, color='black')
        ax[4].set_ylabel('Occ. Ht (km)')
        ax2=ax[4].twinx()
        ax2.plot(ds['RFI'].orbtime-ds['RFI'].orbtime[0]+ds.startTime-init_time, ds['RFI'].values, color='tab:orange')
        ax2.set_ylabel('RFI Index', color='tab:orange')
            
        ax[2].plot(ds['exL%s'%freq].time+ds.startTime-init_time, ds['exL%s'%freq].values, color=color, label='L%s'%freq)
        ax[2].set_ylabel(r'$\delta \phi$' "\n" '(m)')

        ax[3].plot(lv2['sigma_phi_L%s'%freq].time+lv2.startTime-init_time, lv2['sigma_phi_L%s'%freq].values, color=color)
        ax[3].set_ylabel(r'$\sigma \phi$' "\n" '(m)')
    
    ax[1].legend(ncol=2, loc='upper right')
    
    for valz, labz, colz in zip (np.arange(6),labels,['#e41a1c','white','#4daf4a','#984ea3','#ff7f00','#ffff33']): #377eb8
        [ax[5].axvline(i-init_time) for i in times]
        [ax[5].add_patch(Rectangle((i-init_time, 0), 15, .5, alpha=.75, label=labz, color=colz)) for i in times[y_pred==valz]]
        [ax[5].add_patch(Rectangle((i-init_time, .5), 15, 1, alpha=.75, label=labz, color=colz)) for i in times[y_true==valz]]
    ax[5].axhline(0.5)
    handles,labels_=ax[5].get_legend_handles_labels() #get existing legend item handles and labels
    by_label = dict(zip(labels_, handles))
    ax[5].legend(by_label.values(), by_label.keys(), ncol=4, loc='lower left')
    ax[5].get_yaxis().set_ticks([])
    ax[5].set_ylabel('Pred Label')

    ax[5].set_xlim([times.min()-init_time, times.max()-init_time])
    ax[5].set_xlabel('Time (s)', loc="left")

    import string
    alph_li=list(string.ascii_lowercase)
    alph_li=[s + ')' for s in alph_li]
    for j,label in enumerate(alph_li[:(len(ax)-1)]):
    # Use Axes.annotate to put the label
    # - at the top left corner (axes fraction (0, 1)),
    # - offset half-a-fontsize right and half-a-fontsize down
    #   (offset fontsize (+0.5, -0.5)),
    # i.e. just inside the axes.
        ax[j].annotate(
            label,
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    ax[5].annotate(
            'f-a)',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    ax[5].annotate(
            'f-b)',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -2.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    
    # Shrink current axis's height by 10% on the bottom
    box = ax[5].get_position()
    ax[5].set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax[5].legend(by_label.values(), by_label.keys(), ncol=4, loc='upper left', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
    fig.tight_layout()
    if plot_f:
        plt.savefig("../manuscript/%s.png" %sample, dpi=300)