import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 
from sklearn.decomposition import PCA
import argparse
import pandas as pd

def load_leo(fn):
    ds = xr.open_dataset(fn, decode_times=False)
    return ds

def load_plot(fn):
    ds=load_leo(fn)
    plot_leo(ds)

def std_l2ol1(ds):
    return (ds['exL2'].std()/ds['exL1'].std()).values
def ratio_l2ol1(ds):
    p=np.polyfit(ds['exL1'], ds['exL2'], deg=1)
    return p[0]
def coef_l2ol1_explained_l2ol1(ds):
    X=np.column_stack([ds['exL1'], ds['exL2']])
    pca = PCA(n_components=2)
    pca.fit(X)
    coeff = (pca.components_)[0,:] #use only PC0 coefficients, per Paul's work
    explained =pca.explained_variance_
    return coeff, explained
def snr_l1_std(ds):
    snr_l1_std=np.std(abs(10*np.log10(ds['caL1Snr'])))
    return snr_l1_std
    #std(abs(10*log10(output.dpa_l1_snr
def snr_l2_std(ds):
    snr_l2_std=np.std(abs(10*np.log10(ds['pL2Snr'])))
    return snr_l2_std
def snr_l1_range(ds):
    snr_l1_range=max(abs(10*np.log10(ds['caL1Snr'])))-min(abs(10*np.log10(ds['caL1Snr'])))
    #max(abs(10*log10(output.dpa_l1_snr{ii}))) - min(abs(10*log10(output.dpa_l1_snr{ii})))
    return snr_l1_range
def snr_l2_range(ds):
    snr_l2_range=max(abs(10*np.log10(ds['pL2Snr'])))-min(abs(10*np.log10(ds['pL2Snr'])))
    return snr_l2_range

def spectro_(ds, freq):
    fs=50
    nsc=int(fs*3)
    f,t,Sxx=signal.spectrogram(ds['exL%s'%freq], fs=fs, window=np.hamming(nsc), noverlap=0, mode='psd')
    g=10*np.log10(abs(Sxx[1:,:][5:25,:]))
    from numpy import unravel_index
    grow, gcol=unravel_index(g.argmax(), g.shape) #returns frequency bins 5:25, excluding 0, during time slice of maximum power
    spl=g[:, gcol]
    names=['spl%s_%.1f'%(freq,i) for i in f[1:][5:25]]
    return spl, names

def rfi_max(ds):
    rfi_max=max(ds['RFI'])
    return rfi_max

def get_features_lv1(ds):
    coeff, explained=coef_l2ol1_explained_l2ol1(ds)
    
    spectro_l1_vals, spl1_names=spectro_(ds, 1)
    spectro_l2_vals, spl2_names=spectro_(ds, 2)
    
    features=np.array((ds.time.min()+ds.startTime.values[0], std_l2ol1(ds), ratio_l2ol1(ds), 
                       snr_l1_std(ds).values, snr_l2_std(ds).values,
                    snr_l1_range(ds).values, snr_l2_range(ds).values, rfi_max(ds)))
    feature_names=['time','std_l2ol1','ratio_l2ol1', 'snr_l1_std', 
                   'snr_l2_std', 'snr_l1_range','snr_l2_range','rfi_max']
    features=np.hstack((features,coeff, explained, spectro_l1_vals, spectro_l2_vals))
    feature_names=np.hstack((feature_names, ['coeffpca0','coeffpca1'],['explpca0','explpca1'],spl1_names,spl2_names)) 
    return features, feature_names

def s4_max(ds, freq):
    s4_max=max(ds['s4_%s'%freq])
    return s4_max

def sigphi_max(ds, freq):
    sigphi_max=max(ds['sigma_phi_%s'%freq])
    return sigphi_max

def get_features_lv2(ds):
    features=np.array((ds.time.min()+ds.startTime, s4_max(ds,'L1'), 
                       s4_max(ds,'L2'), sigphi_max(ds,'L1'), sigphi_max(ds,'L2'), 
                       ds.lat.mean(), ds.lon.mean(), ds.elevation.mean(), ds.occheight.mean(), ds.slip_L1.sum(), ds.slip_L2.sum()))
    feature_names=['time2','s4_max_L1','s4_max_L1', 'sigphi_max_L1', 
                   'sigphi_max_L2', 'lat_m', 'lon_m', 'elevation_m', 'occheight_m', 'slip_L1', 'slip_L2' ]
    return features, feature_names

def extract_fs(lv1, lv2):
    feature_list=[]
    window_len=15
    windows=np.arange(lv2.time.max()+lv2.startTime,lv2.time.min()+lv2.startTime,-window_len)
    for i in range(len(windows)-2): #dont include first due to filter?
        
        # lv1 and lv2 reference times aren't the same!
        #lv1 features
        subds=lv1.sel(time=slice(windows[i+1]-lv1.startTime.values[0],windows[i]-lv1.startTime.values[0]))
        subds=subds.sel(orbtime=slice(windows[i+1],windows[i]))
        features_lv1, feature_names_lv1=get_features_lv1(subds)
        
        #lv2 features .values[0]
        subds=lv2.sel(time=slice(windows[i+1]-lv2.startTime,windows[i]-lv2.startTime))
        features_lv2, feature_names_lv2=get_features_lv2(subds)
        feature_list+=[np.hstack((features_lv1,features_lv2))]

    fdf=pd.DataFrame(feature_list, columns=list(feature_names_lv1)+feature_names_lv2)
    return fdf


def plot_leo_feat(fn, ds, lv2, fdf):
    window_len=(fdf.time[0]-fdf.time[1])/3600

    hour_init=ds.hour+(ds.minute/60)+(ds.second/3600)
    
    fig,ax=plt.subplots(7, sharex=True, figsize=(8,8))

    for freq,snr,color in zip([1,2],['caL1Snr','pL2Snr'],['#1b9e77','#d95f02']): 
        ax[0].plot(hour_init+ds[snr].time/3600, ds[snr].values, alpha=.5, color=color, label='L%s'%freq)
        ax[0].set_ylabel('SNR' "\n" '(v/v)')
        
        #ax[1].plot(ds['raw_exL%s'%freq].time, ds['raw_exL%s'%freq].values, color=color, label='L%s'%freq)
        #ax[1].set_ylabel(r'$\delta \phi$ (cycles)')

        ax[2].plot(hour_init+ds['exL%s'%freq].time/3600, ds['exL%s'%freq].values, color=color, label='L%s'%freq)
        ax[2].set_ylabel(r'$\delta \phi$' "\n" '(cycles)')
        ax[2].set_ylim([-.15,.15])

        ax[3].plot(hour_init+lv2['sigma_phi_L%s'%freq].time/3600, lv2['sigma_phi_L%s'%freq].values, color=color)
        ax[3].set_ylabel('sigma_phi')
        ax[3].set_ylim([0,.1])

        ax[4].plot(hour_init+lv2['s4_L%s'%freq].time/3600, lv2['s4_L%s'%freq].values, color=color)
        ax[4].set_ylabel('s4')

    ax[5].scatter(hour_init+((fdf['time']-lv2.startTime)/3600)+window_len/2, fdf['explpca0'])
    ax[5].set_ylabel(r'explpca0')

    ax[1].plot(hour_init+lv2['occheight'].time/3600, lv2['occheight'].values)
    ax[1].set_ylabel('occheight')
    ax2=ax[1].twinx()
    ax2.plot(hour_init+(ds['RFI'].orbtime-ds.startTime.values[0])/3600, ds['RFI'].values)
    ax2.set_ylabel('RFI')
    
    #ax[4].scatter(fdf['time']+window_len/2, fdf['snr_l1_std'])
    ax[6].scatter(hour_init+(fdf['time']-lv2.startTime)/3600+window_len/2, fdf['ratio_l2ol1'])
    ax[6].set_ylabel('ratio_l2ol1')
    ax[0].legend(ncol=2)
    for fign in np.arange(5):
        [ax[fign].axvline((i-lv2.startTime)/3600+hour_init, alpha=.5) for i in fdf.time]
        
    [ax[0].text((i[1]-lv2.startTime)/3600+hour_init, y=10, s=i[0]) for i in enumerate(fdf.time)]
    [ax[2].text((i[1]-lv2.startTime)/3600+hour_init, y=.1, s=i[0]) for i in enumerate(fdf.time)]
    #fig.tight_layout()
    plt.savefig('/home/stdi2687/gnss-leo-data/figures/labeling/%s_features.png'%fn)

def plot_leo_feat_RFI(fn, ds, lv2, fdf):
    window_len=(fdf.time[0]-fdf.time[1])/3600

    hour_init=ds.hour+(ds.minute/60)+(ds.second/3600)
    
    fig,ax=plt.subplots(7, sharex=True, figsize=(8,8))

    for freq,snr,color in zip([1,2],['caL1Snr','pL2Snr'],['#1b9e77','#d95f02']): 
        ax[0].plot(hour_init+ds[snr].time/3600, ds[snr].values, alpha=.5, color=color, label='L%s'%freq)
        ax[0].set_ylabel('SNR' "\n" '(v/v)')
        
        #ax[1].plot(ds['raw_exL%s'%freq].time, ds['raw_exL%s'%freq].values, color=color, label='L%s'%freq)
        #ax[1].set_ylabel(r'$\delta \phi$ (cycles)')

        ax[2].plot(hour_init+ds['exL%s'%freq].time/3600, ds['exL%s'%freq].values, color=color, label='L%s'%freq)
        ax[2].set_ylabel(r'$\delta \phi$' "\n" '(cycles)')
        ax[2].set_ylim([-.15,.15])

        ax[3].plot(hour_init+lv2['sigma_phi_L%s'%freq].time/3600, lv2['sigma_phi_L%s'%freq].values, color=color)
        ax[3].set_ylabel('sigma_phi')
        ax[3].set_ylim([0,.1])

        ax[4].plot(hour_init+lv2['s4_L%s'%freq].time/3600, lv2['s4_L%s'%freq].values, color=color)
        ax[4].set_ylabel('s4')

    ax[5].scatter(hour_init+((fdf['time']-lv2.startTime)/3600)+window_len/2, fdf['explpca0'])
    ax[5].set_ylabel(r'explpca0')

    ax[1].plot(hour_init+lv2['occheight'].time/3600, lv2['occheight'].values)
    ax[1].set_ylabel('occheight')
    ax2=ax[1].twinx()
    ax2.plot(hour_init+(ds['RFI'].orbtime-ds.startTime.values[0])/3600, ds['RFI'].values)
    ax2.set_ylabel('RFI')
    
    #ax[4].scatter(fdf['time']+window_len/2, fdf['snr_l1_std'])
    ax[6].scatter(hour_init+(fdf['time']-lv2.startTime)/3600+window_len/2, fdf['ratio_l2ol1'])
    ax[6].set_ylabel('ratio_l2ol1')
    ax[0].legend(ncol=2)
    for fign in np.arange(5):
        [ax[fign].axvline((i-lv2.startTime)/3600+hour_init, alpha=.5) for i in fdf.time]
        
    [ax[0].text((i[1]-lv2.startTime)/3600+hour_init, y=10, s=i[0]) for i in enumerate(fdf.time)]
    [ax[2].text((i[1]-lv2.startTime)/3600+hour_init, y=.1, s=i[0]) for i in enumerate(fdf.time)]
    #fig.tight_layout()
    plt.savefig('/home/stdi2687/leo-ml/figures/labeling_rfi2/%s_features.png'%fn)
    plt.close()

#####ADDDED

def get_features_lv1_v2(ds, sample):
    features=np.array((ds.time.min()+ds.startTime.values[0],
                       ds.xLeoLR.mean(), ds.yLeoLR.mean(),ds.zLeoLR.mean(),
                       ds.xGnssLR.mean(), ds.yGnssLR.mean(),ds.zGnssLR.mean()))
    feature_names=['time',
                   'xLeo_m', 'yLeo_m', 'zLeo_m',
                   'xGnss_m', 'yGnss_m', 'zGnss_m',]
    return features, feature_names

import numpy as np
def extract_fs_addition(lv1, lv2, sample):
    feature_list=[]
    window_len=15
    windows=np.arange(lv2.time.max()+lv2.startTime,lv2.time.min()+lv2.startTime,-window_len)
    for i in range(len(windows)-2): #dont include first due to filter?
        
        # lv1 and lv2 reference times aren't the same!
        #lv1 features
        subds=lv1.sel(time=slice(windows[i+1]-lv1.startTime.values[0],windows[i]-lv1.startTime.values[0]))
        subds=subds.sel(orbtime=slice(windows[i+1],windows[i]))
        #features_lv1, feature_names_lv1=get_features_lv1(subds)
        features, feature_names=get_features_lv1_v2(subds, sample)
        feature_list+=[features]

    fdf=pd.DataFrame(feature_list, columns=list(feature_names))
    #fdf['sample']=len(feature_list)*[sample]
    return fdf