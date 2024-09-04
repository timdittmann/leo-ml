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
def plot_leo(fn, ds, lv2):

    c=299792458 #m/s
    iono_free=((ds.scnfreq1**2)*(c/ds['exL1'])-(ds.scnfreq2**2)*(c/ds['exL2']))/(ds.scnfreq1**2-ds.scnfreq2**2)
    
    ku=40.3*1e16
    dTEC=(ds['exL2']-ds['exL1'])/(ku*(1/ds.scnfreq1**2-1/ds.scnfreq2**2))

    fig,ax=plt.subplots(5, sharex=True)

    for freq,snr,color in zip([1,2],['caL1Snr','pL2Snr'],['#1b9e77','#d95f02']): 
        ax[0].plot(ds[snr].time, ds[snr].values, alpha=.5, color=color, label='L%s'%freq)
        ax[0].set_ylabel('SNR' "\n" '(v/v)')
        
        #ax[1].plot(ds['raw_exL%s'%freq].time, ds['raw_exL%s'%freq].values, color=color, label='L%s'%freq)
        #ax[1].set_ylabel(r'$\delta \phi$ (cycles)')
        ax[1].plot(lv2['elevation'].time, lv2['elevation'].values)
        ax[1].set_ylabel('elevation')

            
        ax[2].plot(ds['exL%s'%freq].time, ds['exL%s'%freq].values, color=color, label='L%s'%freq)
        ax[2].set_ylabel(r'$\delta \phi$' "\n" '(cycles)')
    ax[3].plot(dTEC.time, dTEC.values)
    ax[3].set_ylabel(r'$\Delta TEC$' "\n" '(TECU)')
    
    ax[4].plot(iono_free.time, iono_free.values)
    ax[4].set_ylabel('iono-free' "\n" '(m)')
    ax[0].legend(ncol=2)
    plt.savefig('/home/stdi2687/gnss-leo-data/figures/%s_raw.png'%fn)

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

def get_features(ds):
    coeff, explained=coef_l2ol1_explained_l2ol1(ds)
    
    spectro_l1_vals, spl1_names=spectro_(ds, 1)
    spectro_l2_vals, spl2_names=spectro_(ds, 2)
    
    features=np.array((ds.time.mean(), std_l2ol1(ds), ratio_l2ol1(ds), snr_l1_std(ds).values, snr_l2_std(ds).values,
                    snr_l1_range(ds).values, snr_l2_range(ds).values))
    feature_names=['time','std_l2ol1','ratio_l2ol1', 'snr_l1_std', 'snr_l2_std', 'snr_l1_range','snr_l2_range']
    features=np.hstack((features,coeff, explained, spectro_l1_vals, spectro_l2_vals))
    feature_names=np.hstack((feature_names, ['coeffpca0','coeffpca1'],['explpca0','explpca1'],spl1_names,spl2_names)) 
    return features, feature_names

def extract_fs(ds):
    feature_list=[]
    window_len=15
    windows=np.arange(ds.time.max(),ds.time.min(),-window_len)
    for i in range(len(windows)-2): #dont include first due to filter?
        subds=ds.sel(time=slice(windows[i+1],windows[i]))
        features, feature_names=get_features(subds)
        feature_list+=[features]
    fdf=pd.DataFrame(feature_list, columns=feature_names)
    return fdf

def plot_leo_feat(fn, ds, lv2, fdf):
    window_len=fdf.time[0]-fdf.time[1]
    fig,ax=plt.subplots(5, sharex=True)

    for freq,snr,color in zip([1,2],['caL1Snr','pL2Snr'],['#1b9e77','#d95f02']): 
        ax[0].plot(ds[snr].time, ds[snr].values, alpha=.5, color=color, label='L%s'%freq)
        ax[0].set_ylabel('SNR' "\n" '(v/v)')
        
        #ax[1].plot(ds['raw_exL%s'%freq].time, ds['raw_exL%s'%freq].values, color=color, label='L%s'%freq)
        #ax[1].set_ylabel(r'$\delta \phi$ (cycles)')
        ax[1].plot(lv2['elevation'].time, lv2['elevation'].values)
        ax[1].set_ylabel('elevation')

            
        ax[2].plot(ds['exL%s'%freq].time, ds['exL%s'%freq].values, color=color, label='L%s'%freq)
        ax[2].set_ylabel(r'$\delta \phi$' "\n" '(cycles)')
    ax[3].scatter(fdf['time']+window_len/2, fdf['explpca0'])
    ax[3].set_ylabel(r'explpca0')
    
    #ax[4].scatter(fdf['time']+window_len/2, fdf['snr_l1_std'])
    ax[4].scatter(fdf['time']+window_len/2, fdf['ratio_l2ol1'])
    ax[4].set_ylabel('ratio_l2ol1')
    ax[0].legend(ncol=2)
    for fign in np.arange(5):
        [ax[fign].axvline(i, alpha=.5) for i in fdf.time]
    plt.savefig('/home/stdi2687/gnss-leo-data/figures/%s_features.png'%fn)

def parse_arguments():
    # must input start and end year
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", help="file string, starting with YYYY to _0001", type=str)
    args = parser.parse_args().__dict__
    return {key: value for key, value in args.items() if value is not None}

def main():
    args = parse_arguments()
    
    sample=args['sample']
    fn='/media/datastore/mirror/spwxdp/test/spire/level1b/scnPhs/%s/scnPhs_%s.0001_nc'%(sample[:8],sample)
    print(fn)
    sp=load_leo(fn)
    fn='/media/datastore/mirror/spwxdp/test/spire/level2/scnLv2/%s/scnLv2_%s.0001_nc'%(sample[:8],sample)
    lv2=load_leo(fn)

    plot_leo(sample, sp, lv2)
    fdf=extract_fs(sp)
    plot_leo_feat(sample, sp,lv2,fdf)

if __name__ == "__main__":
    main()