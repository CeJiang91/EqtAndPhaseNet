#!/usr/bin/env python
# coding: utf-8
"""
Created on Thu Oct 21 21:14:14 2020

@author: jc
last update: 10/08/2020
PS:Included information: snr, dist, magnitude by man

"""
import numpy as np
import h5py
from os.path import join, basename
import matplotlib.pyplot as plt
from EQTransformer.core.predictor import _get_snr
from obspy import UTCDateTime
import time


def snr_dist_mag(h5py_dir, processed_dir, snr_calculator):
    fl = h5py.File(join(h5py_dir, 'traces.hdf5'), 'r')
    net_phases = np.load(join(processed_dir, "seismic_phases.npy"), allow_pickle=True).item()
    dist = np.load(join(processed_dir, "station_dist.npy"), allow_pickle=True).item()
    event = np.load(join(processed_dir, "event.npy"), allow_pickle=True).item()
    x_axis = []
    y_axis = []
    mag = []
    nms = []
    snr = []
    for catalog_nm in event:
        if (catalog_nm in dist) and (catalog_nm in net_phases):
            words = catalog_nm.split('.')
            for sta in dist[catalog_nm]:
                h5pynm = sta + '.' + words[0] + '_' + words[1] + '.' + words[2] + '_EV'
                if h5pynm not in fl['data']:
                    continue
                x_axis.append(111*dist[catalog_nm][sta])
                mag.append(event[catalog_nm]['ML'])
                nms.append(catalog_nm)
                if snr_calculator == 'Y':
                    data = fl['data'][h5pynm]
                    pat = int((net_phases[catalog_nm][sta]['P'] - UTCDateTime(data.attrs['trace_start_time'])
                               - 8 * 3600) * 100)
                    snr.append(_get_snr(data, pat, window=100))
    if snr_calculator == 'Y':
        np.save(join(processed_dir, 'snr.npy'), snr)
    else:
        snr = np.load(join(processed_dir, 'snr.npy'), snr)
    y_axis = snr
    mag2 = list(enumerate(mag))
    splitn = 4
    magnify_num = 0.5
    for ii in range(splitn):
        if splitn == 1:
            ind = [i for i, x in mag2 if (x < magnify_num * (ii + 1)) and (x >= magnify_num * ii)]
        elif splitn != 1 and (ii <= splitn-1):
            ind = [i for i, x in mag2 if (x < magnify_num*(ii+1)) and (x >= magnify_num*ii)]
            label = 'ML: (' + str(magnify_num * ii) + ',' + str(magnify_num * (ii + 1)) + ')'
        else:
            ind = [i for i, x in mag2 if (x >= magnify_num*ii)]
            label = 'ML: (' + str(magnify_num * ii) + ',' + '~ )'
        if len(ind) == 0:
            continue
        xylist = [[x_axis[i], y_axis[i]] for i in range(len(x_axis)) if (i in ind) and (x_axis[i] < 180)]
        xyarr = np.array(xylist)
        xarr = xyarr[:, 0]
        yarr = xyarr[:, 1]
        if ii == 0:
            plt.subplot(2, 2, ii+1)
            spot = plt.scatter(xarr, yarr, s=30, c='tab:red', alpha=0.2, label=label)
        elif ii == 1:
            plt.subplot(2, 2, ii + 1)
            spot = plt.scatter(xarr, yarr, s=30, c='tab:pink', alpha=0.2, label=label)
        elif ii == 2:
            plt.subplot(2, 2, ii + 1)
            spot = plt.scatter(xarr, yarr, s=30, c='tab:blue', alpha=0.2, label=label)
        elif ii == 3:
            plt.subplot(2, 2, ii + 1)
            spot = plt.scatter(xarr, yarr, s=30, c='tab:brown', alpha=0.2, label=label)
        plt.legend(handles=[spot], loc='upper right')
        plt.ylim([0, 80])
        plt.xlim([0, 200])
        plt.xlabel('Distance/km')
        plt.ylabel('SNR/db')
        # plt.title('ML: ('+str(magnify_num * ii)+','+str(magnify_num * (ii+1))+')')
    plt.tight_layout()
    plt.savefig('dist_snr_man.png', dpi=150)
    plt.close()


def run_snr_dist_mag():
    snr_dist_mag(h5py_dir='../data/processed_data/xfj_processeddata/train_data',
                 processed_dir='../data/processed_data/xfj_processeddata',
                 snr_calculator='N')


if __name__ == '__main__':
    start = time.process_time()
    run_snr_dist_mag()
    end = time.process_time()
    print(end - start)
