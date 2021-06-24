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
import matplotlib as mpl

mpl.use('TkAgg')
import pandas as pd


def snr_outputer(h5py_dir, catalog_file, snr_dir='./'):
    fl = h5py.File(join(h5py_dir, 'traces.hdf5'), 'r')
    catalog = np.load(catalog_file, allow_pickle=True).item()
    net_phases = catalog['phase']
    dist = catalog['dist']
    event = catalog['head']
    x_axis = []
    y_axis = []
    mag = []
    nms = []
    snr_list = []
    snr = {}
    for catalog_nm in event:
        if (catalog_nm in dist) and (catalog_nm in net_phases):
            words = catalog_nm.split('.')
            snr[catalog_nm] = {}
            for sta in dist[catalog_nm]:
                h5pynm = sta + '.' + words[0] + '_' + words[1] + '.' + words[2] + '_EV'
                if h5pynm not in fl['data']:
                    continue
                x_axis.append(111 * dist[catalog_nm][sta])
                mag.append(event[catalog_nm]['ML'])
                nms.append(catalog_nm)
                data = fl['data'][h5pynm]
                pat = int((net_phases[catalog_nm][sta]['P'] - UTCDateTime(data.attrs['trace_start_time'])
                           - 8 * 3600) * 100)
                now_snr = _get_snr(data, pat, window=100)
                snr_list.append(now_snr)
                snr[catalog_nm][sta] = now_snr
    np.savez(join(snr_dir, 'snr.npz'), snr_list=snr_list, snr=snr)


def snr_dist_mag(h5py_dir, catalog_file, snr_calculator='N', snr_dir='./'):
    fl = h5py.File(join(h5py_dir, 'traces.hdf5'), 'r')
    catalog = np.load(catalog_file, allow_pickle=True).item()
    net_phases = catalog['phase']
    dist = catalog['dist']
    event = catalog['head']
    x_axis = []
    y_axis = []
    mag = []
    nms = []
    snr_list = []
    SNR = {}
    for catalog_nm in event:
        if (catalog_nm in dist) and (catalog_nm in net_phases):
            words = catalog_nm.split('.')
            for sta in dist[catalog_nm]:
                h5pynm = sta + '.' + words[0] + '_' + words[1] + '.' + words[2] + '_EV'
                if h5pynm not in fl['data']:
                    continue
                x_axis.append(111 * dist[catalog_nm][sta])
                mag.append(event[catalog_nm]['ML'])
                nms.append(catalog_nm)
                if snr_calculator == 'Y':
                    data = fl['data'][h5pynm]
                    pat = int((net_phases[catalog_nm][sta]['P'] - UTCDateTime(data.attrs['trace_start_time'])
                               - 8 * 3600) * 100)
                    now_snr = _get_snr(data, pat, window=100)
                    snr_list.append(now_snr)
                    SNR[catalog_nm] = now_snr
    if snr_calculator == 'Y':
        np.savez(join(snr_dir, 'snr.npz'), snr_list=snr_list, SNR=SNR)
    else:
        snr_list = np.load(join(snr_dir, 'snr.npz'))['snr_list']
    y_axis = snr_list
    mag2 = list(enumerate(mag))
    splitn = 4
    magnify_num = 0.5
    for ii in range(splitn):
        if splitn == 1:
            ind = [i for i, x in mag2 if (x < magnify_num * (ii + 1)) and (x >= magnify_num * ii)]
        elif splitn != 1 and (ii <= splitn - 1):
            ind = [i for i, x in mag2 if (x < magnify_num * (ii + 1)) and (x >= magnify_num * ii)]
            label = 'ML: (' + str(magnify_num * ii) + ',' + str(magnify_num * (ii + 1)) + ')'
        else:
            ind = [i for i, x in mag2 if (x >= magnify_num * ii)]
            label = 'ML: (' + str(magnify_num * ii) + ',' + '~ )'
        if len(ind) == 0:
            continue
        xylist = [[x_axis[i], y_axis[i]] for i in range(len(x_axis)) if (i in ind) and (x_axis[i] < 180)]
        xyarr = np.array(xylist)
        xarr = xyarr[:, 0]
        yarr = xyarr[:, 1]
        if ii == 0:
            plt.subplot(2, 2, ii + 1)
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


def snr_deltat_3d_statistical_graph(catalog_file, snr_file, phnet_file, eqt_file):
    catalog = np.load(catalog_file, allow_pickle=True).item()
    phnet = np.load(phnet_file, allow_pickle=True).item()
    eqt = np.load(eqt_file, allow_pickle=True).item()
    net_phases = catalog['phase']
    event = catalog['head']
    snr = np.load(snr_file, allow_pickle=True)['snr'].item()
    x_phnet = []
    x_eqt = []
    y_phnet = []
    y_eqt = []
    phnetnum, eqtnum = 0, 0
    for evn in eqt:
        for st in eqt[evn]:
            eqtnum += 1
    for evn in phnet:
        for st in phnet[evn]:
            phnetnum += 1
    for evn in event:
        if (evn in phnet) and (evn in eqt) and (evn in net_phases):
            words = evn.split('.')
            for st in net_phases[evn]:
                h5pynm = st + '.' + words[0] + '_' + words[1] + '.' + words[2] + '_EV'
                if st in phnet[evn] and ('P' in phnet[evn][st]) and (st in snr[evn]):
                    erral_phnet = phnet[evn][st]['P'][0] - net_phases[evn][st]['P'] + 8 * 3600
                    x_phnet.append(erral_phnet)
                    y_phnet.append(snr[evn][st])
                if st in eqt[evn] and ('P' in eqt[evn][st]) and (st in snr[evn]):
                    erral_eqt = eqt[evn][st]['P'][0] - net_phases[evn][st]['P'] + 8 * 3600
                    x_eqt.append(erral_eqt)
                    y_eqt.append(snr[evn][st])
    xrange = np.arange(-5, 6, 1)
    yrange = np.arange(0, 61, 2)
    xv, yv = np.meshgrid(xrange, yrange)
    zz_phnet = np.zeros(xv.shape)
    zz_eqt = np.zeros(xv.shape)
    # zz_phnet = np.full(xv.shape, np.nan)
    # zz_eqt = np.full(xv.shape, np.nan)
    for i in range(len(x_eqt)):
        px = round(x_eqt[i], 2)
        py = round(y_eqt[i], 0)
        index_x = np.where(xv[0] == px)[0]
        if len(np.where(xv[0] == px)[0]) == 0:
            continue
        else:
            index_x = np.where(xv[0] == px)[0][0]
        index_y = np.where(yv[:, 0] == py)[0]
        if len(np.where(yv[:, 0] == py)[0]) == 0:
            continue
        else:
            index_y = np.where(yv[:, 0] == py)[0][0]
        zz_eqt[index_y, index_x] += 1
    # ---phnet
    for i in range(len(x_phnet)):
        px = round(x_phnet[i], 2)
        py = round(y_phnet[i], 0)
        index_x = np.where(xv[0] == px)[0]
        if len(np.where(xv[0] == px)[0]) == 0:
            continue
        else:
            index_x = np.where(xv[0] == px)[0][0]
        index_y = np.where(yv[:, 0] == py)[0]
        if len(np.where(yv[:, 0] == py)[0]) == 0:
            continue
        else:
            index_y = np.where(yv[:, 0] == py)[0][0]
        zz_phnet[index_y, index_x] += 1
    # ------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xv, yv, 100 * zz_eqt / eqtnum, color='b', alpha=0.3)
    ax.plot_surface(xv, yv, 100 * zz_phnet / phnetnum, color='r', alpha=0.3)
    ax.set_xlabel('T(AI)-T(man)')
    ax.set_ylabel('SNR')
    ax.set_zlabel('Percent')
    # ax.plot_wireframe(xv, yv, zz_eqt / eqtnum, color='b', alpha=0.3)
    # ax.plot_wireframe(xv, yv, zz_phnet / phnetnum, color='r', alpha=0.3)
    plt.savefig('3d.svg')
    plt.show()
    print(mpl.get_backend)
    # breakpoint()


def eqtphn_recall(phnet_file, eqt_file, phnet_csv, eqt_csv, catalog):
    catalog = np.load(catalog, allow_pickle=True).item()
    phase = catalog['phase']
    phnet = np.load(phnet_file, allow_pickle=True).item()
    eqt = np.load(eqt_file, allow_pickle=True).item()
    csv_ph = pd.read_csv(phnet_csv)
    csv_eqt = pd.read_csv(eqt_csv)
    PP = 0
    FP = 0
    MissP = 0
    PS = 0
    FS = 0
    MissS = 0
    ExtraS = 0
    for evn in csv_ph['fname']:
        st = evn.split('_')[0]
        ev = evn.split('_')[1][:-4]
        if (ev not in phnet) or (st not in phnet[ev]):
            MissS += 1
            MissP += 1
        else:
            if 'P' in phnet[ev][st]:
                for pt in phnet[ev][st]['P']:
                    if abs(pt - phase[ev][st]['P'] + 8 * 3600) < 0.4:
                        PP += 1
                    else:
                        FP += 1
            if ('S' in phnet[ev][st]) and ('S' in phase[ev][st]):
                for pt in phnet[ev][st]['S']:
                    if abs(pt - phase[ev][st]['S'] + 8 * 3600) < 0.4:
                        PS += 1
                    else:
                        FS += 1
            elif ('S' in phase[ev][st]) and ('S' not in phnet[ev][st]):
                MissS += 1
            elif ('S' in phnet[ev][st]) and ('S' not in phase[ev][st]):
                ExtraS += 1
    with open('picker_report.txt', 'a') as the_file:
        the_file.write('================== PhaseNet Info ==============================' + '\n')
        the_file.write('the number of PhaseNet input: ' + str(len(csv_ph)) + '\n')
        the_file.write('P recall of PhaseNet: ' + str(PP / len(csv_ph)) + '\n')
        the_file.write('P precision of PhaseNet: ' + str(PP / (PP + FP)) + '\n')
        the_file.write('P false picking num of PhaseNet: ' + str(FP) + '\n')
        the_file.write('P positive picking num of PhaseNet: ' + str(PP) + '\n')
        the_file.write('S recall of PhaseNet: ' + str(PS / (PS + MissS)) + '\n')
        the_file.write('S precision of PhaseNet: ' + str(PS / (PS + FS)) + '\n')
        the_file.write('S false picking num of PhaseNet: ' + str(FS) + '\n')
        the_file.write('S positive picking num of PhaseNet: ' + str(PS) + '\n')
    # breakpoint()
    PP = 0
    FP = 0
    MissP = 0
    PS = 0
    FS = 0
    MissS = 0
    ExtraS = 0
    for evn in csv_eqt['trace_name']:
        st = evn.split('.')[0]
        ev = 'GD.'+evn.split('_')[1]
        if (ev not in eqt) or (st not in eqt[ev]):
            MissS += 1
            MissP += 1
        else:
            if 'P' in eqt[ev][st]:
                for pt in eqt[ev][st]['P']:
                    try:
                        if abs(pt - phase[ev][st]['P'] + 8 * 3600) < 0.4:
                            PP += 1
                        else:
                            FP += 1
                    except KeyError:
                        breakpoint()
            if ('S' in eqt[ev][st]) and ('S' in phase[ev][st]):
                for pt in eqt[ev][st]['S']:
                    if abs(pt - phase[ev][st]['S'] + 8 * 3600) < 0.4:
                        PS += 1
                    else:
                        FS += 1
            elif ('S' in phase[ev][st]) and ('S' not in eqt[ev][st]):
                MissS += 1
            elif ('S' in eqt[ev][st]) and ('S' not in phase[ev][st]):
                ExtraS += 1
    with open('picker_report.txt', 'a') as the_file:
        the_file.write('================== EQTransformer Info ==============================' + '\n')
        the_file.write('the number of EQTransformer input: ' + str(len(csv_ph)) + '\n')
        the_file.write('P recall of EQTransformer: ' + str(PP / len(csv_ph)) + '\n')
        the_file.write('P precision of EQTransformer: ' + str(PP / (PP + FP)) + '\n')
        the_file.write('P false picking num of EQTransformer: ' + str(FP) + '\n')
        the_file.write('P positive picking num of EQTransformer: ' + str(PP) + '\n')
        the_file.write('S recall of EQTransformer: ' + str(PS / (PS + MissS)) + '\n')
        the_file.write('S precision of EQTransformer: ' + str(PS / (PS + FS)) + '\n')
        the_file.write('S false picking num of EQTransformer: ' + str(FS) + '\n')
        the_file.write('S positive picking num of EQTransformer: ' + str(PS) + '\n')
    # breakpoint()


if __name__ == '__main__':
    start = time.process_time()
    # snr_outputer(h5py_dir='/media/jiangce/My Passport/work/SeismicData/XFJ1121/eqtinput/tenyears_set',
    #              catalog_file='/media/jiangce/My Passport/work/SeismicData/XFJ1121/catalog.npy',
    #              snr_dir='/media/jiangce/My Passport/work/SeismicData/XFJ1121/')
    # snr_deltat_3d_statistical_graph(catalog_file='/media/jiangce/My Passport/work/SeismicData/XFJ1121/catalog.npy',
    #                                 snr_file='/media/jiangce/My Passport/work/SeismicData/XFJ1121/snr.npz',
    #                                 phnet_file='/media/jiangce/My Passport/work/SeismicData/XFJ1121/phasenet_output'
    #                                            '/phnet.npy',
    #                                 eqt_file='/media/jiangce/My Passport/work/SeismicData/XFJ1121/eqtoutput/EQT.npy',
    #                                 )
    eqtphn_recall(phnet_file='../../../SeismicData/XFJ1121/phasenet_output_2020/phnet.npy',
                  eqt_file='../../../SeismicData/XFJ1121/eqtoutput0.02/EQT.npy',
                  phnet_csv='../../../SeismicData/XFJ1121/phasenet_output_2020//waveform.csv',
                  eqt_csv='../../../SeismicData/XFJ1121/eqtinput/tenyears_set/traces.csv',
                  catalog='../../../SeismicData/XFJ1121/catalog.npy')
    end = time.process_time()
    print(end - start)
