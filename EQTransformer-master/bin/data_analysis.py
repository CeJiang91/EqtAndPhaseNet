#!/usr/bin/env python
# coding: utf-8


from obspy import UTCDateTime, read
import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import h5py
import csv
from os.path import join
import time


def seed2errhist(detection_dir, processed_dir, output_dir):
    net_phases = np.load(join(processed_dir, "seismic_phases.npy"), allow_pickle=True).item()
    f = open(join(detection_dir, "X_prediction_results.csv"), 'r')
    pt_err = []
    pt_prob = []
    pt_snr = []
    st_err = []
    st_prob = []
    st_snr = []
    f.__next__()
    csv_file = csv.reader(f)
    for line in csv_file:
        evn = line[0]
        catalog_nm = 'GD' + '.' + evn.split('_')[1]
        sta_nm = evn.split('.')[0]
        event_phases = net_phases[catalog_nm][sta_nm]
        if 'P' in event_phases:
            pt_manu = event_phases['P']
        else:
            pt_manu = None
        if 'S' in event_phases:
            st_manu = event_phases['S']
        else:
            st_manu = None
        if pt_manu and line[11]:
            temp_err = UTCDateTime(line[11]) + 8 * 3600 - pt_manu
            if abs(temp_err) <= 1:
                pt_err.append(temp_err)
                pt_prob.append(line[12])
                pt_snr.append(line[14])
        if st_manu and line[15]:
            temp_err = UTCDateTime(line[15]) + 8 * 3600 - st_manu
            if abs(temp_err) <= 1:
                st_err.append(temp_err)
                st_prob.append(line[16])
                st_snr.append(line[18])
        # break
    pt_err = np.array(pt_err)
    st_err = np.array(st_err)
    pt_prob = np.array(pt_prob)
    st_prob = np.array(st_prob)
    pt_snr = np.array(pt_snr)
    st_snr = np.array(st_snr)
    f.close()
    # ------------------------------------
    # P pick image
    num_bins = 101
    plt.hist(pt_err, num_bins, edgecolor='black', linewidth=1,
             facecolor='blue', range=[-0.5, 0.5], alpha=0.5)
    plt.grid()
    plt.xlabel('Teqt - Tmanu')
    plt.ylabel('Frequency')
    plt.title(r'P Picks')
    plt.savefig(join(output_dir, 'P_Pick.png'))
    plt.close()
    # S pick image
    plt.hist(st_err, num_bins, edgecolor='black', linewidth=1,
             facecolor='blue', range=[-0.5, 0.5], alpha=0.5)
    plt.grid()
    plt.xlabel('Teqt - Tmanu')
    plt.ylabel('Frequency')
    plt.title(r's Picks')
    plt.savefig(join(output_dir, 'S_Pick.png'))
    plt.close()


def seed2maghist(detection_dir, processed_dir, output_dir):
    net_phases = np.load(join(processed_dir, "seismic_phases.npy"), allow_pickle=True).item()
    mag = np.load(join(processed_dir, "magnitude.npy"), allow_pickle=True).item()
    mag_eqt = []
    mag_manu =[]
    f = open(join(detection_dir, "X_prediction_results.csv"), 'r')
    f.__next__()
    csv_file = csv.reader(f)
    for line in csv_file:
        evn = line[0]
        catalog_nm = 'GD' + '.' + evn.split('_')[1]
        if catalog_nm in mag:
            mag_eqt.append(mag[catalog_nm])
        else:
            continue
    mag_eqt = np.array(mag_eqt)
    # --------------------
    for catalog_nm in net_phases:
        for _ in net_phases[catalog_nm]:
            if catalog_nm in mag:
                mag_manu.append(mag[catalog_nm])
            else:
                continue

    mag_manu = np.array(mag_manu)
    # magnitude ----------------unfinished
    plt.hist(mag_eqt, bins=41, edgecolor='black', linewidth=1,
             facecolor='blue', range=[-1, 3], alpha=0.5)
    plt.hist(mag_manu, bins=41, edgecolor='black', linewidth=1,
             facecolor='green', range=[-1, 3], alpha=0.5)
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.savefig(join(output_dir, 'magnitude.png'))
    plt.close()


def sac2analysisdata(detection_dir, processed_dir, output_dir):
    net_phases = np.load(join(processed_dir, "seismic_phases.npy"), allow_pickle=True).item()
    aieq_phases = np.load(join(processed_dir, "aieq_phases.npy"), allow_pickle=True).item()
    static_num = {}
    static_num['man'] = {}
    static_num['man']['P'] = 0
    static_num['man']['S'] = 0
    for evn in net_phases:
        for st in net_phases[evn]:
            if 'P' in net_phases[evn][st]:
                static_num['man']['P'] += 1
            if 'P' in net_phases[evn][st]:
                static_num['man']['S'] += 1
    static_num['ai'] = {}
    static_num['ai']['P'] = 0
    static_num['ai']['S'] = 0
    for evn in aieq_phases:
        for st in aieq_phases[evn]:
            if 'P' in aieq_phases[evn][st]:
                static_num['ai']['P'] += 1
            if 'P' in aieq_phases[evn][st]:
                static_num['ai']['S'] += 1
    np.save(join(output_dir, 'static_num.npy'), static_num)
    f = open(join(detection_dir, "X_prediction_results.csv"), 'r')
    peqt_err = []; peqt_prob = []; peqt_snr = []; seqt_err = []; seqt_prob = []; seqt_snr = []
    f.__next__()
    csv_file = csv.reader(f)
    for line in csv_file:
        evn = line[0]
        catalog_nm = evn.split('_')[1]
        sta_nm = evn.split('.')[0]
        if sta_nm not in net_phases[catalog_nm]:
            continue
        event_phases = net_phases[catalog_nm][sta_nm]
        if 'P' in event_phases:
            pt_manu = event_phases['P']
        else:
            pt_manu = None
        if 'S' in event_phases:
            st_manu = event_phases['S']
        else:
            st_manu = None
        if pt_manu and line[11]:
            temp_err = UTCDateTime(line[11]) - pt_manu
            if abs(temp_err) <= 1:
                peqt_err.append(temp_err)
                peqt_prob.append(float(line[12]))
                peqt_snr.append(float(line[14]))
        if st_manu and line[15]:
            temp_err = UTCDateTime(line[15]) - st_manu
            if abs(temp_err) <= 1:
                try:
                    seqt_snr.append(float(line[18]))
                    seqt_err.append(temp_err)
                    seqt_prob.append(float(line[16]))
                except ValueError:
                    print(line[0]+'data lost')
    peqt_err = np.array(peqt_err)
    seqt_err = np.array(seqt_err)
    peqt_prob = np.array(peqt_prob)
    seqt_prob = np.array(seqt_prob)
    peqt_snr = np.array(peqt_snr)
    seqt_snr = np.array(seqt_snr)
    f.close()
    #-------------------------------------
    pai_err = []; pai_prob = []; pai_snr = []; sai_err = []; sai_prob = []; sai_snr = []
    aieq_snr = np.load(join(processed_dir, "aieq_snr.npy"), allow_pickle=True).item()
    for evn_man in net_phases:
        words = evn_man.split('.')
        date_man = UTCDateTime(words[0]+words[1]+'.'+words[2])
        for evn_ai in aieq_phases:
            if evn_ai not in aieq_snr:
                continue
            words = evn_ai.split('.')
            date_ai = UTCDateTime(words[0] + words[1] + '.' + words[2])-8*3600
            delta_t = abs(date_ai - date_man)
            if delta_t < 60:
                for st in aieq_phases[evn_ai]:
                    aiph = aieq_phases[evn_ai][st]
                    if st not in net_phases[evn_man]:
                        continue
                    event_phases = net_phases[evn_man][st]
                    if 'P' in event_phases:
                        pt_manu = event_phases['P']
                    else:
                        pt_manu = None
                    if 'S' in event_phases:
                        st_manu = event_phases['S']
                    else:
                        st_manu = None
                    # -----
                    if 'P' in aiph:
                        pt_ai = aiph['P']
                    else:
                        pt_ai = None
                    if 'S' in aiph:
                        st_ai = aiph['S']
                    else:
                        st_ai = None
                    if pt_manu and pt_ai:
                        temp_err = pt_ai - pt_manu - 3600 * 8
                        if abs(temp_err) <= 1:
                            pai_err.append(temp_err)
                            pai_snr.append(aieq_snr[evn_ai][st]['P'])
                    if st_manu and st_ai:
                        temp_err = st_ai - st_manu - 3600 * 8
                        if abs(temp_err) <= 1:
                            sai_err.append(temp_err)
                            sai_snr.append(aieq_snr[evn_ai][st]['S'])
    pai_err = np.array(pai_err)
    sai_err = np.array(sai_err)
    pai_snr = np.array(pai_snr)
    sai_snr = np.array(sai_snr)
    analyd = {}
    analyd['peqt_err'] = peqt_err
    analyd['peqt_snr'] = peqt_snr
    analyd['seqt_err'] = seqt_err
    analyd['seqt_snr'] = seqt_snr
    analyd['pai_err'] = pai_err
    analyd['pai_snr'] = pai_snr
    analyd['sai_err'] = sai_err
    analyd['sai_snr'] = sai_snr
    np.save(join(output_dir, 'analyd.npy'), analyd)


def to_percent(y, position):
    return str(100*y)+"%"


def plot_analysis(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdirs(output_dir)
    analyd = np.load(join(input_dir, "analyd.npy"), allow_pickle=True).item()
    peqt_snr = analyd['peqt_snr']
    peqt_err = analyd['peqt_err']
    pai_snr = analyd['pai_snr']
    pai_err = analyd['pai_err']
    # plt.scatter(peqt_snr, peqt_err, color='black')
    # plt.xlim([0, 100])
    # plt.ylim([-1, 1])
    # plt.title(r'EQT: std'+str(np.std(peqt_err))+' var: '+str(np.var(peqt_err)))
    # plt.savefig(join(output_dir, 'Peqt_snr.png'))
    # plt.close()
    # plt.scatter(pai_snr, pai_err, color='blue')
    # plt.xlim([0, 100])
    # plt.ylim([-1, 1])
    # plt.title(r'PhaseNet: std '+str(np.std(pai_err))+' var: '+str(np.var(pai_err)))
    # plt.savefig(join(output_dir, 'Pai_snr.png'))
    # plt.close()
    #-----he bing
    plt.scatter(peqt_snr, peqt_err, s=10, color='red', edgecolors='black', label='EQT')
    plt.scatter(pai_snr, pai_err, s=10, color='blue', edgecolors='black', label='PhaseNet')
    plt.xlim([0, 100])
    plt.ylim([-1, 1])
    plt.xlabel('SNR/db')
    plt.ylabel('Teqt - Tmanu')
    plt.legend()
    print(r'EQT: std'+str(np.std(peqt_err))[0:5]+' var: '+str(np.var(peqt_err))[0:5]+' mean:'+
              str(np.mean(abs(peqt_err)))[0:5]+'   PhaseNet: std '+ str(np.std(pai_err))[0:5]+' var: '+
               str(np.var(pai_err))[0:5] + ' mean:' +str(np.mean(abs(pai_err)))[0:5])
    plt.savefig(join(output_dir, 'Peqt_snr.png'))
    plt.close()
    # P pick image
    num_bins = 101
    plt.hist(peqt_err, num_bins, weights=[1./len(peqt_err)]*len(peqt_err), edgecolor='red', linewidth=1,
             facecolor='black', range=[-0.5, 0.5], alpha=0.5, label='EQT')
    plt.hist(pai_err, num_bins,  weights=[1./len(pai_err)]*len(pai_err), edgecolor='blue', linewidth=1,
             facecolor='black', range=[-0.5, 0.5], alpha=0.5, label='PhaseNet')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Teqt - Tmanu')
    plt.ylabel('Frequency')
    plt.title(r'P Picks')
    plt.savefig(join(output_dir, 'P_Pick.png'))
    plt.close()
    # S pick image
    seqt_snr = analyd['seqt_snr']
    seqt_err = analyd['seqt_err']
    sai_snr = analyd['sai_snr']
    sai_err = analyd['sai_err']
    plt.hist(seqt_err, num_bins, weights=[1. / len(seqt_err)] * len(seqt_err), edgecolor='red', linewidth=1,
             facecolor='black', range=[-0.5, 0.5], alpha=0.5, label='EQT')
    plt.hist(sai_err, num_bins, weights=[1. / len(sai_err)] * len(sai_err), edgecolor='blue', linewidth=1,
             facecolor='black', range=[-0.5, 0.5], alpha=0.5, label='PhaseNet')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Teqt - Tmanu')
    plt.ylabel('Frequency')
    plt.title(r's Picks')
    plt.savefig(join(output_dir, 'S_Pick.png'))
    plt.close()
    with open(os.path.join(output_dir, 'error_report.txt'), 'a') as the_file:
        the_file.write(r'EQT: std'+str(np.std(peqt_err))[0:5]+' var: '+str(np.var(peqt_err))[0:5]+' mean:'+
              str(np.mean(abs(peqt_err)))[0:5]+'   PhaseNet: std '+ str(np.std(pai_err))[0:5]+' var: '+
               str(np.var(pai_err))[0:5] + ' mean:' +str(np.mean(abs(pai_err)))[0:5] + '\n')
