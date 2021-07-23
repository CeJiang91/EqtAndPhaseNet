#!/usr/bin/env python
# coding: utf-8
import glob

import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read
import time
import pandas as pd
import os


def to_percent(y, position):
    return str(100 * y) + "%"


def xfj_errhist(phnet_file, eqt_file, catalog, snr_file=None):
    catalog = np.load(catalog, allow_pickle=True).item()
    phnet = np.load(phnet_file, allow_pickle=True).item()
    eqt = np.load(eqt_file, allow_pickle=True).item()
    if snr_file:
        snr = np.load(snr_file, allow_pickle=True)['snr'].item()
        ph_SNR = []
        eqt_SNR = []
    phnerr = {}
    phnerr['P'] = []
    phnerr['S'] = []
    phndist = []
    phnmag = []
    phnfp = 0
    phntp = 0
    eqterr = {}
    eqterr['P'] = []
    eqterr['S'] = []
    eqtdist = []
    eqtmag = []
    eqtfp = 0
    eqttp = 0
    for evn in phnet:
        if (evn in eqt) and (evn in catalog['phase']):
            # print(evn)
            for st in phnet[evn]:
                if (st in eqt[evn]) and (st in catalog['phase'][evn]):
                    # print(st)
                    # print(catalog['phase'][evn][st])
                    if ('P' in phnet[evn][st]) and ('P' in eqt[evn][st]) \
                            and ('P' in catalog['phase'][evn][st]):
                        # ----phnet
                        phnet_erral = np.array(phnet[evn][st]['P']) - \
                                      catalog['phase'][evn][st]['P'] + 8 * 3600
                        if min(abs(phnet_erral)) < 0.5:
                            phnfp = phnfp + len(phnet_erral) - 1
                            phntp += 1
                            phnerr['P'].append(phnet_erral[np.argmin(abs(phnet_erral))])
                            phndist.append(catalog['dist'][evn][st])
                            phnmag.append(catalog['head'][evn]['ML'])
                            if snr_file and (st in snr[evn]):
                                ph_SNR.append(snr[evn][st])
                        else:
                            phnfp = phnfp + len(phnet_erral)
                        # ----EQT
                        EQT_erral = np.array(eqt[evn][st]['P']) - \
                                    catalog['phase'][evn][st]['P'] + 8 * 3600
                        if min(abs(EQT_erral)) < 0.5:
                            eqtfp = eqtfp + len(EQT_erral) - 1
                            eqttp += 1
                            eqterr['P'].append(min(EQT_erral))
                            eqtdist.append(catalog['dist'][evn][st])
                            eqtmag.append(catalog['head'][evn]['ML'])
                            if snr_file and (st in snr[evn]):
                                eqt_SNR.append(snr[evn][st])
                        else:
                            eqtfp = eqtfp + len(EQT_erral)
                    if ('S' in phnet[evn][st]) and ('S' in eqt[evn][st]) \
                            and ('S' in catalog['phase'][evn][st]):
                        # ----phnet
                        phnet_erral = np.array(phnet[evn][st]['S']) - \
                                      catalog['phase'][evn][st]['S'] + 8 * 3600
                        if min(abs(phnet_erral)) < 0.5:
                            phnfp = phnfp + len(phnet_erral) - 1
                            phntp += 1
                            phnerr['S'].append(min(phnet_erral))
                            phndist.append(catalog['dist'][evn][st])
                            phnmag.append(catalog['head'][evn]['ML'])
                        else:
                            phnfp = phnfp + len(phnet_erral)
                        # ----EQT
                        EQT_erral = np.array(eqt[evn][st]['S']) - \
                                    catalog['phase'][evn][st]['S'] + 8 * 3600
                        if min(abs(EQT_erral)) < 0.5:
                            eqtfp = eqtfp + len(EQT_erral) - 1
                            eqttp += 1
                            eqterr['S'].append(min(EQT_erral))
                            eqtdist.append(catalog['dist'][evn][st])
                            eqtmag.append(catalog['head'][evn]['ML'])
                        else:
                            eqtfp = eqtfp + len(EQT_erral)
    eqt_accuracy = eqttp / (eqtfp + eqttp)
    phn_accuracy = phntp / (phnfp + phntp)
    # breakpoint()
    # P pick image
    num_bins = 41
    plt.hist(phnerr['P'], num_bins, weights=[1. / len(phnerr['P'])] * len(phnerr['P']), edgecolor='red', linewidth=1,
             facecolor='red', range=[-0.2, 0.2], alpha=0.3, label='PhaseNet')
    plt.hist(eqterr['P'], num_bins, weights=[1. / len(eqterr['P'])] * len(eqterr['P']), edgecolor='blue', linewidth=1,
             facecolor='blue', range=[-0.2, 0.2], alpha=0.3, label='EQT')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Tai - Tmanual')
    plt.ylabel('Frequency')
    plt.title(r'P Picks')
    plt.savefig('P_Pick.png')
    plt.close()
    # S pick image
    num_bins = 41
    plt.hist(phnerr['S'], num_bins, weights=[1. / len(phnerr['S'])] * len(phnerr['S']), edgecolor='red', linewidth=1,
             facecolor='red', range=[-0.2, 0.2], alpha=0.3, label='PhaseNet')
    plt.hist(eqterr['S'], num_bins, weights=[1. / len(eqterr['S'])] * len(eqterr['S']), edgecolor='blue', linewidth=1,
             facecolor='blue', range=[-0.2, 0.2], alpha=0.3, label='EQT')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Tai - Tmanual')
    plt.ylabel('Frequency')
    plt.title(r'S Picks')
    plt.savefig('S_Pick.png')
    plt.close()
    # SNR pick image
    num_bins = 71
    plt.hist(ph_SNR, num_bins, weights=[1. / len(ph_SNR)] * len(ph_SNR), edgecolor='red', linewidth=1,
             facecolor='red', range=[-10, 60.], alpha=0.3, label='PhaseNet')
    plt.hist(eqt_SNR, num_bins, weights=[1. / len(eqt_SNR)] * len(eqt_SNR), edgecolor='blue', linewidth=1,
             facecolor='blue', range=[-10, 60], alpha=0.3, label='EQT')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('Frequency')
    # plt.title(r'SNR Picks')
    plt.savefig('SNR_Pick.png')
    plt.close()
    with open('error_report.txt', 'a') as the_file:
        the_file.write(r'EQT: std' + str(np.std(eqterr['P']))[0:5] + ' var: ' + str(np.var(eqterr['P']))[0:5]
                       + ' mean/abs:' + str(np.mean(np.abs(eqterr['P'])))[0:5] + ' mean:' +
                       str(np.mean(eqterr['P']))[0:5] + '\n')
        the_file.write(r'PhaseNet: std' + str(np.std(phnerr['P']))[0:5] + ' var: ' + str(np.var(phnerr['P']))[0:5]
                       + ' mean/abs:' + str(np.mean(np.abs(phnerr['P'])))[0:5] + ' mean:' +
                       str(np.mean(phnerr['P']))[0:5] + '\n')
    # breakpoint()


def xc_errhist(phnet_file, eqt_file, catalog):
    catalog = np.load(catalog, allow_pickle=True).item()
    phnet = np.load(phnet_file, allow_pickle=True).item()
    eqt = np.load(eqt_file, allow_pickle=True).item()
    phnerr = {};
    phnerr['P'] = [];
    phnerr['S'] = [];
    phndist = [];
    phnmag = [];
    phnfp = 0;
    phntp = 0
    eqterr = {};
    eqterr['P'] = [];
    eqterr['S'] = [];
    eqtdist = [];
    eqtmag = [];
    eqtfp = 0;
    eqttp = 0
    for evn in phnet:
        if (evn in eqt) and (evn in catalog['phase']):
            # print(evn)
            for st in phnet[evn]:
                if (st in eqt[evn]) and (st in catalog['phase'][evn]):
                    # print(st)
                    # print(catalog['phase'][evn][st])
                    if ('P' in phnet[evn][st]) and ('P' in eqt[evn][st]) \
                            and ('P' in catalog['phase'][evn][st]):
                        # ----phnet
                        phnet_erral = np.array(phnet[evn][st]['P']) - \
                                      catalog['phase'][evn][st]['P']
                        if abs(phnet_erral) < 0.5:
                            phnfp = phnfp
                            phntp += 1
                            phnerr['P'].append(phnet_erral)
                            phndist.append(catalog['dist'][evn][st])
                            phnmag.append(catalog['head'][evn]['ML'])
                        else:
                            phnfp = phnfp + 1
                        # ----EQT
                        EQT_erral = np.array(eqt[evn][st]['P']) - \
                                    catalog['phase'][evn][st]['P']
                        if min(abs(EQT_erral)) < 0.5:
                            eqtfp = eqtfp + len(EQT_erral) - 1
                            eqttp += 1
                            eqterr['P'].append(min(EQT_erral))
                            eqtdist.append(catalog['dist'][evn][st])
                            eqtmag.append(catalog['head'][evn]['ML'])
                        else:
                            eqtfp = eqtfp + len(EQT_erral)
                    if ('S' in phnet[evn][st]) and ('S' in eqt[evn][st]) \
                            and ('S' in catalog['phase'][evn][st]):
                        # ----phnet
                        phnet_erral = np.array(phnet[evn][st]['S']) - \
                                      catalog['phase'][evn][st]['S']
                        if abs(phnet_erral) < 0.5:
                            phnfp = phnfp
                            phntp += 1
                            phnerr['S'].append(phnet_erral)
                            phndist.append(catalog['dist'][evn][st])
                            phnmag.append(catalog['head'][evn]['ML'])
                        else:
                            phnfp = phnfp + 1
                        # ----EQT
                        EQT_erral = np.array(eqt[evn][st]['S']) - \
                                    catalog['phase'][evn][st]['S']
                        if min(abs(EQT_erral)) < 0.5:
                            eqtfp = eqtfp + len(EQT_erral) - 1
                            eqttp += 1
                            eqterr['S'].append(min(EQT_erral))
                            eqtdist.append(catalog['dist'][evn][st])
                            eqtmag.append(catalog['head'][evn]['ML'])
                        else:
                            eqtfp = eqtfp + len(EQT_erral)
    # P pick image
    num_bins = 101
    plt.hist(phnerr['P'], num_bins, weights=[1. / len(phnerr['P'])] * len(phnerr['P']), edgecolor='red', linewidth=1,
             facecolor='red', range=[-0.5, 0.5], alpha=0.3, label='PhaseNet')
    plt.hist(eqterr['P'], num_bins, weights=[1. / len(eqterr['P'])] * len(eqterr['P']), edgecolor='blue', linewidth=1,
             facecolor='blue', range=[-0.5, 0.5], alpha=0.3, label='EQT')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Tai - Tmanu')
    plt.ylabel('Frequency')
    plt.title(r'P Picks')
    plt.savefig('P_Pick.png')
    plt.close()
    # S pick image
    num_bins = 101
    plt.hist(phnerr['S'], num_bins, weights=[1. / len(phnerr['S'])] * len(phnerr['S']), edgecolor='red', linewidth=1,
             facecolor='red', range=[-0.5, 0.5], alpha=0.3, label='PhaseNet')
    plt.hist(eqterr['S'], num_bins, weights=[1. / len(eqterr['S'])] * len(eqterr['S']), edgecolor='blue', linewidth=1,
             facecolor='blue', range=[-0.5, 0.5], alpha=0.3, label='EQT')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Tai - Tmanu')
    plt.ylabel('Frequency')
    plt.title(r'S Picks')
    plt.savefig('S_Pick.png')
    plt.close()
    with open('error_report.txt', 'a') as the_file:
        the_file.write(r'EQT: std' + str(np.std(eqterr['P']))[0:5] + ' var: ' + str(np.var(eqterr['P']))[0:5]
                       + ' mean/abs:' + str(np.mean(np.abs(eqterr['P'])))[0:5] + ' mean:' +
                       str(np.mean(eqterr['P']))[0:5] + '\n')
        the_file.write(r'PhaseNet: std' + str(np.std(phnerr['P']))[0:5] + ' var: ' + str(np.var(phnerr['P']))[0:5]
                       + ' mean/abs:' + str(np.mean(np.abs(phnerr['P'])))[0:5] + ' mean:' +
                       str(np.mean(phnerr['P']))[0:5] + '\n')


def xfj_eqt_hist(eqt_file, catalog, snr_file=None):
    catalog = np.load(catalog, allow_pickle=True).item()
    eqt = np.load(eqt_file, allow_pickle=True).item()
    if snr_file:
        snr = np.load(snr_file, allow_pickle=True)['snr'].item()
        SNR = []
    eqterr = {}
    eqterr['P'] = []
    eqterr['S'] = []
    eqtdist = []
    eqtmag = []
    eqtfp = 0
    eqttp = 0
    for evn in eqt:
        if evn in catalog['phase']:
            for st in eqt[evn]:
                if st in catalog['phase'][evn]:
                    if ('P' in eqt[evn][st]) and ('P' in catalog['phase'][evn][st]):
                        EQT_erral = np.array(eqt[evn][st]['P']) - \
                                    catalog['phase'][evn][st]['P'] + 8 * 3600
                        if abs(min(EQT_erral)) < 1:
                            eqtfp = eqtfp + len(EQT_erral) - 1
                            eqttp += 1
                            eqterr['P'].append(min(EQT_erral))
                            eqtdist.append(catalog['dist'][evn][st])
                            eqtmag.append(catalog['head'][evn]['ML'])
                            if snr_file and (st in snr[evn]):
                                SNR.append(snr[evn][st])
                        else:
                            eqtfp = eqtfp + len(EQT_erral)
                    if ('S' in eqt[evn][st]) and ('S' in catalog['phase'][evn][st]):
                        # ----EQT
                        EQT_erral = np.array(eqt[evn][st]['S']) - \
                                    catalog['phase'][evn][st]['S'] + 8 * 3600
                        if abs(min(EQT_erral)) < 1:
                            eqtfp = eqtfp + len(EQT_erral) - 1
                            eqttp += 1
                            eqterr['S'].append(min(EQT_erral))
                            eqtdist.append(catalog['dist'][evn][st])
                            eqtmag.append(catalog['head'][evn]['ML'])
                        else:
                            eqtfp = eqtfp + len(EQT_erral)
    # P pick image
    num_bins = 101
    plt.hist(eqterr['P'], num_bins, weights=[1. / len(eqterr['P'])] * len(eqterr['P']), edgecolor='blue',
             linewidth=1,
             facecolor='blue', range=[-0.5, 0.5], alpha=0.3, label='EQT')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Tai - Tmanu')
    plt.ylabel('Frequency')
    plt.title(r'P Picks')
    plt.savefig('P_Pick.png')
    plt.close()
    # S pick image
    num_bins = 101
    plt.hist(eqterr['S'], num_bins, weights=[1. / len(eqterr['S'])] * len(eqterr['S']), edgecolor='blue',
             linewidth=1,
             facecolor='blue', range=[-0.5, 0.5], alpha=0.3, label='EQT')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Tai - Tmanu')
    plt.ylabel('Frequency')
    plt.title(r'S Picks')
    plt.savefig('S_Pick.png')
    plt.close()
    # SNR pick image
    num_bins = 101
    plt.hist(SNR, num_bins, weights=[1. / len(SNR)] * len(SNR), edgecolor='blue',
             linewidth=1,
             facecolor='blue', range=[-10, 60], alpha=0.3, label='EQT')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('Frequency')
    # plt.title(r'SNR Picks')
    plt.savefig('SNR_Pick.png')
    plt.close()


def xfj_phnet_hist(phnet_file, catalog):
    catalog = np.load(catalog, allow_pickle=True).item()
    phnet = np.load(phnet_file, allow_pickle=True).item()
    phneterr = {}
    phneterr['P'] = []
    phneterr['S'] = []
    phnetdist = []
    phnetmag = []
    phnetfp = 0
    phnettp = 0
    for evn in phnet:
        if evn in catalog['phase']:
            for st in phnet[evn]:
                if st in catalog['phase'][evn]:
                    if ('P' in phnet[evn][st]) and ('P' in catalog['phase'][evn][st]):
                        PHN_erral = np.array(phnet[evn][st]['P']) - \
                                    catalog['phase'][evn][st]['P'] + 8 * 3600
                        if abs(min(PHN_erral)) < 1:
                            phnetfp = phnetfp + len(PHN_erral) - 1
                            phnettp += 1
                            phneterr['P'].append(min(PHN_erral))
                            phnetdist.append(catalog['dist'][evn][st])
                            phnetmag.append(catalog['head'][evn]['ML'])
                        else:
                            phnetfp = phnetfp + len(PHN_erral)
                    if ('S' in phnet[evn][st]) and ('S' in catalog['phase'][evn][st]):
                        # ----EQT
                        PHN_erral = np.array(phnet[evn][st]['S']) - \
                                    catalog['phase'][evn][st]['S'] + 8 * 3600
                        if abs(min(PHN_erral)) < 1:
                            phnetfp = phnetfp + len(PHN_erral) - 1
                            phnettp += 1
                            phneterr['S'].append(min(PHN_erral))
                            phnetdist.append(catalog['dist'][evn][st])
                            phnetmag.append(catalog['head'][evn]['ML'])
                        else:
                            phnetfp = phnetfp + len(PHN_erral)
    # P pick image
    num_bins = 101
    plt.hist(phneterr['P'], num_bins, weights=[1. / len(phneterr['P'])] * len(phneterr['P']), edgecolor='blue',
             linewidth=1,
             facecolor='blue', range=[-0.5, 0.5], alpha=0.3, label='phasenet')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Tai - Tmanu')
    plt.ylabel('Frequency')
    plt.title(r'P Picks')
    plt.savefig('P_Pick.png')
    plt.close()
    # P pick image
    num_bins = 101
    plt.hist(phneterr['S'], num_bins, weights=[1. / len(phneterr['S'])] * len(phneterr['S']), edgecolor='blue',
             linewidth=1,
             facecolor='blue', range=[-0.5, 0.5], alpha=0.3, label='EQT')
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Tai - Tmanu')
    plt.ylabel('Frequency')
    plt.title(r'S Picks')
    plt.savefig('S_Pick.png')
    plt.close()


def mseed_phnerr(input_file, output_file, catalog_file):
    catalog = np.load(catalog_file, allow_pickle=True).item()
    df = pd.read_csv(input_file)
    if os.path.isfile('./man_phase.csv'):
        df_cata = pd.read_csv('./man_phase.csv')
    else:
        df_cata = pd.DataFrame(columns=['station', 'itp', 'its'])
        for evn in catalog['phase']:
            for sta in catalog['phase'][evn]:
                if 'P' in catalog['phase'][evn][sta]:
                    itp = catalog['phase'][evn][sta]['P']
                else:
                    itp = -999
                if 'S' in catalog['phase'][evn][sta]:
                    its = catalog['phase'][evn][sta]['S']
                else:
                    its = -999
                df_cata = df_cata.append([{'station': sta, 'itp': itp, 'its': its}])
        df_cata.to_csv('./man_phase.csv')
    # if os.path.isfile(output_file):
    #     phnerr = np.load(output_file, allow_pickle=True).item()
    #     breakpoint()
    # else:
    phnerr = {'P': [], 'S': []}
    sta_list = []
    TP = 0
    FP = 0
    FN = 0
    label_df = pd.DataFrame()
    for i in range(df.shape[0]):
        sta_list.append(df['fname'][i].split('.')[1].strip() + '/' + df['fname'][i].split('.')[2].strip())
    sta_list = list(set(sta_list))
    for sta in sta_list:
        time_range = UTCDateTime('2021-05-21')
        for i in range(7):
            label_df = label_df.append(df_cata.loc[(df_cata.station == sta) &
                                                   (df_cata.itp.str.contains(time_range.__str__()[0:10]))])
            time_range += 3600 * 24
            label_df.index = range(len(label_df))
    for i in range(df.shape[0]):
        utc_phn = UTCDateTime(df['fname'][i].split('.')[0]) + 0.01 * int(df['fname'][i].split('_')[1])
        sta = df['fname'][i].split('.')[1].strip() + '/' + df['fname'][i].split('.')[2].strip()
        itp = list(map(eval, df['itp'][i].strip('[]').split()))
        its = list(map(eval, df['its'][i].strip('[]').split()))
        FP += len(itp)
        FP += len(its)
        if len(itp) > 0:
            utc_itp = [utc_phn + pt * 0.01 for pt in itp]
        else:
            utc_itp = []
        if len(its) > 0:
            utc_its = [utc_phn + st * 0.01 for st in its]
        else:
            utc_its = []
        selected = df_cata.loc[(df_cata.station == sta) & (df_cata.itp.str.contains(utc_phn.__str__()[0:13]))]
        # df_cata.loc[df_cata.itp.str.contains(df_cata.itp[1].__str__()[0:16])]
        selected.index = range(len(selected))
        breakrange = False
        continuerange = False
        for j in range(selected.shape[0]):
            if selected['itp'][j] != '-999':
                for pt in utc_itp:
                    phnet_erral = pt - UTCDateTime(selected['itp'][j])
                    if abs(phnet_erral) < 0.5:
                        phnerr['P'].append(phnet_erral)
                        TP += 1
                        FP -=1
                    if phnet_erral < -30:
                        breakrange = True
                        break
                    if phnet_erral > 30:
                        continuerange = True
                        continue
            if selected['its'][j] != '-999':
                for st in utc_its:
                    phnet_erral = st - UTCDateTime(selected['its'][j])
                    if abs(phnet_erral) < 0.5:
                        phnerr['S'].append(phnet_erral)
                        TP += 1
                        FP -= 1
                    if phnet_erral < -30:
                        breakrange = True
                        break
                    if phnet_erral > 30:
                        continuerange = True
                        continue
            if breakrange:
                break
            if continuerange:
                continue
        # break
    label_P_num = label_df.loc[label_df.itp != '-999'].shape[0]
    label_S_num = label_df.loc[label_df.its != '-999'].shape[0]
    true_label_num = label_S_num + label_P_num
    with open('phasenet_report.txt', 'a') as the_file:
        the_file.write('================== PhaseNet Info ==============================' + '\n')
        the_file.write('the number of PhaseNet input: ' + str(true_label_num) + '\n')
        the_file.write('P recall of PhaseNet: ' + str(TP / (true_label_num)) + '\n')
        the_file.write('P precision of PhaseNet: ' + str(TP / (TP + FP)) + '\n')
    np.save(output_file, phnerr)


def mseed_eqterr(input_dir, output_file, catalog_file):
    files = glob.glob(r'%s/*/*.csv' % input_dir)
    if os.path.isfile('./man_phase.csv'):
        df_cata = pd.read_csv('./man_phase.csv')
    else:
        catalog = np.load(catalog_file, allow_pickle=True).item()
        df_cata = pd.DataFrame(columns=['station', 'itp', 'its'])
        for evn in catalog['phase']:
            for sta in catalog['phase'][evn]:
                if 'P' in catalog['phase'][evn][sta]:
                    itp = catalog['phase'][evn][sta]['P']
                else:
                    itp = -999
                if 'S' in catalog['phase'][evn][sta]:
                    its = catalog['phase'][evn][sta]['S']
                else:
                    its = -999
                df_cata = df_cata.append([{'station': sta, 'itp': itp, 'its': its}])
        df_cata.to_csv('./man_phase.csv')
    eqterr = {'P': [], 'S': []}
    TP = 0
    FP = 0
    FN = 0
    label_df = pd.DataFrame()
    for f in files:
        df = pd.read_csv(f)
        sta = df.network[0].strip() + '/' + df.station[0].strip()
        time_range = UTCDateTime('2021-05-21')
        for i in range(7):
            label_df = label_df.append(df_cata.loc[(df_cata.station == sta) &
                                                   (df_cata.itp.str.contains(time_range.__str__()[0:10]))])
            time_range += 3600 * 24
            label_df.index = range(len(label_df))
        for i in range(df.shape[0]):
            matched_p = False
            matched_s = False
            if isinstance(df.p_arrival_time[i], str):
                utc_itp = UTCDateTime(df.p_arrival_time[i])
                exist_p = True
            else:
                exist_p = False
            if isinstance(df.s_arrival_time[i], str):
                utc_its = UTCDateTime(df.s_arrival_time[i])
                exist_s = True
            else:
                exist_s = False
            utc_eqt = UTCDateTime(df.event_start_time[i]) + 8 * 3600
            sta = df.network[i].strip() + '/' + df.station[i].strip()
            selected = df_cata.loc[(df_cata.station == sta) & (df_cata.itp.str.contains(utc_eqt.__str__()[0:13]))]
            selected.index = range(len(selected))
            for j in range(selected.shape[0]):
                if selected['itp'][j] != '-999':
                    eqt_erral = utc_itp - UTCDateTime(selected['itp'][j]) + 8 * 3600
                    if abs(eqt_erral) < 0.5:
                        eqterr['P'].append(eqt_erral)
                        TP += 1
                        matched_p = True
                    if eqt_erral < -30:
                        break
                    if eqt_erral > 30:
                        continue
                if selected['its'][j] != '-999':
                    eqt_erral = utc_its - UTCDateTime(selected['its'][j]) + 8 * 3600
                    if abs(eqt_erral) < 0.5:
                        eqterr['S'].append(eqt_erral)
                        TP += 1
                        matched_s = True
                    if eqt_erral < -30:
                        break
                    if eqt_erral > 30:
                        continue
            if (not matched_p) and exist_p:
                FP += 1
            if (not matched_s) and exist_s:
                FP += 1
    # breakpoint()
    label_P_num = label_df.loc[label_df.itp != '-999'].shape[0]
    label_S_num = label_df.loc[label_df.its != '-999'].shape[0]
    true_label_num = label_S_num + label_P_num
    with open('eqt_report.txt', 'a') as the_file:
        the_file.write('================== EQTransformer Info ==============================' + '\n')
        the_file.write('the number of EQTransformer input: ' + str(true_label_num) + '\n')
        the_file.write('P recall of EQTransformer: ' + str(TP / (true_label_num)) + '\n')
        the_file.write('P precision of EQTransformer: ' + str(TP / (TP + FP)) + '\n')
    np.save(output_file, eqterr)


def err_hist(eqt_file, phn_file,output_dir=None):
    phnerr = np.load(phn_file, allow_pickle=True).item()
    eqterr = np.load(eqt_file, allow_pickle=True).item()
    num_bins = 51
    range = [-0.5, 0.5]
    plt.hist(phnerr['P'], num_bins, weights=[1. / len(phnerr['P'])] * len(phnerr['P']), edgecolor='red', linewidth=1,
             facecolor='red', range=range, alpha=0.3, label='PhaseNet')
    plt.hist(eqterr['P'], num_bins, weights=[1. / len(eqterr['P'])] * len(eqterr['P']), edgecolor='blue', linewidth=1,
             facecolor='blue', range=range, alpha=0.3, label='EQT')
    plt.xlim(range)
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Tai - Tmanual')
    plt.ylabel('Frequency')
    plt.title(r'P Picks')
    if output_dir:
        plt.savefig(os.path.join(output_dir,'P_Pick.png'))
    else:
        plt.savefig('P_Pick.png')
    plt.close()
    # S pick image
    num_bins = 51
    plt.hist(phnerr['S'], num_bins, weights=[1. / len(phnerr['S'])] * len(phnerr['S']), edgecolor='red', linewidth=1,
             facecolor='red', range=range, alpha=0.3, label='PhaseNet')
    plt.hist(eqterr['S'], num_bins, weights=[1. / len(eqterr['S'])] * len(eqterr['S']), edgecolor='blue', linewidth=1,
             facecolor='blue', range=range, alpha=0.3, label='EQT')
    plt.xlim(range)
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Tai - Tmanual')
    plt.ylabel('Frequency')
    plt.title(r'S Picks')
    if output_dir:
        plt.savefig(os.path.join(output_dir,'S_Pick.png'))
    else:
        plt.savefig('S_Pick.png')
    plt.close()


if __name__ == '__main__':
    start = time.process_time()
    # xfj_phnet_hist(phnet_file='/media/jiangce/My Passport/work/SeismicData/XFJ1121/phasenet_output_2020/phnet.npy',
    #                catalog='/media/jiangce/My Passport/work/SeismicData/XFJ1121/catalog.npy')
    xfj_errhist(phnet_file='/media/jiangce/My Passport/work/SeismicData/XFJ1121/phasenet_output_2020/phnet.npy',
                eqt_file='../../../SeismicData/XFJ1121/eqtoutputv2/EQT.npy',
                catalog='../../../SeismicData/XFJ1121/catalog.npy',
                snr_file='../../../SeismicData/XFJ1121/snr.npz')
    # mseed_phnerr(input_file='/home/jiangce/work/SeismicData/Yangbi_result/Yangbi.phasenet_output_all/picks.csv',
    #              output_file='/home/jiangce/work/SeismicData/Yangbi_result/phnerr.npy',
    #              catalog_file='/home/jiangce/work/SeismicData/Yangbi_result/man_catalog.npy')
    # mseed_eqterr(input_dir='/home/jiangce/work/SeismicData/Yangbi_result/Yangbi.eqt_output',
    #              output_file='/home/jiangce/work/SeismicData/Yangbi_result/eqterr.npy',
    #              catalog_file='/home/jiangce/work/SeismicData/Yangbi_result/man_catalog.npy')
    # err_hist(eqt_file='/home/jiangce/work/SeismicData/Yangbi_result/eqterr.npy',
    #          phn_file='/home/jiangce/work/SeismicData/Yangbi_result/phnerr.npy')
    end = time.process_time()
    print(end - start)
