#!/usr/bin/env python
# coding: utf-8
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read
import time


def errhist(phnet_file, eqt_file, catalog):
    def to_percent(y, position):
        return str(100 * y) + "%"

    print('errhist')
    catalog = np.load(catalog, allow_pickle=True).item()
    phnet = np.load(phnet_file, allow_pickle=True).item()
    eqt = np.load(eqt_file, allow_pickle=True).item()
    phnerr = {};phnerr['P'] = [];phnerr['S'] = [];phndist = [];phnmag = [];phnfp = 0;phntp = 0
    eqterr = {};eqterr['P'] = [];eqterr['S'] = [];eqtdist = [];eqtmag = [];eqtfp = 0;eqttp = 0
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
                        if min(phnet_erral) < 1:
                            phnfp = phnfp + len(phnet_erral) - 1
                            phntp += 1
                            phnerr['P'].append(min(phnet_erral))
                            phndist.append(catalog['dist'][evn][st])
                            phnmag.append(catalog['head'][evn]['ML'])
                        else:
                            phnfp = phnfp + len(phnet_erral)
                        # ----EQT
                        EQT_erral = np.array(eqt[evn][st]['P']) - \
                                      catalog['phase'][evn][st]['P'] + 8 * 3600
                        if min(EQT_erral) < 1:
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
                                      catalog['phase'][evn][st]['S'] + 8 * 3600
                        if min(phnet_erral) < 1:
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
                        if min(EQT_erral) < 1:
                            eqtfp = eqtfp + len(EQT_erral) - 1
                            eqttp += 1
                            eqterr['S'].append(min(EQT_erral))
                            eqtdist.append(catalog['dist'][evn][st])
                            eqtmag.append(catalog['head'][evn]['ML'])
                        else:
                            eqtfp = eqtfp + len(EQT_erral)
    # breakpoint()
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
    # P pick image
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


if __name__ == '__main__':
    start = time.process_time()
    errhist(phnet_file='../data/phnet.npy', eqt_file='../data/EQT.npy',
            catalog='../data/catalog.npy')
    end = time.process_time()
    print(end - start)
