#!/usr/bin/env python
# coding: utf-8

import csv
import numpy as np
from obspy import UTCDateTime


def eqtcsv2npy(csv_file, npy_file):
    f = open(csv_file, 'r')
    csv_file = csv.reader(f)
    csv_file.__next__()
    EQT = {}
    for line in csv_file:
        net = line[1].strip()
        st = line[2].strip()
        evn = net+'.'+line[0].split('_')[1]
        p_arrival = line[11]
        p_prob = line[12]
        p_snr = line[14]
        s_arrival = line[15]
        s_prob = line[16]
        s_snr = line[18]
        if evn not in EQT:
            EQT[evn] = {}
        if st not in EQT[evn]:
            EQT[evn][st] = {}
        if len(p_arrival) != 0:
            p_arrival = UTCDateTime(p_arrival)
            p_prob = float(p_prob)
            if len(p_snr) == 0:
                p_snr = -999
            else:
                p_snr = float(p_snr)
            if 'P' not in EQT[evn][st]:
                EQT[evn][st]['P'] = [p_arrival]
                EQT[evn][st]['P_prob'] = [p_prob]
                EQT[evn][st]['P_snr'] = [p_snr]
            else:
                EQT[evn][st]['P'].append(p_arrival)
                EQT[evn][st]['P_prob'].append(p_prob)
                EQT[evn][st]['P_snr'].append(p_snr)
        if len(s_arrival) != 0:
            s_arrival = UTCDateTime(s_arrival)
            s_prob = float(s_prob)
            if len(s_snr) == 0:
                s_snr = -999
            else:
                s_snr = float(s_snr)
            if 'S' not in EQT[evn][st]:
                EQT[evn][st]['S'] = [s_arrival]
                EQT[evn][st]['S_prob'] = [s_prob]
                EQT[evn][st]['S_snr'] = [s_snr]
            else:
                EQT[evn][st]['S'].append(s_arrival)
                EQT[evn][st]['S_prob'].append(s_prob)
                EQT[evn][st]['S_snr'].append(s_snr)
    f.close()
    np.save(npy_file, EQT)


def phnetcsv2npy(csv_file, waveform_csv, npy_file):
    # catalog = np.load('../raw_data/XFJ/catalog.npy', allow_pickle=True).item()
    wav = np.loadtxt(waveform_csv,
                     dtype='|S30', skiprows=1, delimiter=',', usecols=(0, 1, 2), unpack=True).astype(str)
    picks = np.loadtxt(csv_file,
                       dtype='|S30', skiprows=1, delimiter=',', usecols=(0, 1, 2, 3, 4), unpack=True).astype(str)
    phnet = {}
    for i in range(len(picks[0])):
        fn = picks[0][i]
        ind = np.where(wav[0] == fn)[0][0]
        evnfromwav = wav[1][ind]
        evn = fn.split('_')[1][0:-4]
        if evn != evnfromwav:
            print(fn)
        st = fn.split('_')[0]
        if evn not in phnet:
            phnet[evn] = {}
        if st not in phnet[evn]:
            phnet[evn][st] = {}
        starttime = UTCDateTime(wav[2][ind])
        p_point = picks[1][i].strip('[]').split(' ')
        if len(picks[1][i].strip('[]')) > 0:
            p_point = list(map(float, p_point))
            p_arrival = []
            for point in p_point:
                p_arrival.append(starttime + point / 100)
            p_prob = picks[2][i].strip('[]').split(' ')
            p_prob = list(map(float, p_prob))
            phnet[evn][st]['P'] = p_arrival
            phnet[evn][st]['P_prob'] = p_prob
        s_point = picks[3][i].strip('[]').split(' ')
        if len(picks[3][i].strip('[]')) > 0:
            s_point = list(map(float, s_point))
            s_arrival = []
            for point in s_point:
                s_arrival.append(starttime + point / 100)
            s_prob = picks[4][i].strip('[]').split()
            s_prob = list(map(float, s_prob))
            phnet[evn][st]['S'] = s_arrival
            phnet[evn][st]['S_prob'] = s_prob
        if not phnet[evn][st]:
            del phnet[evn][st]

    np.save(npy_file, phnet)


def okeqpha2npy(input_file, catalogfile, npy_file):
    fp = open(input_file)
    catalog = np.load(catalogfile, allow_pickle=True).item()
    lines = fp.readlines()
    phase = {}
    evn = 'None'
    mag = {}
    dist = {}
    for line in lines:
        if line[0] == '#':
            words = line.split()
            date = UTCDateTime(line[2:25]) + 0.001 - 8 * 3600
            for c in catalog['head']:
                if abs(catalog['head'][c]['starttime'] - date) <= 5:
                    evn = c
                    break
                elif (catalog['head'][c]['starttime'] - date) > 5:
                    evn = date.strftime('%Y%m%d.%H%M%S.%f')[:-4] + '.SAC'
                    break
                else:
                    evn = date.strftime('%Y%m%d.%H%M%S.%f')[:-4] + '.SAC'
            if not phase.__contains__(evn):
                phase[evn] = {}
                mag[evn] = words[6]
                dist[evn] = {}
        else:
            words = line.split()
            st = words[0].split('/')[1]
            phase_type = words[3][0]
            arrival = UTCDateTime(words[4] + ' ' + words[5]) - 8 * 3600
            if not phase[evn].__contains__(st):
                phase[evn][st] = {}
                dist[evn][st] = words[1]
            if phase[evn][st].__contains__(phase_type):
                phase[evn][st][phase_type] = min(arrival, phase[evn][st][phase_type])
            else:
                phase[evn][st][phase_type] = arrival
    phnet = phase
    np.save(npy_file, phnet)


if __name__ == '__main__':
    phnetcsv2npy(csv_file='/home/jc/work/XFJ1121/phasenet_output/picks.csv',
                 waveform_csv = '/home/jc/work/XFJ1121/phasenet_input/waveform.csv',
                 npy_file='/home/jc/work/XFJ1121/phasenet_output/phnet.npy')
    # eqtcsv2npy(csv_file='../data/XFJ1121V2/traces_outputs/X_prediction_results.csv',
    #            npy_file='../data/XFJ1121V2/EQT.npy')
    # okeqpha2npy(input_file='../data/xc/original_pick_data/okeqpha.dat',
    #             catalogfile='../data/xc/catalog.npy',
    #             npy_file='../data/xc/phnet.npy')
