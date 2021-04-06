#!/usr/bin/env python
# coding: utf-8
"""
Created on Thu Oct 08 19:30:14 2020

@author: jc
last update: 10/08/2020
PS:Considering the specialty of phase data, this function could only be used at once. In this case,
    we present function here for reference learning

"""
from obspy import UTCDateTime, read
import numpy as np
import glob
import h5py
import csv
from os.path import join, basename
import time
import re
from os import walk
from EQTransformer.core.predictor import _get_snr


# # 读取震相，输出npy文件
def phases2npy(input_dir, output_dir):
    files = glob.glob(r'%s/*.phase' % input_dir)
    phase = {}
    dist = {}
    for file in files:
        fn = basename(file).split('.phase')[0]  # 获取文件名
        f = open(file)
        lines = f.readlines()
        for i in range(3):
            lines.pop(0)
        for line in lines:
            if line.__eq__('#Station Magnitudes:\n'):
                break
            words = line.split()
            arrival = UTCDateTime(words[4] + ' ' + words[5])
            st = words[0].split('.')[1]
            # words[7] is weight. We skip the loop when the weight is zero and distance larger than 300km.
            if (words[7] != '1.0') or (111 * float(words[1]) > 300):
                continue
            if words[3].rfind('Pn') != -1:
                phase_type = 'P'
            elif words[3].rfind('Pg') != -1:
                phase_type = 'P'
            elif words[3].rfind('Sg') != -1:
                phase_type = 'S'
            else:
                continue
            # Initializing.
            if not phase.__contains__(fn):
                phase[fn] = {}
                dist[fn] = {}
            if not phase[fn].__contains__(st):
                phase[fn][st] = {}
                dist[fn][st] = float(words[1])
            if phase[fn][st].__contains__(phase_type):
                phase[fn][st][phase_type] = min(arrival, phase[fn][st][phase_type])
            else:
                phase[fn][st][phase_type] = arrival
    catalog_ph_dist = {'phase': phase, 'dist': dist}
    np.save(join(output_dir, 'catalog_ph_dist'), catalog_ph_dist)
    # np.save(join(output_dir, 'seismic_phases'), phase)
    # np.save(join(output_dir, 'station_dist'), dist)


def cataloglist2npy(input_dir, output_dir):
    f = open(join(input_dir, 'cataloglist.txt'), 'r', encoding='gbk')
    lines = f.readlines()
    lines.pop(0)
    mag = {}
    for line in lines:
        words = line.split()
        # ml=(ms+1.08)/1.13
        reg = re.search(r'\d\d:\d\d:\d\d.\d\d\d', line)
        M = line[reg.end():reg.end() + 5]
        ML = (float(M) + 1.08) / 1.13
        ML = round(ML, 2)
        evn = words[0]
        mag[evn] = ML
        # break
    np.save(join(output_dir, 'magnitude.npy'), mag)
    f.close()


def fullcatalog_reader(input_file, output_dir):
    # This function is suitable for seismic network interchange full catalog
    f = open(input_file, 'r', encoding='gbk')
    lines = f.readlines()
    catalog = {}
    for line in lines:
        if line == '\n':
            continue
        if line[0:3] == 'DBO':
            starttime = UTCDateTime(line[7:29])
            lat = float(line[31:37])
            lon = float(line[39:46])
            if line[58:61].isspace():
                ML = -999.0
            else:
                ML = float(line[58:61])
            if line[29:32].isspace():
                depth = -999
            else:
                depth = int(line[29:32])
        if line[0:3] == 'DEO':
            evn = line[6:26]
            catalog[evn] = {}
            catalog[evn]['lat'] = lat
            catalog[evn]['lon'] = lon
            catalog[evn]['starttime'] = starttime
            catalog[evn]['ML'] = ML
            catalog[evn]['depth'] = depth
        # if line[0:3] == 'DMB':
        # break
    f.close()
    np.save(join(output_dir, 'catalog_head.npy'), catalog)


def seed2h5py(seed_dir, train_path, processed_dir):
    seed_files = glob.glob(r'%s/*.seed' % seed_dir)
    net_phases = np.load(join(processed_dir, "seismic_phases.npy"), allow_pickle=True).item()
    # # 数据处理逻辑
    # 1.读取文件名,波形数据和文件（事件）对应的震相数据
    # 2.遍历波形
    # 3.若该文件（事件）对应的震相数据中包含波形对应的台站，且其震相类型为P，则保存该数据
    # 4.波形序列号为trn，P波到时为p_t,波形范围是[pt-10,pt+50]，故pt在截取后对应点为1001
    # 5.预处理（去倾，滤波，归一化），滤波频带为[3,20]
    # 后续可改良，统一输入目录，根据发震时刻截取地震
    startt = time.process_time()
    trn = 0
    f = h5py.File(join(train_path, "traces.hdf5"), "w")
    data = f.create_group('data')
    sts = []
    trs_name = []
    for file in seed_files:
        fn = basename(file).split('.seed')[0]
        waves = read(file)
        event_phases = net_phases[fn]
        channel_num = 0
        for tr in waves:
            if (event_phases.keys().__contains__(tr.stats.station)) and (
                    event_phases[tr.stats.station].keys().__contains__('P')):
                trn += 1
                now_st = tr.stats.station
                now_net = tr.stats.network
                tn = now_st + '.' + now_net + '_' + fn.split('.')[1] + '.' + fn.split('.')[2] + '_EV'
                p_t = event_phases[now_st]['P'] - 8 * 3600
                start_time_str = (p_t - 10).strftime('%Y-%m-%d %H:%M:%S.%f')
                temp_tr = tr.slice(p_t - 10 + 0.01, p_t + 50)
                temp_tr.detrend()
                # temp_tr.filter("bandpass", freqmin=3.0, freqmax=20)
                temp_tr.normalize()
                td = temp_tr.data.reshape(1, 6000)
                if channel_num == 0:
                    trs_data = np.zeros((3, 6000))
                trs_data[channel_num, :] = td
                channel_num += 1
                if channel_num == 3:
                    channel_num = 0
                    i = (trn - 1) // 3
                    dsF = data.create_dataset(tn, data=trs_data.transpose(), dtype=np.dtype('<f4'))
                    trs_name.append(tn)
                    sts.append(start_time_str)
                    dsF.attrs["trace_name"] = trs_name[i]
                    dsF.attrs["receiver_code"] = trs_name[i].split('.')[0]
                    dsF.attrs["network_code"] = trs_name[i].split('.')[1].split('_')[0]
                    dsF.attrs["receiver_latitude"] = '-999'
                    dsF.attrs["receiver_longitude"] = '-999'
                    dsF.attrs["receiver_elevation_m"] = '-999'
                    dsF.attrs['trace_start_time'] = sts[i]
    endt = time.process_time()
    print('Consumed time :' + str(endt - startt))
    f.close()
    # output to csv_file
    csv_file = open(join(train_path, "traces.csv"), 'w', newline='')
    output_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_writer.writerow(['trace_name'])
    for i in range(trn // 3):
        output_writer.writerow([trs_name[i]])
        csv_file.flush()
    csv_file.close()


def seed2h5pyv2(seed_dir, output_dir):
    seed_files = glob.glob(r'%s/*.seed' % seed_dir)
    catalog = np.load(join(seed_dir, '..', "catalog.npy"), allow_pickle=True).item()
    net_phases = catalog['phase']
    # # 数据处理逻辑
    # 1.读取文件名,波形数据和文件（事件）对应的震相数据
    # 2.遍历波形
    # 3.若该文件（事件）对应的震相数据中包含波形对应的台站，且其震相类型为P，则保存该数据
    # 4.波形序列号为trn，P波到时为p_t,波形范围是[pt-10,pt+50]，故pt在截取后对应点为1001
    # 5.预处理（去倾，滤波，归一化），滤波频带为[3,20]
    # 后续可改良，统一输入目录，根据发震时刻截取地震
    startt = time.process_time()
    trn = 0
    f = h5py.File(join(output_dir, "traces.hdf5"), "w")
    data = f.create_group('data')
    sts = []
    trs_name = []
    for file in seed_files:
        #----------delete----------------
        if trn>5000:
            break
        #------------
        skip_station = 'NoStation'
        fn = basename(file).split('.seed')[0]
        try:
            waves = read(file)
        except TypeError:
            print('Unknown format:')
            print(fn)
            continue
        if fn in net_phases:
            event_phases = net_phases[fn]
        else:
            continue
        channel_num = 0
        for tr in waves:
            if (event_phases.keys().__contains__(tr.stats.station)) and (
                    event_phases[tr.stats.station].keys().__contains__('P')):
                if (tr.stats.sampling_rate != 100) or (tr.stats.station == skip_station):
                    continue
                now_st = tr.stats.station
                now_net = tr.stats.network
                tn = now_st + '.' + now_net + '_' + fn.split('.')[1] + '.' + fn.split('.')[2] + '_EV'
                p_t = event_phases[now_st]['P'] - 8 * 3600
                start_time_str = (p_t - 10).strftime('%Y-%m-%d %H:%M:%S.%f')
                temp_tr = tr.slice(p_t - 10 + 0.01, p_t + 50)
                if temp_tr.stats.npts < 6000:
                    skip_station = tr.stats.station
                    trn = trn - channel_num
                    channel_num = 0
                    continue
                temp_tr.detrend()
                trn += 1
                # temp_tr.filter("bandpass", freqmin=3.0, freqmax=20)
                temp_tr.normalize()
                td = temp_tr.data.reshape(1, 6000)
                if channel_num == 0:
                    trs_data = np.zeros((3, 6000))
                trs_data[channel_num, :] = td
                channel_num += 1
                # print(tr.id)
                if channel_num == 3:
                    # print('----------')
                    channel_num = 0
                    i = (trn - 1) // 3
                    dsF = data.create_dataset(tn, data=trs_data.transpose(), dtype=np.dtype('<f4'))
                    trs_name.append(tn)
                    sts.append(start_time_str)
                    try:
                        dsF.attrs["trace_name"] = trs_name[i]
                    except IndexError:
                        breakpoint()
                    dsF.attrs["receiver_code"] = trs_name[i].split('.')[0]
                    dsF.attrs["network_code"] = trs_name[i].split('.')[1].split('_')[0]
                    dsF.attrs["receiver_latitude"] = '-999'
                    dsF.attrs["receiver_longitude"] = '-999'
                    dsF.attrs["receiver_elevation_m"] = '-999'
                    dsF.attrs['trace_start_time'] = sts[i]
    endt = time.process_time()
    print('Consumed time :' + str(endt - startt))
    f.close()
    # output to csv_file
    csv_file = open(join(output_dir, "traces.csv"), 'w', newline='')
    output_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_writer.writerow(['trace_name'])
    for i in range(trn // 3):
        output_writer.writerow([trs_name[i]])
        csv_file.flush()
    csv_file.close()


def pharep2npy(input_dir, output_dir):
    fp = open(join(input_dir, 'se_pharep.dat'))
    lines = fp.readlines()
    phase = {}
    evn = 'None'
    mag = {}
    dist = {}
    head = {}
    for line in lines:
        if line[0] == '#':
            words = line.split()
            date = UTCDateTime(line[2:24])
            evn = date.strftime('%Y%m%d.%H%M%S.%f')[:-4] + '.SAC'
            if not phase.__contains__(evn):
                phase[evn] = {}
                mag[evn] = words[10]
                dist[evn] = {}
                head[evn] = {}
                head[evn]['lat'] = float(words[7])
                head[evn]['lon'] = float(words[8])
                head[evn]['starttime'] = UTCDateTime(line[2:24])
                head[evn]['ML'] = float(words[9])
                head[evn]['depth'] = float(words[10])
        else:
            words = line.split()
            st = words[0]
            phase_type = words[3][0]
            arrival = UTCDateTime(words[4] + ' ' + words[5])
            if not phase[evn].__contains__(st):
                phase[evn][st] = {}
                dist[evn][st] = words[1]
            if phase[evn][st].__contains__(phase_type):
                phase[evn][st][phase_type] = min(arrival, phase[evn][st][phase_type])
            else:
                phase[evn][st][phase_type] = arrival
    catalog = {'head': head, 'phase': phase, 'dist': dist, 'mag': mag}
    np.save(join(output_dir, 'catalog.npy'), catalog)


def sac2h5py(input_dir, processed_dir):
    catalog = np.load(join(processed_dir, "catalog.npy"), allow_pickle=True).item()
    phase = catalog['phase']
    h5f = h5py.File(join(processed_dir, 'train_data', "traces.hdf5"), "w")
    data = h5f.create_group('data')
    channel_num = 0
    cutnpts = 6000
    trs_data = np.zeros((3, cutnpts))
    trn = 0
    trs_name = []
    sts = []
    for root, dirs, files in walk(input_dir):
        # dirs.sort()
        # for f in files:
        for f in sorted(files):
            tr = read(join(root, f))[0]
            if tr.stats.npts < cutnpts:
                continue
            evn = root.split('/')[-1]
            if phase.__contains__(evn):
                trn += 1
                now_st = tr.stats.station
                now_net = tr.stats.network
                start_time = tr.stats.starttime
                start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S.%f')
                tn = now_st + '.' + now_net + '_' + evn + '_EV'
                tr = tr.slice(start_time, start_time + cutnpts / 100 - 0.01)
                tr.detrend()
                # tr.filter("bandpass", freqmin=1.0, freqmax=45)
                tr.normalize()
                td = tr.data.reshape(1, cutnpts)
                if channel_num == 0:
                    trs_data = np.zeros((3, cutnpts))
                trs_data[channel_num, :] = td
                channel_num += 1
                if channel_num == 3:
                    channel_num = 0
                    i = (trn - 1) // 3
                    try:
                        dsF = data.create_dataset(tn, data=trs_data.transpose(), dtype=np.dtype('<f4'))
                    except [Exception]:
                        breakpoint()
                    trs_name.append(tn)
                    sts.append(start_time_str)
                    dsF.attrs["trace_name"] = trs_name[i]
                    dsF.attrs["receiver_code"] = trs_name[i].split('.')[0]
                    dsF.attrs["network_code"] = trs_name[i].split('.')[1].split('_')[0]
                    dsF.attrs["receiver_latitude"] = tr.stats.sac['stla'] if 'stla' in tr.stats.sac else -999
                    dsF.attrs["receiver_longitude"] = tr.stats.sac['stlo'] if 'stlo' in tr.stats.sac else -999
                    dsF.attrs["receiver_elevation_m"] = tr.stats.sac['stel'] if 'stel' in tr.stats.sac else -999
                    dsF.attrs['trace_start_time'] = sts[i]
    h5f.close()
    # output to csv_file
    csv_file = open(join(processed_dir, 'train_data', "traces.csv"), 'w', newline='')
    output_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_writer.writerow(['trace_name'])
    for i in range(trn // 3):
        output_writer.writerow([trs_name[i]])
        csv_file.flush()
    csv_file.close()


def get_phasenet_snr(phase_dir, sac_dir):
    aieq_phases = np.load(join(phase_dir, 'aieq_phases.npy'), allow_pickle=True).item()
    aieq_snr = {};
    for root, dirs, files in walk(sac_dir):
        for f in files:
            tr = read(join(root, f))[0]
            st = tr.stats.station
            evn_ai = root.split('/')[-1]
            evn_man = evn_ai.split(evn_ai.split('.')[-2])[0] + \
                      str(int(evn_ai.split('.')[-2]) + 1)[0:2] + evn_ai.split(evn_ai.split('.')[-2])[1]
            if evn_man not in aieq_phases:
                continue
            if evn_man not in aieq_snr:
                aieq_snr[evn_man] = {}
                chnum = {}
            if st in aieq_phases[evn_man]:
                tr.detrend()
                tr.filter("bandpass", freqmin=1.0, freqmax=45)
                tr.normalize()
                if st not in aieq_snr[evn_man]:
                    aieq_snr[evn_man][st] = {}
                    data = np.zeros((tr.stats.npts, 3))
                    chnum[st] = 0
                    data[:, chnum[st]] = tr.data
                    continue
                chnum[st] += 1
                if chnum[st] == 3:
                    breakpoint()
                data[:, chnum[st]] = tr.data
                if chnum[st] == 2:
                    for phtyp in aieq_phases[evn_man][st]:
                        pat = int((aieq_phases[evn_man][st][phtyp] - tr.stats.starttime - 8 * 3600) * 100)
                        aieq_snr[evn_man][st][phtyp] = _get_snr(tr.data, pat)
                        # print(aieq_snr)
                # plt.plot(tr.data)
                # plt.show()
                # plt.savefig('temp.png')
                # plt.close()
    np.save(join(phase_dir, 'aieq_snr'), aieq_snr)


def station_loc(input_dir, output_dir):
    files = glob.glob(r'%s/*.SAC' % input_dir)
    sts = []
    lon = []
    lat = []
    the_file = open(join(output_dir, 'station_loc.txt'), 'a')
    for f in files:
        tr = read(f)
        st = tr[0].stats.station
        if st not in sts:
            stla = tr[0].stats.sac['stla']
            stlo = tr[0].stats.sac['stlo']
            lat.append(stla)
            lon.append(stlo)
            sts.append(st)
            the_file.write(st + '%8.2f' % stla + '%8.2f' % stlo + '\n')
