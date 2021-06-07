#!/usr/bin/env python
# coding: utf-8
import time
from dataIO import fullcatalog_reader, phases2npy, seed2h5pyv3, seed2h5py_continue
from sacIO import seed2sac_jopens
import glob
import os
import shutil
import numpy as np
from obspy import UTCDateTime, read, Stream
import csv
import pandas as pd


def run_xfj_catalog():
    fullcatalog_reader(input_file=r'../../XFJ1121/台网完全目录交换格式1121.txt'
                       , output_dir=r'../../XFJ1121/__xfjcache__')
    phases2npy(input_dir=r'../../XFJ1121/xfj.phase',
               output_dir=r'../../XFJ1121/__xfjcache__')
    catalog_head = np.load('../../XFJ1121/__xfjcache__/catalog_head.npy', allow_pickle=True).item()
    catalog_ph_dist = np.load('../../XFJ1121/__xfjcache__/catalog_ph_dist.npy', allow_pickle=True).item()
    catalog = {'head': {}, 'phase': {}, 'dist': {}}
    for ev in catalog_ph_dist['phase']:
        if ev in catalog_head:
            catalog['head'][ev] = catalog_head[ev]
            catalog['phase'][ev] = catalog_ph_dist['phase'][ev]
            catalog['dist'][ev] = catalog_ph_dist['dist'][ev]
            # breakpoint()
    np.save('../../XFJ1121//catalog.npy', catalog)


def run_xfj_eqtdata(seed_dir, output_dir):
    # seed2h5pyv2(seed_dir, output_dir `)
    # seed2h5pyv3(seed_dir, output_dir)
    seed2h5py_continue(seed_dir, output_dir)


def run_xfj_seed2sac(input_dir, output_dir):
    files = glob.glob(r'%s/*.seed' % input_dir)
    if os.path.isdir(output_dir):
        print('============================================================================')
        print(f' *** {output_dir} already exists!')
        inp = input(" --> Type (Yes or y) to create a new empty directory! otherwise it will overwrite!   ")
        if inp.lower() == "yes" or inp.lower() == "y":
            shutil.rmtree(output_dir)
    for file in files:
        sd_name = file.split('/')[-1][0:-5] + '.SAC'
        sd_path = os.path.join(output_dir, sd_name)
        os.makedirs(sd_path)
        seed2sac_jopens(file, sd_path)
        # break


def run_xfj_sac2phasenetdata(input_dir, output_dir, catalogfile):
    def standardize(data):
        std_data = np.std(data, axis=1, keepdims=True)
        data -= np.mean(data, axis=1, keepdims=True)
        assert (std_data.shape[0] == data.shape[0])
        std_data[std_data == 0] = 1
        data /= std_data
        return data

    if os.path.isdir(output_dir):
        print('============================================================================')
        print(f' *** {output_dir} already exists!')
        inp = input(" --> Type (Yes or y) to create a new empty directory! otherwise it will overwrite!   ")
        if inp.lower() == "yes" or inp.lower() == "y":
            shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'waveform_xfj'))
    catalog = np.load(catalogfile, allow_pickle=True).item()
    fname = []
    trn = 0
    csv_file = open(os.path.join(output_dir, "waveform.csv"), 'w', newline='')
    output_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_writer.writerow(['fname', 'eventID', 'starttime'])
    for root, dirs, files in os.walk(input_dir, topdown=True):
        dirs.sort()
        dic = {'data': {}, 'cn': {}}
        for f in sorted(files):
            if trn > 50:
                break
            tr = read(os.path.join(root, f))[0]
            if tr.stats.npts < 6000:
                continue
            evn = root.split('/')[-1][0:-4]
            st = tr.stats.station
            if (evn not in catalog['phase']) or (st not in catalog['phase'][evn]) or \
                    ('P' not in catalog['phase'][evn][st]):
                continue
            trn += 1
            start_time = catalog['phase'][evn][st]['P'] - 10 - 8 * 3600
            tn = st + '_' + evn + '.npz'
            tr = tr.trim(start_time, start_time + 60 - 0.01, pad=True, fill_value=0)
            if tr.stats.npts < 6000:
                continue
            tr.detrend('constant')
            td = tr.data.reshape(1, 6000)
            if st not in dic['data']:
                dic['data'][st] = np.zeros((3, 6000))
                dic['cn'][st] = 0
                cn = dic['cn'][st]
                dic['data'][st][cn, :] = td
                dic['cn'][st] += 1
            else:
                cn = dic['cn'][st]
                if cn >= 3:
                    print(f)
                    continue
                dic['data'][st][cn, :] = td
                dic['cn'][st] += 1
                if dic['cn'][st] == 3:
                    data = dic['data'][st]
                    data = standardize(data)
                    np.savez(os.path.join(output_dir, 'waveform_xfj', tn), data=data.transpose())
                    fname.append(tn)
                    output_writer.writerow([tn, evn, start_time])
                    csv_file.flush()
                    # breakpoint()
    csv_file.close()


def phn_continue_sac2mseed(input_dir, output_dir):
    if os.path.isdir(output_dir):
        print('============================================================================')
    print(f' *** {output_dir} already exists!')
    inp = input(" --> Type (Yes or y) to create a new empty directory! otherwise it will overwrite!   ")
    if inp.lower() == "yes" or inp.lower() == "y":
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'mseed_xfj'))
    trn = 0
    csv_file = open(os.path.join(output_dir, "fname.csv"), 'w', newline='')
    output_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_writer.writerow(['fname', 'E', 'N', 'Z'])
    for root, dirs, files in os.walk(input_dir, topdown=True):
        dirs.sort()
        if len(files) != 0:
            st = Stream()
            datastr = root.split('/')[-1].split('.')[0]
            utc_start = UTCDateTime(datastr) - 8 * 3600
            if utc_start < UTCDateTime('2021/05/21/') - 8 * 3600:
                continue
        for f in sorted(files):
            tr = read(os.path.join(root, f))
            st.append(tr[0])
        if len(files) != 0:
            nets = []
            stas = []
            st.merge()
            st.trim(utc_start, utc_start + 3600, pad=True, fill_value=0)
            for i in range(len(st)):
                net = st[i].stats.network
                sta = st[i].stats.station
                receiver_type = st[i].stats.channel[:-1]
                if net in nets and sta in stas:
                    continue
                else:
                    nets.append(net)
                    stas.append(sta)
                    fname = datastr + f'.{net}.{sta}.mseed'
                    now_st = st.select(network=net, station=sta)
                    now_st.write(os.path.join(output_dir, 'mseed_xfj', fname), format='MSEED')
                    output_writer.writerow([fname, receiver_type + 'E', receiver_type + 'N', receiver_type + 'Z'])
                    csv_file.flush()
            # break


def eqt_continue_sac2mseed(input_dir, output_dir):
    if os.path.isdir(output_dir):
        print('============================================================================')
    print(f' *** {output_dir} already exists!')
    inp = input(" --> Type (Yes or y) to create a new empty directory! otherwise it will overwrite!   ")
    if inp.lower() == "yes" or inp.lower() == "y":
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'mseed_xfj'))
    trn = 0
    csv_file = open(os.path.join(output_dir, "fname.csv"), 'w', newline='')
    output_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_writer.writerow(['fname', 'E', 'N', 'Z'])
    for root, dirs, files in os.walk(input_dir, topdown=True):
        dirs.sort()
        if len(files) != 0:
            st = Stream()
            datastr = root.split('/')[-1].split('.')[0]
            utc_start = UTCDateTime(datastr) - 8 * 3600
            if utc_start < UTCDateTime('2021/05/21/') - 8 * 3600:
                continue
        for f in sorted(files):
            tr = read(os.path.join(root, f))
            st.append(tr[0])
        if len(files) != 0:
            nets = []
            stas = []
            st.merge()
            st.trim(utc_start, utc_start + 3600, pad=True, fill_value=0)
            for i in range(len(st)):
                net = st[i].stats.network
                sta = st[i].stats.station
                receiver_type = st[i].stats.channel[:-1]
                if net in nets and sta in stas:
                    continue
                else:
                    nets.append(net)
                    stas.append(sta)
                    fname = datastr + f'.{net}.{sta}.mseed'
                    now_st = st.select(network=net, station=sta)
                    now_st.write(os.path.join(output_dir, 'mseed_xfj', fname), format='MSEED')
                    output_writer.writerow([fname, receiver_type + 'E', receiver_type + 'N', receiver_type + 'Z'])
                    csv_file.flush()
            # break


def phn_mseed_pick(input_dir, output_dir, station_file):
    files = glob.glob(r'%s/*.mseed' % input_dir)
    csv_file = open(os.path.join(output_dir, '..',"pick_mseed.csv"), 'w', newline='')
    output_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_writer.writerow(['fname', 'E', 'N', 'Z'])
    netstalist = pd.read_csv(station_file, delimiter=',',header=None)
    sta_list = list(netstalist.iloc[:, 1])
    net_list = list(netstalist.iloc[:, 0])
    for f in files:
        net = f.split('/')[-1].split('.')[1]
        sta = f.split('/')[-1].split('.')[2]
        utc =UTCDateTime(f.split('/')[-1].split('.')[0])
        fname = f.split('/')[-1]
        if (net in net_list and sta in sta_list) and utc < UTCDateTime('20210524'):
        # if (net in net_list and sta in sta_list) and UTCDateTime('20210524')<utc<UTCDateTime('20210528'):
            os.system(f'cp {f} {output_dir}')
            output_writer.writerow([fname, 'BHE', 'BHN', 'BHZ'])
            csv_file.flush()
    csv_file.close()


if __name__ == '__main__':
    start = time.process_time()
    # run_xfj_catalog()
    # run_xfj_eqtdata(seed_dir='/home/jc/disk1/Yangbi',
    #                 output_dir='/home/jc/disk1/Yangbi.eqt_input')
    # run_xfj_seed2sac(input_dir='../raw_data/XFJ/xfjml0_seed', output_dir='../raw_data/XFJ/xfjml0_sac')
    # continue_sac2mseed(input_dir='/home/jc/disk1/Yangbi_sac',
    #                    output_dir='/home/jc/disk1/Yangbi.phasenet_input')
    eqt_continue_sac2mseed(input_dir='/home/jc/disk1/Yangbi_sac',
                           output_dir='/home/jc/disk1/Yangbi.phasenet_input')
    # phn_mseed_pick(input_dir='/home/jiangce/work/SeismicData/Yangbi.phasenet_input/mseed',
    #                output_dir='/home/jiangce/work/SeismicData/Yangbi.phasenet_input/pick_mseed',
    #                station_file='/home/jiangce/work/SeismicData/Yangbi.phasenet_input/sta_list.csv')
    end = time.process_time()
    print(end - start)
