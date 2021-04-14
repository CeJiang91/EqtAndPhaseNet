#!/usr/bin/env python
# coding: utf-8
import time
from dataIO import fullcatalog_reader, phases2npy, seed2h5pyv2
from sacIO import seed2sac
import glob
import os
import shutil
import numpy as np
from obspy import read
import csv
import matplotlib.pyplot as plt


def run_xfj_catalog():
    # fullcatalog_reader(input_file=r'../raw_data/XFJ1121/台网完全目录交换格式1121.txt'
    #                    , output_dir=r'../raw_data/__xfjcache__')
    # phases2npy(input_dir=r'../raw_data/XFJ1121/xfj.phase',
    #            output_dir=r'../raw_data/__xfjcache__')
    catalog_head = np.load('../raw_data/__xfjcache__/catalog_head.npy', allow_pickle=True).item()
    catalog_ph_dist = np.load('../raw_data/__xfjcache__/catalog_ph_dist.npy', allow_pickle=True).item()
    catalog = {'head': {}, 'phase': {}, 'dist': {}}
    for ev in catalog_ph_dist['phase']:
        if ev in catalog_head:
            catalog['head'][ev] = catalog_head[ev]
            catalog['phase'][ev] = catalog_ph_dist['phase'][ev]
            catalog['dist'][ev] = catalog_ph_dist['dist'][ev]
            # breakpoint()
    np.save('../raw_data/XFJ1121/catalog.npy', catalog)


def run_xfj_eqtdata(seed_dir, output_dir):
    # seed2h5pyv2(seed_dir, output_dir `)
    seed2h5pyv2(seed_dir, output_dir)


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
        seed2sac(file, sd_path)
        # break


def run_xfj_sac2phasenetdata(input_dir, output_dir, catalogfile):
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
            tr = tr.slice(start_time, start_time + 60 - 0.01)
            if tr.stats.npts < 6000:
                continue
            tr.detrend()
            tr.normalize()
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
                    np.savez(os.path.join(output_dir, 'waveform_xfj', tn), data=data.transpose())
                    fname.append(tn)
                    output_writer.writerow([tn, evn, start_time])
                    csv_file.flush()
                    # breakpoint()
    csv_file.close()


if __name__ == '__main__':
    start = time.process_time()
    # run_xfj_catalog()
    run_xfj_eqtdata()
    # run_xfj_seed2sac(input_dir='../raw_data/XFJ/xfjml0_seed', output_dir='../raw_data/XFJ/xfjml0_sac')
    # run_xfj_sac2phasenetdata(input_dir='../raw_data/XFJ/xfjml0_sac', output_dir='../raw_data/phasenet_input',
    #                          catalogfile='../raw_data/XFJ/catalog.npy')
    end = time.process_time()
    print(end - start)
