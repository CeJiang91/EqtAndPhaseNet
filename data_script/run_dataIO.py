#!/usr/bin/env python
# coding: utf-8
import time
from dataIO import phases2npy, seed2h5py, cataloglist2npy, sac2h5py, pharep2npy, get_phasenet_snr\
    , fullcatalog_reader, station_loc


def run_phases2npy():
    phases2npy(input_dir=r'../raw_data/XFJ1121/xfj.phase',
               output_dir=r'../processed_data/XFJ1121')


def run_fullcatalog_reader():
    fullcatalog_reader(input_file=r'../raw_data/XFJ1121/台网完全目录交换格式1121.txt'
                       , output_dir=r'../processed_data/XFJ1121')


def run_cataloglist2npy():
    cataloglist2npy(input_dir=r'../data/raw_data/xfjml0_seed', output_dir=r'../data/processed_data')


def run_seed2h5py():
    seed2h5py(seed_dir=r'../data/raw_data/xfj/xfjml0_seed',
              train_path=r'../data/processed_data/xfj_processeddata/train_data',
              processed_dir=r'../data/processed_data/xfj_processeddata')
    # data2h5py(seed_path=r'../data/test_seed', train_path=r'../data/train_data', phase_path=r'../data/phases')


def run_sac2h5py():
    sac2h5py(input_dir=r'../data/raw_data/phasenet_manul/maneq/eqwfs', train_path=r'../data/processed_data/train_data',
             processed_dir=r'../data/processed_data')


def run_pharep2npy():
    pharep2npy(input_dir=r'../data/raw_data/test_sac', output_dir='../data/processed_data')


def run_get_phasenet_snr():
    get_phasenet_snr(phase_dir='../data/processed_data', sac_dir=r'../data/raw_data/phasenet_manul/aieq/03')


def run_station_loc():
    station_loc(input_dir=r'../data/raw_data/test_sac_xfj'
                , output_dir='../data/processed_data/xfj_processeddata')


if __name__ == '__main__':
    start = time.process_time()
    project = 'xfj'
    if project == 'xfj':
        # run_fullcatalog_reader()
        run_phases2npy()
        # run_seed2h5py()
    # if project == 'xc':
        # empty
    # run_station_loc()
    print('haha')
    end = time.process_time()
    print(end - start)
