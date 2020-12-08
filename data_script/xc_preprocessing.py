#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/12/7 上午11:31
# @Author  : jiang ce
# @Email   : cehasone@outlk.com
# @File    : xc_preprocessing.py
# @Software: PyCharm
import time
from dataIO import sac2h5py, pharep2npy
import h5py
import os
from EQTransformer.core.predictor import predictor
from obspy.core.stream import Stream
"""
    Code organization:
        1. check all replicated codes and replace them. 
"""


def run_pharep2npy():
    pharep2npy(input_dir=r'../raw_data/phasenet_manul/maneq/',
               output_dir='../EQTransformer/data/processed_data/xc_processeddata')


def run_sac2h5py():
    sac2h5py(input_dir=r'../raw_data/phasenet_manul/maneq/eqwfs',
             processed_dir=r'../EQTransformer/data/processed_data/xc_processeddata/')


def run_predictor():
    predictor(input_dir='../data/processed_data/xfj_processeddata/train_data',
              input_model='../data/EqT_model.h5',
              output_dir='../results/xfj_detections',
              estimate_uncertainty=False,
              output_probabilities=True,
              number_of_sampling=5,
              loss_weights=[0.02, 0.40, 0.58],
              detection_threshold=0.1,
              P_threshold=0.1,
              S_threshold=0.1,
              number_of_plots=10,
              plot_mode='time',
              batch_size=500,
              number_of_cpus=8,
              keepPS=False,
              spLimit=60)


def xc_validation():
    import matplotlib.pyplot as plt
    input_dir = r'../EQTransformer/data/processed_data/xc_processeddata/'
    output_dir = '__valipic__'
    h5f = h5py.File(os.path.join(input_dir, 'train_data', "traces.hdf5"), "r")
    data = h5f['data']
    num = 0
    for ev in data:
        for i in range(3):
            plt.plot(data[ev][:, i]+i)
        plt.show()
        plt.savefig(os.path.join(output_dir, ev+'.png'))
        plt.close()
        num += 1
        if num > 100:
            break
        # breakpoint()


if __name__ == '__main__':
    start = time.process_time()
    # run_pharep2npy()
    run_sac2h5py()
    # xc_validation()
    end = time.process_time()
    print('The program took ' + str(end - start) + 'seconds')
