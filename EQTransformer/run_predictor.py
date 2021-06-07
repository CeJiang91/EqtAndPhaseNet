#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:48:05 2020

@author: jc
"""

from EQTransformer.core.predictor import predictor
from EQTransformer.core.mseed_predictor import mseed_predictor
import time


def run_predictor():
    predictor(input_dir='../../SeismicData/XFJ1121/eqtinputv2/tenyears_set/',
              input_model='./data/EqT_model.h5',
              output_dir='../../SeismicData/XFJ1121/eqtoutputv2',
              estimate_uncertainty=False,
              output_probabilities=False,
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


def run_mseed_predictor():
    mseed_predictor(input_dir='/home/jiangce/work/SeismicData/Yangbi.phasenet_input/pick_mseed',
                    input_model='data/EqT_model.h5',
                    # stations_json='station_list.json',
                    output_dir='/home/jiangce/work/SeismicData/Yangbi.eqt_output',
                    detection_threshold=0.2,
                    P_threshold=0.1,
                    S_threshold=0.1,
                    number_of_plots=10,
                    plot_mode='time_frequency',
                    batch_size=500,
                    overlap=0.3)


if __name__ == '__main__':
    start = time.process_time()
    # run_predictor()
    run_mseed_predictor()
    end = time.process_time()
    print(end - start)
