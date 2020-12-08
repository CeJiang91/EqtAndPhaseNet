#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:48:05 2020

@author: jc
"""

from EQTransformer.core.predictor import predictor
import time


def run_predictor():
    predictor(input_dir='./data/processed_data/xc_processeddata/train_data',
              input_model='./data/EqT_model.h5',
              output_dir='../results/data/xc/original_pick_data/xc_detections',
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


if __name__ == '__main__':
    start = time.process_time()
    run_predictor()
    end = time.process_time()
    print(end - start)
