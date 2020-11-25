#!/usr/bin/env python
# coding: utf-8
import time
from data_analysis import seed2errhist, seed2maghist, sac2analysisdata, plot_analysis


def run_seed2errhist():
    seed2errhist(detection_dir='../xfj_detections/traces_outputs', processed_dir=r'../data/processed_data',
                 output_dir='../analysis_image')


def run_seed2maghist():
    seed2maghist(detection_dir='../xfj_detections/traces_outputs', processed_dir=r'../data/processed_data',
                 output_dir='../analysis_image')


def run_sac2analysisdata():
    sac2analysisdata(detection_dir='../XC_detections/traces_outputs', processed_dir=r'../data/processed_data',
                     output_dir='../XC_detections/')


def run_plot_analysis():
    plot_analysis(input_dir='../XC_detections_raw', output_dir='../XC_detections_raw/analysis_image')


if __name__ == '__main__':
    start = time.process_time()
    # run_seed2hist()
    # run_seed2maghis()
    # run_sac2analysisdata()
    run_plot_analysis()
    end = time.process_time()
    print(end - start)
