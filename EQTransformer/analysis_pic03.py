#!/usr/bin/env python
# coding: utf-8
"""
Created on Thu Oct 15 23:11:14 2020

@author: jc
last update: 10/08/2020
PS:xfj data plot and statistical report

"""
from matplotlib.ticker import FuncFormatter
from obspy import UTCDateTime, read
import numpy as np
import glob
import h5py
import csv
import os
from os.path import join, basename
import matplotlib.pyplot as plt
import time
import re
from os import walk
from EQTransformer.core.predictor import _get_snr


def to_percent(y, position):
    return str(100*y)+"%"
detection_dir = '../xfj_detections/traces_outputs'
processed_dir = '../data/processed_data/xfj_processeddata'
net_phases = np.load(join(processed_dir, "seismic_phases.npy"), allow_pickle=True).item()
event = np.load(join(processed_dir, "event.npy"), allow_pickle=True).item()
output_dir = '../xfj_detections/analysis_image'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = open(join(detection_dir, "X_prediction_results.csv"), 'r')
f.__next__()
csv_file = csv.reader(f)
pt_err = []
st_err = []
pt_snr = []
st_snr = []
eqtn = 0
for line in csv_file:
    evn = line[0]
    catalog_nm = 'GD.'+evn.split('_')[1]
    sta = evn.split('.')[0]
    eqtn += 1
    if (len(line[14]) != 0) and (catalog_nm in net_phases) and (sta in net_phases[catalog_nm]) and \
            ('P' in net_phases[catalog_nm][sta]):
        try:
            tempt = UTCDateTime(line[11]) - net_phases[catalog_nm][sta]['P'] + 8*3600
            if abs(tempt) > 1:
                continue
            else:
                pt_err.append(tempt)
                pt_snr.append(float(line[14]))
        except:
            continue
    if (len(line[18]) != 0) and (catalog_nm in net_phases) and (sta in net_phases[catalog_nm]) and \
            ('S' in net_phases[catalog_nm][sta]):
        try:
            tempt = UTCDateTime(line[15]) - net_phases[catalog_nm][sta]['S'] + 8 * 3600
            if abs(tempt) > 1:
                continue
            else:
                st_err.append(tempt)
                st_snr.append(float(line[18]))
        except:
            continue
pt_err = np.array(pt_err)
plt.scatter(pt_snr, pt_err, s=10, color='red', edgecolors='black', label='EQT')
plt.xlim([0, 100])
plt.ylim([-1, 1])
plt.legend()
plt.savefig(join(output_dir, 'Peqt_snr.png'))
plt.close()
# P picks
# P pick image
num_bins = 101
plt.hist(pt_err, num_bins, weights=[1./len(pt_err)]*len(pt_err), edgecolor='red', linewidth=1,
         facecolor='red', range=[-0.5, 0.5], alpha=0.7, label='EQT')
formatter = FuncFormatter(to_percent)
ax = plt.gca()
ax.yaxis.set_major_formatter(formatter)
plt.legend(loc="best")
plt.grid()
plt.xlabel('Teqt - Tmanu')
plt.ylabel('Frequency')
plt.title(r'P Picks')
plt.savefig(join(output_dir, 'P_Pick.png'))
plt.close()
# S pick image
num_bins = 101
plt.hist(st_err, num_bins, weights=[1./len(st_err)]*len(st_err), edgecolor='red', linewidth=1,
         facecolor='red', range=[-0.5, 0.5], alpha=0.7, label='EQT')
formatter = FuncFormatter(to_percent)
ax = plt.gca()
ax.yaxis.set_major_formatter(formatter)
plt.legend(loc="best")
plt.grid()
plt.xlabel('Teqt - Tmanu')
plt.ylabel('Frequency')
plt.title(r'S Picks')
plt.savefig(join(output_dir, 'S_Pick.png'))
plt.close()
pn = 0
for a in net_phases:
    for b in net_phases[a]:
        for c in net_phases[a][b]:
            pn += 1
print('manuel phase num = '+str(pn)+'\n')
print('EQT num = '+str(eqtn)+'\n')
with open(os.path.join(output_dir, 'error_report.txt'), 'a') as the_file:
    the_file.write(r'EQT: std' + str(np.std(pt_err))[0:5] + ' var: ' + str(np.var(pt_err))[0:5] + ' mean:' +
                   str(np.mean(abs(pt_err)))[0:5] +'\n')
    the_file.write(r'manuel phase num = '+str(pn)+'\n')
    the_file.write(r'EQT num = '+str(eqtn)+'\n')