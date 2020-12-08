#!/usr/bin/env python
# coding: utf-8
"""
Created on Thu Oct 15 23:11:14 2020

@author: jc
last update: 10/08/2020
PS:Included information: snr, dist, magnitude y EQTransformer

"""
import numpy as np
import csv
from os.path import join, basename
import matplotlib.pyplot as plt


detection_dir = '../xfj_detections/traces_outputs'
processed_dir = '../data/processed_data/xfj_processeddata'
net_phases = np.load(join(processed_dir, "seismic_phases.npy"), allow_pickle=True).item()
dist = np.load(join(processed_dir, "station_dist.npy"), allow_pickle=True).item()
event = np.load(join(processed_dir, "event.npy"), allow_pickle=True).item()
f = open(join(detection_dir, "X_prediction_results.csv"), 'r')
f.__next__()
csv_file = csv.reader(f)
x_axis = []
y_axis = []
mag = []
nms = []
for line in csv_file:
    evn = line[0]
    catalog_nm = 'GD.'+evn.split('_')[1]
    sta = evn.split('.')[0]
    if (len(line[14]) != 0) and (catalog_nm in dist) and (catalog_nm in event):
        x_axis.append(dist[catalog_nm][sta])
        y_axis.append(float(line[14]))
        mag.append(event[catalog_nm]['ML'])
        nms.append(catalog_nm)
x_axis = np.array(x_axis)*111
mag2 = list(enumerate(mag))
splitn = 4
magnify_num = 0.5
for ii in range(splitn):
    if splitn == 1:
        ind = [i for i, x in mag2 if (x < magnify_num * (ii + 1)) and (x >= magnify_num * ii)]
    elif splitn != 1 and (ii <= splitn-1):
        ind = [i for i, x in mag2 if (x < magnify_num*(ii+1)) and (x >= magnify_num*ii)]
        label = 'ML: (' + str(magnify_num * ii) + ',' + str(magnify_num * (ii + 1)) + ')'
    else:
        ind = [i for i, x in mag2 if (x >= magnify_num*ii)]
        label = 'ML: (' + str(magnify_num * ii) + ',' + '~ )'
    if len(ind) == 0:
        continue
    xylist = [[x_axis[i], y_axis[i]] for i in range(len(x_axis)) if (i in ind) and (x_axis[i] < 180)]
    # el = [[nms[i], x_axis[i], y_axis[i]] for i in range(len(x_axis)) if (i in ind) and (x_axis[i]>120)]
    xyarr = np.array(xylist)
    xarr = xyarr[:, 0]
    yarr = xyarr[:, 1]
    if ii == 0:
        plt.subplot(2, 2, ii+1)
        spot = plt.scatter(xarr, yarr, s=30, c='tab:red', alpha=0.2, label=label)
    elif ii == 1:
        plt.subplot(2, 2, ii + 1)
        spot = plt.scatter(xarr, yarr, s=30, c='tab:pink', alpha=0.2, label=label)
    elif ii == 2:
        plt.subplot(2, 2, ii + 1)
        spot = plt.scatter(xarr, yarr, s=30, c='tab:blue', alpha=0.2, label=label)
    elif ii == 3:
        plt.subplot(2, 2, ii + 1)
        spot = plt.scatter(xarr, yarr, s=30, c='tab:brown', alpha=0.2, label=label)
    plt.ylim([0, 80])
    plt.xlim([0, 200])
    plt.xlabel('Distance/km')
    plt.ylabel('SNR/db')
    plt.legend(handles=[spot], loc='upper right')
# plt.scatter(x_axis, y_axis, c=mag, s=20, cmap='hsv', alpha=0.75)
plt.tight_layout()
plt.savefig('dist_snr_EQT.png', dpi=150)
plt.close()