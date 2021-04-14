#!/usr/bin/env python
# coding: utf-8
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read
import os
import time
import glob


def gmt_inputer(catalogfile, stfile, eqfile):
    print('gmt_inputer')
    catalog = np.load(catalogfile, allow_pickle=True).item()
    stfile = open(stfile, 'w')
    eqfile = open(eqfile, 'w')
    # get all station locations
    stnm = []
    for evn in catalog['phase']:
        for st in catalog['phase'][evn]:
            if st not in stnm:
                sac_root = ('../../raw_data/XFJ/xfjml0_sac')
                sac_dir = evn + '.SAC'
                files = glob.glob(r'%s/*%s*' % (os.path.join(sac_root, sac_dir), st))
                for file in files:
                    sac_data = read(file)
                    stlo = sac_data[0].stats.sac['stlo']
                    stla = sac_data[0].stats.sac['stla']
                    if sac_data[0].stats.network == 'GD':
                        stfile.write(st.ljust(4) + '%8.3f' % stla + '%8.3f' % stlo + '\n')
                        stnm.append(st)
                    break
    stfile.close()
    for evn in catalog['head']:
        eqlon = catalog['head'][evn]['lon']
        eqlat = catalog['head'][evn]['lat']
        eqmag = catalog['head'][evn]['ML']
        eqfile.write(evn + '%8.3f' % eqlat + '%8.3f' % eqlon + '%8.2f' % eqmag + '\n')
        # breakpoint()
    eqfile.close()


def phasenum_statistical(catalogfile):
    catalog = np.load(catalogfile, allow_pickle=True).item()
    stnum = 0
    phnum = 0
    for evn in catalog['phase']:
        for st in catalog['phase'][evn]:
            stnum += 1
            for ph in catalog['phase'][evn][st]:
                phnum += 1
    print('STnum:' + str(stnum) + '\n')
    print('phnum:' + str(phnum) + '\n')


if __name__ == '__main__':
    start = time.process_time()
    # gmt_inputer(catalogfile='../data/xfj/catalog.npy', stfile='../gmt/st.dat', eqfile='../gmt/eq.dat')
    phasenum_statistical(catalogfile='../data/xfj/catalog.npy')
    end = time.process_time()
    print(end - start)
