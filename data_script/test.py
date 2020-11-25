from obspy import read
import numpy as np
import matplotlib.pyplot as plt
import os
# aa = np.load('/home/jiangce/work/Research_on_EQTransformer&PhaseNet_/raw_data/output/waveform_xfj/XFJ_GD.201512010907.0001.npz')
# data = aa['data']
# plt.plot(data[:, 0])
# plt.show()
# plt.close()

for root, dirs, files in os.walk('./test_sac_min'):
    dirs.sort()
    for f in sorted(files):
        print(root)
        print(f)
    # for name in (dirs):
    #     print(os.path.join(root, name))
        # break