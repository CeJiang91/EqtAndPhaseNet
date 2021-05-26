import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


'''
    Before running this script, you should cancel the comment in line 533 of run.py
'''


def trace_example(file='../../../work/SeismicData/XFJ1121/1daytest'
                       '/1day.phnetinput/waveform1_xfj/XFJ_GD.201112231459.0002.npz'):
    tr = np.load(file)['data']
    pred_prob = np.load('trace_prob.npy', allow_pickle=True)
    ########################################## ploting only in time domain
    fig = plt.figure()
    widths = [1]
    heights = [1.6, 1.6, 1.6, 2.5]
    spec5 = fig.add_gridspec(ncols=1, nrows=4, width_ratios=widths,
                             height_ratios=heights)

    ax = fig.add_subplot(spec5[0, 0])
    ymin, ymax = ax.get_ylim()
    a = pred_prob[0, :, 0, 1]
    pt=np.argmax(a)
    b = pred_prob[0, :, 0, 2]
    st = np.argmax(b)
    plt.plot(tr[:, 0], 'k')
    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
    x = np.arange(6000)
    plt.xlim(500, 2000)
    plt.ylabel('Amplitude\nCounts')
    plt.rcParams["figure.figsize"] = (8, 6)
    legend_properties = {'weight': 'bold'}
    plt.title('Trace Name: ' + file.split('/')[-1])
    ax = fig.add_subplot(spec5[1, 0])
    plt.plot(tr[:, 1], 'k')
    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
    plt.xlim(500, 2000)
    plt.ylabel('Amplitude\nCounts')
    ax = fig.add_subplot(spec5[2, 0])
    plt.plot(tr[:, 2], 'k')
    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
    plt.xlim(500, 2000)
    plt.ylabel('Amplitude\nCounts')
    ax.set_xticks([])
    ax = fig.add_subplot(spec5[3, 0])
    x = np.linspace(0, tr.shape[0], tr.shape[0], endpoint=True)
    # plt.plot(x, pred_prob[0, :, 0, 0], '--', color='g', alpha=0.5, linewidth=1.5, label='Earthquake')
    plt.plot(x, pred_prob[0, :, 0, 1], '--', color='b', alpha=0.5, linewidth=1.5, label='P_arrival')
    plt.plot(x, pred_prob[0, :, 0, 2], '--', color='r', alpha=0.5, linewidth=1.5, label='S_arrival')
    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
    plt.tight_layout()
    plt.ylim((-0.1, 1.1))
    plt.xlim(500, 2000)
    plt.ylabel('Probability')
    plt.xlabel('Sample')
    plt.legend(loc='lower center', bbox_to_anchor=(0., 1.17, 1., .102), ncol=3, mode="expand",
               prop=legend_properties, borderaxespad=0., fancybox=True, shadow=True)
    plt.yticks(np.arange(0, 1.1, step=0.2))
    axes = plt.gca()
    axes.yaxis.grid(color='lightgray')

    font = {'family': 'serif',
            'color': 'dimgrey',
            'style': 'italic',
            'stretch': 'condensed',
            'weight': 'normal',
            'size': 12,
            }
    fig.tight_layout()
    fig.savefig(file.split('/')[-1] + '.png', dpi=600)
    plt.close(fig)
    plt.clf()


if __name__ == '__main__':
    trace_example()
