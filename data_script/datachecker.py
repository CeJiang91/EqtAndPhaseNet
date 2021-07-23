import h5py
import os
import matplotlib.pyplot as plt
import numpy as np


def awgn(x, snr):
    """

    Parameters
    ----------
    x: original signal
    snr: signal noise ratio

    Returns signal with gaussian noise
    -------

    """
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise


def hdf5_addnoise_exhibition(input_dir, event_id, SNR):
    f = h5py.File(os.path.join(input_dir, "traces.hdf5"), "r")
    d = np.array(f['data'][event_id])
    for i in range(0, 3):
        print(i)
        d[:, i] = awgn(d[:, i], SNR)
    d2 = d[:, 1]
    plt.subplot(211)
    plt.plot(d2)
    plt.subplot(212)
    plt.plot(d[:, 1])
    plt.show()
    f.close()


def hdf5_validation(input_dir, output_dir, num_of_plots):
    h5f = h5py.File(os.path.join(input_dir, "traces.hdf5"), "r")
    data = h5f['data']
    num = 0
    for ev in data:
        # if ev.split('.')[0] != 'XFJ':
        #     continue
        for i in range(3):
            plt.plot(data[ev][:, i] + i)
        plt.show()
        plt.savefig(os.path.join(output_dir, ev + '.png'))
        plt.close()
        num += 1
        # set numer of plots
        if num > num_of_plots:
            break
    h5f.close()


def catalog_exhibition(catalog_file, snr_file=None):
    catalog = np.load(catalog_file, allow_pickle=True).item()
    head = catalog['head']
    ML = []
    for ev in head:
        ML.append(head[ev]['ML'])
    print(len(ML))
    plt.figure()
    num_bins = 46
    plt.hist(ML, num_bins, edgecolor='white', linewidth=0.5,
             facecolor='cornflowerblue', range=[-0.5, 4], alpha=1, label='ML')
    plt.xlabel('Magnitude')
    plt.ylabel('Number of events')
    plt.savefig('ML_number')
    plt.close()
    if snr_file:
        SNR = []
        snr = np.load(snr_file, allow_pickle=True)['snr'].item()
        phase = catalog['phase']
        for evn in phase:
            for st in snr[evn]:
                SNR.append(snr[evn][st])
        plt.figure()
        num_bins = 71
        plt.hist(SNR, num_bins, edgecolor='red', linewidth=1,
                 facecolor='red', range=[-10, 60], alpha=0.6, label='ML')
        plt.xlabel('SNR(dB)')
        plt.ylabel('Number of traces')
        plt.savefig('SNR_number')
        plt.close()
        breakpoint()


if __name__ == '__main__':
    hdf5_validation(input_dir='/media/jiangce/work_disk/project/SeismicData/XFJ1121/eqtinputv2/tenyears_set',
                    output_dir='/media/jiangce/work_disk/project/SeismicData/XFJ1121/__valipic__',
                    num_of_plots=300)
    # hdf5_addnoise_exhibition(input_dir=r'/media/jiangce/My Passport/work/SeismicData/XFJ1121/eqtinput/tenyears_set',
    #                          event_id='XFJ.GD_201112231459.0002_EV',
    #                          SNR=0)
    # catalog_exhibition(catalog_file=r'/media/jiangce/My Passport/work/SeismicData/XFJ1121/catalog.npy',
    #                    snr_file='/media/jiangce/My Passport/work/SeismicData/XFJ1121/snr.npz')
