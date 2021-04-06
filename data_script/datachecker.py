import h5py
import os


def eqtchecker(input_dir):
    f = h5py.File(os.path.join(input_dir, "traces.hdf5"), "r")
    d = f['data']
    # f.close()
    for ev in d:
        z = d[ev][:, 2]
    breakpoint()


if __name__ == '__main__':
    eqtchecker(r'../processed_data/XFJ1121/train_data')
