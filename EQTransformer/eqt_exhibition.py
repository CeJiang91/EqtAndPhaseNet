from __future__ import division
from __future__ import print_function

import matplotlib
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.python.framework import composite_tensor
from tensorflow.python.keras import backend

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import time
import copy
from os import listdir
import os
from EQTransformer.core.EqT_utils import generate_arrays_from_file
from EQTransformer.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
from tensorflow.python.framework import ops
from tqdm import tqdm
from tensorflow.python.util import nest
import contextlib
import sys
import warnings

warnings.filterwarnings("ignore")
from EQTransformer.core.predictor import predictor

try:
    f = open('setup.py')
    for li, l in enumerate(f):
        if li == 8:
            EQT_VERSION = l.split('"')[1]
except Exception:
    EQT_VERSION = None


def run_predictor():
    predictor(input_dir='../processed_data/1daytest',
              input_model='./data/EqT_model.h5',
              output_dir='../results/data/thiscanbedelete/',
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
              number_of_cpus=16,
              keepPS=False,
              spLimit=60)


def model_exhibit(input_dir='../processed_data/1daytest',
                  input_model='./data/EqT_model.h5',
                  output_dir='../results/data/thiscanbedelete/',
                  output_probabilities=False,
                  detection_threshold=0.3,
                  P_threshold=0.1,
                  S_threshold=0.1,
                  number_of_plots=10,
                  plot_mode='time',
                  estimate_uncertainty=False,
                  number_of_sampling=5,
                  loss_weights=[0.03, 0.40, 0.58],
                  loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                  input_dimention=(6000, 3),
                  normalization_mode='std',
                  batch_size=500,
                  gpuid=None,
                  gpu_limit=None,
                  number_of_cpus=16,
                  use_multiprocessing=True,
                  keepPS=False,
                  spLimit=60):
    args = {
        "input_dir": input_dir,
        "input_hdf5": None,
        "input_csv": None,
        "input_model": input_model,
        "output_dir": output_dir,
        "output_probabilities": output_probabilities,
        "detection_threshold": detection_threshold,
        "P_threshold": P_threshold,
        "S_threshold": S_threshold,
        "number_of_plots": number_of_plots,
        "plot_mode": plot_mode,
        "estimate_uncertainty": estimate_uncertainty,
        "number_of_sampling": number_of_sampling,
        "loss_weights": loss_weights,
        "loss_types": loss_types,
        "input_dimention": input_dimention,
        "normalization_mode": normalization_mode,
        "batch_size": batch_size,
        "gpuid": gpuid,
        "gpu_limit": gpu_limit,
        "number_of_cpus": number_of_cpus,
        "use_multiprocessing": use_multiprocessing,
        "keepPS": keepPS,
        "spLimit": spLimit
    }

    class DummyFile(object):
        file = None

        def __init__(model, file):
            model.file = file

        def write(model, x):
            # Avoid print() second call (useless \n)
            if len(x.rstrip()) > 0:
                tqdm.write(x, file=model.file)

    @contextlib.contextmanager
    def nostdout():
        save_stdout = sys.stdout
        sys.stdout = DummyFile(sys.stdout)
        yield
        sys.stdout = save_stdout

    print('============================================================================')
    print('Running EqTransformer ', str(EQT_VERSION))

    print(' *** Loading the model ...', flush=True)
    model = load_model(args['input_model'],
                       custom_objects={'SeqSelfAttention': SeqSelfAttention,
                                       'FeedForward': FeedForward,
                                       'LayerNormalization': LayerNormalization,
                                       'f1': f1
                                       })
    model.compile(loss=args['loss_types'],
                  loss_weights=args['loss_weights'],
                  optimizer=Adam(lr=0.001),
                  metrics=[f1])
    print('*** Loading is complete!', flush=True)
    station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split('/')[-1] != '.DS_Store'];
    station_list = sorted(set(station_list))
    for ct, st in enumerate(station_list):
        args['input_hdf5'] = args['input_dir'] + '/' + st + '.hdf5'
        args['input_csv'] = args['input_dir'] + '/' + st + '.csv'
    df = pd.read_csv(args['input_csv'])
    prediction_list = df.trace_name.tolist()
    fl = h5py.File(args['input_hdf5'], 'r')
    list_generator = generate_arrays_from_file(prediction_list, args['batch_size'])
    # -------------------
    sa = np.array(fl['data']['XFJ.GD_201102261101.0001_EV']).reshape(1, 6000, 3)
    sout = model(sa)
    # ----------------------code from network._run_internal_graph
    # Dictionary mapping reference tensors to computed tensors.
    inputs = sa

    def _convert_non_tensor(x):
        # Don't call `ops.convert_to_tensor_v2` on all `inputs` because
        # `SparseTensors` can't be converted to `Tensor`.
        if isinstance(x, (np.ndarray, float, int)):
            return ops.convert_to_tensor_v2(x)
        return x

    inputs = nest.map_structure(_convert_non_tensor, inputs)
    inputs = model._maybe_cast_inputs(inputs)
    training = None
    mask = None
    convert_kwargs_to_constants = False
    tensor_dict = {}
    for x, y in zip(model.inputs, sa):
        y = model._conform_to_reference_input(y, ref_input=x)
        x_id = str(id(x))
        tensor_dict[x_id] = [y] * model._tensor_usage_count[x_id]

    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    # Ignore the InputLayers when computing the graph.
    inputs = model._flatten_to_reference_inputs(inputs)
    if mask is None:
        masks = [None for _ in range(len(inputs))]
    else:
        masks = model._flatten_to_reference_inputs(mask)
    for input_t, mask in zip(inputs, masks):
        input_t._keras_mask = mask

    # Dictionary mapping reference tensors to computed tensors.
    tensor_dict = {}
    for x, y in zip(model.inputs, inputs):
        y = model._conform_to_reference_input(y, ref_input=x)
        x_id = str(id(x))
        tensor_dict[x_id] = [y] * model._tensor_usage_count[x_id]

    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    # Ignore the InputLayers when computing the graph.
    depth_keys = depth_keys[1:]

    for depth in depth_keys:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:
            # This is always a single layer, never a list.
            layer = node.outbound_layer

            if all(
                    str(id(tensor)) in tensor_dict
                    for tensor in nest.flatten(node.input_tensors)):

                # Call layer (reapplying ops to new inputs).
                computed_tensors = nest.map_structure(
                    lambda t: tensor_dict[str(id(t))].pop(), node.input_tensors)

                # Ensure `training` arg propagation if applicable.
                kwargs = copy.copy(node.arguments) if node.arguments else {}
                if convert_kwargs_to_constants:
                    kwargs = model._map_tensors_to_constants(kwargs)

                argspec = model._layer_call_argspecs[layer].args
                if 'training' in argspec:
                    if 'training' not in kwargs or kwargs['training'] is None:
                        kwargs['training'] = training
                    elif (type(kwargs['training']) is ops.Tensor and  # pylint: disable=unidiomatic-typecheck
                          any([
                              kwargs['training'] is x
                              for x in backend._GRAPH_LEARNING_PHASES.values()
                          ])):
                        kwargs['training'] = training  # Materialize placeholder.

                # Map Keras tensors in kwargs to their computed value.
                def _map_tensor_if_from_keras_layer(t):
                    if (isinstance(t,
                                   (ops.Tensor, composite_tensor.CompositeTensor)) and
                            hasattr(t, '_keras_history')):
                        t_id = str(id(t))
                        return tensor_dict[t_id].pop()
                    return t

                kwargs = nest.map_structure(_map_tensor_if_from_keras_layer, kwargs)

                # Compute outputs.
                output_tensors = layer(computed_tensors, **kwargs)

                # Update tensor_dict.
                for x, y in zip(
                        nest.flatten(node.output_tensors), nest.flatten(output_tensors)):
                    x_id = str(id(x))
                    tensor_dict[x_id] = [y] * model._tensor_usage_count[x_id]
        # -----------
        for kk in range(0, output_tensors[0].shape[1]):
            plt.plot(output_tensors[0][:, kk] + kk)
        plt.savefig(os.path.join('./eqt_layer_image', '%003d' % depth + '.png'))
        plt.close()
    # ---------------------------------------------
    plt.plot(sa[0, :, 1])
    plt.savefig('./eqt_layer_image/input.png')
    plt.close()


if __name__ == '__main__':
    start = time.process_time()
    # run_predictor()
    model_exhibit()
    end = time.process_time()
    print(end - start)
