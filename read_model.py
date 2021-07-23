from model import Model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse
from model import Model
from data_reader import Config, DataReader, DataReader_test, DataReader_pred, DataReader_mseed
from util import *
from keras.models import load_model
from EQTransformer.core.EqT_utils import SeqSelfAttention,FeedForward,LayerNormalization,f1
from keras.optimizers import Adam


def read_args():

  parser = argparse.ArgumentParser()

  parser.add_argument("--mode",
                      default="train",
                      help="train/valid/test/debug")

  parser.add_argument("--epochs",
                      default=100,
                      type=int,
                      help="number of epochs (default: 10)")

  parser.add_argument("--batch_size",
                      default=200,
                      type=int,
                      help="batch size")

  parser.add_argument("--learning_rate",
                      default=0.01,
                      type=float,
                      help="learning rate")

  parser.add_argument("--decay_step",
                      default=-1,
                      type=int,
                      help="decay step")

  parser.add_argument("--decay_rate",
                      default=0.9,
                      type=float,
                      help="decay rate")

  parser.add_argument("--momentum",
                      default=0.9,
                      type=float,
                      help="momentum")

  parser.add_argument("--filters_root",
                      default=8,
                      type=int,
                      help="filters root")

  parser.add_argument("--depth",
                      default=5,
                      type=int,
                      help="depth")

  parser.add_argument("--kernel_size",
                      nargs="+",
                      type=int,
                      default=[7, 1],
                      help="kernel size")

  parser.add_argument("--pool_size",
                      nargs="+",
                      type=int,
                      default=[4, 1],
                      help="pool size")

  parser.add_argument("--drop_rate",
                      default=0,
                      type=float,
                      help="drop out rate")

  parser.add_argument("--dilation_rate",
                      nargs="+",
                      type=int,
                      default=[1, 1],
                      help="dilation rate")

  parser.add_argument("--loss_type",
                      default="cross_entropy",
                      help="loss type: cross_entropy, IOU, mean_squared")

  parser.add_argument("--weight_decay",
                      default=0,
                      type=float,
                      help="weight decay")

  parser.add_argument("--optimizer",
                      default="adam",
                      help="optimizer: adam, momentum")

  parser.add_argument("--summary",
                      default=True,
                      type=bool,
                      help="summary")

  parser.add_argument("--class_weights",
                      nargs="+",
                      default=[1, 1, 1],
                      type=float,
                      help="class weights")

  parser.add_argument("--log_dir",
                      default="log",
                      help="Tensorboard log directory (default: log)")

  parser.add_argument("--model_dir",
                      default=None,
                      help="Checkpoint directory (default: None)")

  parser.add_argument("--num_plots",
                      default=10,
                      type=int,
                      help="Plotting trainning results")

  parser.add_argument("--tp_prob",
                      default=0.3,
                      type=float,
                      help="Probability threshold for P pick")

  parser.add_argument("--ts_prob",
                      default=0.3,
                      type=float,
                      help="Probability threshold for S pick")

  parser.add_argument("--input_length",
                      default=None,
                      type=int,
                      help="input length")

  parser.add_argument("--input_mseed",
                      action="store_true",
                      help="mseed format")

  parser.add_argument("--data_dir",
                      default="./dataset/waveform_pred/",
                      help="Input file directory")

  parser.add_argument("--data_list",
                      default="./dataset/waveform.csv",
                      help="Input csv file")

  parser.add_argument("--train_dir",
                      default="./dataset/waveform_train/",
                      help="Input file directory")

  parser.add_argument("--train_list",
                      default="./dataset/waveform.csv",
                      help="Input csv file")

  parser.add_argument("--valid_dir",
                      default=None,
                      help="Input file directory")

  parser.add_argument("--valid_list",
                      default=None,
                      help="Input csv file")

  parser.add_argument("--output_dir",
                      default=None,
                      help="Output directory")

  parser.add_argument("--plot_figure",
                      action="store_true",
                      help="If plot figure for test")

  parser.add_argument("--save_result",
                      action="store_true",
                      help="If save result for test")

  parser.add_argument("--fpred",
                      default="picks",
                      help="Ouput filename for test")

  args = parser.parse_args()
  return args


def read_phasenet_model():
    args = read_args()
    args.data_dir='./PhaseNet/demo/mseed'
    args.data_list='./PhaseNet/demo/fname.csv'
    args.batch=20
    coord = tf.train.Coordinator()
    with tf.compat.v1.name_scope('create_inputs'):
        data_reader = DataReader_mseed(
            data_dir=args.data_dir,
            data_list=args.data_list,
            queue_size=args.batch,
            coord=coord,
            input_length=3000)
    config = set_config(args, data_reader)
    with tf.compat.v1.name_scope('Input_Batch'):
        batch = data_reader.dequeue(args.batch_size)
    model = Model(config, batch, "pred")
    params = 0
    for var in tf.compat.v1.trainable_variables():
        if len(var.shape)>0:
            temp=1
            for i in range(len(var.shape)):
                temp = temp*var.shape[i]
            params+=temp
            # breakpoint()
    print(params)
    # breakpoint()


def read_eqt_model():
    input_model = '/media/jiangce/work_disk/project/EqtAndPhaseNet/EQTransformer/data/EqT_model.h5'
    loss_types = ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
    loss_weights = [0.03, 0.40, 0.58]
    print(' *** Loading the model ...', flush=True)
    model = load_model(input_model,
                       custom_objects={'SeqSelfAttention': SeqSelfAttention,
                                       'FeedForward': FeedForward,
                                       'LayerNormalization': LayerNormalization,
                                       'f1': f1
                                        })
    model.compile(loss = loss_types,
                  loss_weights = loss_weights,
                  optimizer = Adam(lr = 0.001),
                  metrics = [f1])
    print('*** Loading is complete!', flush=True)
    print('eqt_model')
    breakpoint()



def set_config(args, data_reader):
  config = Config()

  config.X_shape = data_reader.X_shape
  config.n_channel = config.X_shape[-1]
  config.Y_shape = data_reader.Y_shape
  config.n_class = config.Y_shape[-1]

  config.depths = args.depth
  config.filters_root = args.filters_root
  config.kernel_size = args.kernel_size
  config.pool_size = args.pool_size
  config.dilation_rate = args.dilation_rate
  config.batch_size = args.batch_size
  config.class_weights = args.class_weights
  config.loss_type = args.loss_type
  config.weight_decay = args.weight_decay
  config.optimizer = args.optimizer

  config.learning_rate = args.learning_rate
  if (args.decay_step == -1) and (args.mode == 'train'):
    config.decay_step = data_reader.num_data // args.batch_size
  else:
    config.decay_step = args.decay_step
  config.decay_rate = args.decay_rate
  config.momentum = args.momentum

  config.summary = args.summary
  config.drop_rate = args.drop_rate
  config.class_weights = args.class_weights

  return config




if __name__ == '__main__':
    # read_phasenet_model()
    read_eqt_model()