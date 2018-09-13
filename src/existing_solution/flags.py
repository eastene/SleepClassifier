import tensorflow as tf
from os import path

tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("num_parallel_readers", 8, "number of parallel I/O threads")
tf.flags.DEFINE_integer("shuffle_buffer_size", 100, "size (in batches) of in-memory buffer for dataset shuffling")
tf.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.flags.DEFINE_integer("num_parallel_calls", 8, "number of parallel dataset parsing threads "
                                                "(recommended to be equal to number of CPU cores")
tf.flags.DEFINE_integer("prefetch_buffer_size", 100, "size (in batches) of in-memory buffer to prefetch records before parsing")
tf.flags.DEFINE_integer("num_epochs_pretrain", 100, "number of epochs for pre-training")
tf.flags.DEFINE_integer("num_epochs_finetune", 200, "number of epochs for fine tuning")
tf.flags.DEFINE_integer("sequence_batch_size", 10, "batch size used in finetuning on sequence data")
tf.flags.DEFINE_integer("sequence_length", 25, "length of each sequence fed into the LSTM from the sequence data")
tf.flags.DEFINE_string("data_dir", "", "directory in which input data is located")
tf.flags.DEFINE_string("checkpoint_dir", path.join(path.dirname(path.realpath(__file__)), "tmp", ""), "directory in which to save model parameters while training")

FLAGS = tf.flags.FLAGS
FILE_PATTERN = "*.tfrecord"
