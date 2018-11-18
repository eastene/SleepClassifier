import tensorflow as tf

from os import path

tf.logging.set_verbosity(tf.logging.INFO)

"""
*
*  Input Pipeline Flags
*
"""
tf.flags.DEFINE_integer("num_parallel_readers", 8, "number of parallel I/O threads")
tf.flags.DEFINE_integer("shuffle_buffer_size", 100, "size (in batches) of in-memory buffer for dataset shuffling")
tf.flags.DEFINE_integer("num_parallel_calls", 8, "number of parallel dataset parsing threads "
                                                 "(recommended to be equal to number of CPU cores")
tf.flags.DEFINE_integer("prefetch_buffer_size", 100,
                        "size (in batches) of in-memory buffer to prefetch records before parsing")

"""
*
*  Representation Learner Flags
*
"""
tf.flags.DEFINE_integer("num_epochs_pretrain", 100, "number of epochs for pre-training")
tf.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.flags.DEFINE_float("learn_rate_pre", 0.0001, "learning rate for pretraining")

"""
*
*  Sequence Residual Learner Flags
*
"""
tf.flags.DEFINE_integer("num_epochs_finetune", 200, "number of epochs for fine tuning")
tf.flags.DEFINE_integer("sequence_batch_size", 10, "batch size used in finetuning on sequence data")
tf.flags.DEFINE_integer("sequence_length", 25, "length of each sequence fed into the LSTM from the sequence data")
tf.flags.DEFINE_float("learn_rate_fine", 0.000001, "learning rate for pretraining")

"""
*
*  Overall Model Flags
*
"""
tf.flags.DEFINE_integer("sampling_rate", 2000, "sampling rate used to generate signal (hz)")
tf.flags.DEFINE_integer("s_per_epoch", 30, "seconds of signal data considered as a single epoch")
tf.flags.DEFINE_string("checkpoint_dir", path.join(path.dirname(path.realpath(__file__)), "tmp/"),
                       "directory in which to save model parameters while training")

"""
*
*  Data Flags
*
"""
tf.flags.DEFINE_string("data_dir", path.join(path.dirname(path.realpath(__file__)), "data/"),
                       "Path to directory containing data files")
tf.flags.DEFINE_string("tfrecord_dir", path.join(path.dirname(path.realpath(__file__)), "data/"),
                       "Path to directory containing data in .tfrecord format "
                       "for pretraining (if different from data_dir)")
tf.flags.DEFINE_float("test_split", 0.3,
                      "ratio of data to set aside for evaluation "
                      "(signal epochs in pretraining and signal files in finetuning")
tf.flags.DEFINE_string("file_pattern", "*.csv", "file pattern of data files containing original signals")

FLAGS = tf.flags.FLAGS
