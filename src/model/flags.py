import tensorflow as tf

import os.path

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
tf.flags.DEFINE_integer("resample_rate", 0, "rate at which to downsample the input signal")
tf.flags.DEFINE_list("input_chs", 'eeg', "name of input channels (e.g. [eeg, eog] is 2 channels), input channel "
                                         "name should appear in file name and be unique to that data type")
tf.flags.DEFINE_integer("s_per_epoch", 30, "seconds of signal data considered as a single epoch")
tf.flags.DEFINE_string("checkpoint_dir", os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp/"),
                       "directory in which to save model parameters while training")

"""
*
*  Data Flags
*
"""
tf.flags.DEFINE_string("data_dir",
                       os.path.abspath(
                           os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "data/")),
                       "Path to directory containing data files")
tf.flags.DEFINE_string("seq_dir",
                       os.path.abspath(
                           os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir,
                                        "data/seqs/")),
                       "Path to directory containing sequence data as compressed numpy files")
tf.flags.DEFINE_string("test_dir", "",
                       "Path to separate test sequences (in npz format), [optional]")
tf.flags.DEFINE_string("tfrecord_dir",
                       os.path.abspath(
                           os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir,
                                        "data/tfrecords/")),
                       "Path to directory containing data in .tfrecords format "
                       "for pretraining (if different from data_dir)")
tf.flags.DEFINE_string("meta_dir", os.path.dirname(os.path.realpath(__file__)),
                       "Path to directory containing data meta info")
tf.flags.DEFINE_bool("oversample", True, "whether to oversample input to the representation learner, "
                                         "requires rebuilding the tfrecords if altered")
tf.flags.DEFINE_float("test_split", 0.3,
                      "ratio of data to set aside for evaluation "
                      "(signal epochs in pretraining and signal files in finetuning")
tf.flags.DEFINE_string("file_pattern", "*.csv", "file pattern of data files containing original signals")
tf.flags.DEFINE_integer("seq_buff_size", 15, "number of full sequences to keep in buffer at a time")


"""
*
* Output Flags
*
"""
tf.flags.DEFINE_bool("cnfsn_mat", False, "print confusion matrix on last evaluation after training model")
tf.flags.DEFINE_bool("plot_loss", False, "plot the loss over training")

FLAGS = tf.flags.FLAGS

EFFECTIVE_SAMPLE_RATE = FLAGS.sampling_rate // max(FLAGS.resample_rate, 1)
META_INFO_FNAME = 'meta_info.npz'
