import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("num_parallel_readers", 8, "number of parallel I/O threads")
tf.flags.DEFINE_integer("shuffle_buffer_size", 100, "size (in batches) of in-memory buffer for dataset shuffling")
tf.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.flags.DEFINE_integer("num_parallel_calls", 8, "number of parallel dataset parsing threads "
                                                "(recommended to be equal to number of CPU cores")
tf.flags.DEFINE_integer("prefetch_buffer_size", 100, "size (in batches) of in-memory buffer to prefetch records before parsing")
tf.flags.DEFINE_integer("num_epochs_pretrain", 1, "number of epochs for pre-training")
tf.flags.DEFINE_integer("num_epochs_finetune", 1, "number of epochs for fine tuning")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/existing_model/", "directory in which to save model parameters while training")

FLAGS = tf.flags.FLAGS
TRAIN_FILE_PATTERN = "/home/evan/PycharmProjects/SleepClassifier/data/existing_solution/prepared_tf/SC40*.tfrecord"
EVAL_FILE_PATTERN = "/home/evan/PycharmProjects/SleepClassifier/data/existing_solution/prepared_tf/SC41[0-7]*.tfrecord"
TEST_FILE_PATTERN = "/home/evan/PycharmProjects/SleepClassifier/data/existing_solution/prepared_tf/SC41[8-9]*.tfrecord"