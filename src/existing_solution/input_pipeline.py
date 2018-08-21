import tensorflow as tf

from src.existing_solution.flags import FLAGS

class InputPipeline:

    def __init__(self, file_pattern):
        self.file_pattern = file_pattern
        self.dataset_iterator = self.input_fn().make_initializable_iterator()

    def initializer(self):
        return self.dataset_iterator.initializer

    def parse_fn(self, example):
        # format of each training example
        example_fmt = {
            "signal": tf.FixedLenFeature((1,3000), tf.float32),
            "label": tf.FixedLenFeature((), tf.int64, -1),
            "sampling_rate": tf.FixedLenFeature((), tf.float32, 0.0)
        }

        parsed = tf.parse_single_example(example, example_fmt)

        return parsed['signal'], parsed['label'], parsed['sampling_rate']

    def input_fn(self):
        files = tf.data.Dataset.list_files(file_pattern=self.file_pattern, shuffle=False)

        # interleave reading of dataset for parallel I/O
        dataset = files.apply(
            tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers
            )
        )

        dataset = dataset.cache()

        # shuffle data
        dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)

        # parse the data and prepares the batches in parallel (helps most with larger batches)
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                map_func=self.parse_fn, batch_size=FLAGS.batch_size
            )
        )

        # prefetch data so that the CPU can prepare the next batch(s) while the GPU trains
        # recommmend setting buffer size to number of training examples per training step
        dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

        return dataset

    def next_elem(self):
        return self.dataset_iterator.get_next()