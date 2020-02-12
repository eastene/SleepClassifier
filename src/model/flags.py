import os.path
import argparse

parser = argparse.ArgumentParser()

"""
*
*  Representation Learner Flags
*
"""
parser.add_argument("--num_epochs_pretrain", type=int, default=100, help="number of epochs for pre-training",
                    required=False)
parser.add_argument("--batch_size", type=int, default=50, help="batch size", required=False)
parser.add_argument("--learn_rate_pre", type=float, default=0.0001, help="learning rate for pretraining",
                    required=False)

"""
*
*  Sequence Residual Learner Flags
*
"""
parser.add_argument("--num_epochs_finetune", type=int, default=200, help="number of epochs for fine tuning",
                    required=False)
parser.add_argument("--sequence_batch_size", type=int, default=10,
                    help="batch size used in finetuning on sequence data",
                    required=False)
parser.add_argument("--sequence_length", type=int, default=25,
                    help="length of each sequence fed into the LSTM from the sequence data", required=False)
parser.add_argument("--learn_rate_fine", type=float, default=0.000001, help="learning rate for finetuning",
                    required=False)

"""
*
*  Overall Model Flags
*
"""
parser.add_argument("--sampling_rate", type=int, default=1000, help="sampling rate used to generate signal (hz)",
                    required=False)
parser.add_argument("--downsample_rate", type=int, default=5, help="rate at which to downsample the input signal",
                    required=False)
parser.add_argument("--s_per_epoch", type=int, default=30, help="seconds of signal data considered as a single epoch",
                    required=False)
parser.add_argument("--checkpoint_dir", type=str,
                    default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp"),
                    help="directory in which to save model parameters while training", required=False)

"""
*
*  Data Flags
*
"""
parser.add_argument("--data_dir", type=str,
                    default=os.path.abspath(
                        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "data")),
                    help="Path to top-level directory containing data files", required=False)
parser.add_argument("--test_split", type=float, default=0.3, help="ratio of data to set aside for evaluation",
                    required=False)
parser.add_argument("--val_split", type=float, default=0.1, help="ratio of data to set aside for validation",
                    required=False)

"""
*
* Output Flags
*
"""
parser.add_argument("--cnfsn_mat", type=bool, default=False,
                    help="print confusion matrix on last evaluation after training model", required=False)
parser.add_argument("--plot_loss", type=bool, default=False, help="plot the loss over training", required=False)

FLAGS = parser.parse_args()

EFFECTIVE_SAMPLE_RATE = FLAGS.sampling_rate // max(FLAGS.downsample_rate, 1)
