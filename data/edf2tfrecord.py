import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_edf


data_file = read_raw_edf('pkg_n1.edf')
sampling_rate = data_file.info['sfreq']
channel = data_file.to_data_frame()['Cz']
channel.index = np.arange(len(channel))
print(channel)