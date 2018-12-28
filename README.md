# SleepClassifier

Deep learning tool for classifying sleep stages from the LIBS in-ear sleep monitoring device.

### Overview

Uses a non-linear CNN to generate features from the raw LIBS signal and combines those features with non-linear cross-channel
features from the separated raw signal (EEG and EOG signals). These features are used in combination as input to a bi-directional 
LSTM to learn stage transitions both forward and backward in time.

