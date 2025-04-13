"""
from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Example: sampling frequency fs = 100 Hz, cutoff frequency = 1 Hz
filtered_signal = butter_lowpass_filter(raw_signal, cutoff=1.0, fs=100.0, order=4)


##################################


from scipy.signal import medfilt

filtered_signal = medfilt(raw_signal, kernel_size=5)



###########################

import numpy as np

# Example: a simple moving average filter
window_size = 5  
kernel = np.ones(window_size) / window_size
smoothed_signal = np.convolve(raw_signal, kernel, mode='same')


###########################

import pywt
import numpy as np

# Example parameters
wavelet = 'db4'
level = 2

coeffs = pywt.wavedec(raw_signal, wavelet, level=level)
# Apply thresholding to the detail coefficients
threshold = 0.5 * np.max(coeffs[-1])
coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
smoothed_signal = pywt.waverec(coeffs, wavelet)


#############################
"""
import logging
import numpy as np
import time
#from keras.layers import Dense, Activation
#from keras.models import Sequential, load_model
#from keras.optimizers import Adam
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

