"""Processing module for radar signals"""

import numpy as np
from scipy import fft

def range_doppler_fft(data, rwin=None, dwin=None, rn=None, dn=None):
    """
    Range-Doppler processing.

    Args:
        data (numpy.3darray): Baseband data [channels, pulses, adc_samples]
        rwin (numpy.1darray): Window for Range FFT, length should be equal to adc_samples.
        dwin (numpy.1darray): Window for Doppler FFT, length should be equal to adc_samples.
        rn (int): Range FFT size.
        dn (int): Doppler FFT size.
    Returns:
        numpy.3darray: A 3D array of range dopple map 
    """
    shape = np.shape(data)

    if rwin is None:
        rwin = 1
    else:
        rwin = np.tile(rwin[np.newaxis, np.newaxis, ...], (shape[0], shape[1], 1))

    r_fft = fft.fft(data * rwin, n=rn, axis=2) 

    shape = np.shape(r_fft)

    if dwin is None:
        dwin = 1
    else:
        dwin = np.tile(dwin[np.newaxis, ..., np.newaxis], (shape[0], 1, shape[2])) 
    
    d_fft = fft.fft(r_fft * dwin, n=dn, axis=1)

    return d_fft