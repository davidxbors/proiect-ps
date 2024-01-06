"""Processing module for radar signals"""

import numpy as np
from scipy import fft, linalg
from scipy.signal import find_peaks

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

def doa_music(covmat, no_targets, angle_scan, spacing):
    # get number of elements in the sensor array
    # and create a linearly spaced array with them
    n = np.shape(covmat)[0]
    n_array = np.linspace(0, (n - 1) * spacing, n)


    # get eigenvectors from covariance matrix
    _, eig = linalg.eigh(covmat)
    # select the columns coresponding to the noise
    noise = eig[:, :-no_targets]

    # create arrays with sensor array positions and scan angles in radians
    arr_grid, ang_grid = np.meshgrid(n_array, np.radians(angle_scan), indexing="ij")
    # calculate the steering vector for each array element and each scan angle
    # normalize using sqrt of the number of elements in array
    steering = np.exp(1j * 2 * np.pi * arr_grid * np.sin(ang_grid)) / np.sqrt(n)

    # project the steering vector onto the noise subspace
    # calculates the magnitude across each column and take the inverse of these norms
    spectrum = 1 / linalg.norm((noise.T.conj() @ steering), axis=0)

    # convert to DB
    spec_db = 10 * np.log10(spectrum / spectrum.min())

    # find local maxima in the MUSIC spectrum
    local_maxima, _ = find_peaks(spec_db)
    # select the no_targets highest peaks as these should be our targets
    doa_idx = local_maxima[np.argsort(spec_db[local_maxima])[-no_targets:]]

    return angle_scan[doa_idx], doa_idx, spec_db