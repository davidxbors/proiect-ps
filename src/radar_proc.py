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

def root_music(covmat, no_targets, spacing):
    # get number of elements in the sensor array
    n = np.shape(covmat)[0] 
    
    # get eigenvectors from covariance matrix
    _, eig = linalg.eigh(covmat)
    # select the columns coresponding to the noise
    noise = eig[:, :-no_targets]

    noise_matrix = noise @ noise.T.conj()
    coef = np.zeros((n - 1,), dtype=np.complex_)

    coef[:n - 1] = [np.trace(noise_matrix, i) for i in range(1, n)]
    coef = np.hstack((coef[::-1], np.trace(noise_matrix), coef.conj()))

    # get roots of polynom
    roots = np.roots(coef)

    # mask to identify the roots that are inside or on the unit circle
    mask = np.abs(roots) <= 1

    # if a point is on the unit circle we need to find the closest point and remove it
    for _, i in enumerate(np.where(np.abs(roots) == 1)[0]):
        mask[np.argsort(np.abs(roots - roots[i]))[1]] = False

    # exclude the closest pairs
    roots = roots[mask]
    # sort the roots by how close they are to the unit circle
    sorted_idx = np.argsort(1.0 - np.abs(roots))
    # get the no_targets roots closest to the circle and compute their angles' sines
    sins = np.angle(roots[sorted_idx[:no_targets]]) / (2 * np.pi * spacing)

    # return the angles in degrees
    return np.sort(np.degrees(np.arcsin(sins)))

def espirit(covmat, no_targets, spacing):
    _, eig_vectors = linalg.eigh(covmat)

    # no_targets largest eigenvalues from the eigenvector matrix
    signal = eig_vectors[:, -no_targets:]

    # calculate rotational invariance among the signal subspaces
    phi = linalg.pinv(signal[0:-1]) @ signal[1:]
    eig_values = linalg.eigvals(phi)

    # compute phases of eigenvalues, divides them by pi and scales them
    # then it returns the angle in degreees
    return np.sort(np.degrees(np.arcsin(np.angle(eig_values) / np.pi / (spacing / 0.5))))
