from typing import Dict, List
import numpy as np
from scipy import signal, fft, linalg
import plotly.graph_objs as go
from radarsimpy.simulator import simc
import radarsimpy
from radar_proc import range_doppler_fft
from utils import EXPORT_PATH

import radar_sim

def get_data(no_targets: int, target_angles: List[int], load_stub: str = "", save_stub: str = ""):
    """
    Get simulated data.

    Params:
        no_targets (int): number of targets to simulate
        target_angles (List[int]): list of target angles
        load_stub (str): Stub to use for loading. If None => data will be simulated live.
        save_stub (str): Stub to use for saving. If None and data was simulated => data won't be saved.
    Returns:
        radarsimpy.Radar, np.ndarray, np.ndarray: Radar object, baseband and timestamp arrays 
    """
    radar = radar_sim.create_radar(False)
    # targets = create_targets(3, [-5, -4, 45])
    targets = radar_sim.create_targets(no_targets, target_angles)

    if not load_stub:
        print("no load_stub")
        data = radar_sim.simulate(radar=radar, targets=targets)

        # cache data for further use
        if save_stub:
            radar_sim.save_data(data, save_stub)

    baseband, timestamp = radar_sim.load_data(load_stub)

    return radar, baseband, timestamp

def range_doppler(radar: radarsimpy.Radar, baseband: np.ndarray, single_target_mode: bool =True):
    """
    Use Range-Doppler processing to get the azimuth angle of the target.

    Args:
        radar (radarsimpy.Radar): Radar object. 
        baseband (np.ndarray): Baseband data.
        single_target_mode (bool, optional): Wether we have only one target or not. Defaults to True.

    Returns:
        Optional[float]: Found angle. Only in single mode.
    """
    # generate a Chebyshev window for range processing
    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=80)
    doppler_window = signal.windows.chebwin(radar.radar_prop["transmitter"].waveform_prop["pulses"], at=60)

    range_doppler = range_doppler_fft(
        baseband, rwin=range_window, dwin=doppler_window)

    no_tx = 2
    no_rx = 64

    # identify the index of the strongest target by finding the max value across the mean Doppler spectrum
    det_idx = [np.argmax(np.mean(np.abs(range_doppler[:, 0, :]), axis=0))]

    # extract the beamforming vector for the strongest target 
    bv = range_doppler[:, 0, det_idx[0]]
    bv = bv/linalg.norm(bv)

    # populate the snapshot matrix with shifted versions of the beamforming vector
    snapshots = 20
    bv_snapshot = np.zeros((no_tx*no_rx-snapshots, snapshots), dtype=complex)
    for idx in range(0, snapshots):
        bv_snapshot[:, idx] = bv[idx:(idx+no_tx*no_rx-snapshots)]

    # create covarience matrix
    covmat = np.cov(bv_snapshot.conjugate())

    # apply fft on bv conjugated and convert it to a logarithmic scale (dB)
    fft_spec = 20 * np.log10(np.abs(fft.fftshift(fft.fft(bv.conjugate(), n=1024))))

    # plot FFT
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arcsin(np.linspace(-1, 1, 1024, endpoint=False))/np.pi*180,
                            y=fft_spec,
                            name='FFT')
                )

    fig.update_layout(
        title='FFT',
        yaxis=dict(title='Amplitude (dB)'),
        xaxis=dict(title='Angle (deg)'),
        margin=dict(l=10, r=10, b=10, t=40),
    )

    # save plot
    fig.write_html(f'{EXPORT_PATH}dopler_fft.html')

    if single_target_mode:
        angles = np.arcsin(np.linspace(-1, 1, 1024, endpoint=False)) / np.pi * 180
        return angles[np.argmax(fft_spec)]

if __name__ == "__main__":
    # get radar sim data
    no_targets = 1
    target_angles = [5]
    radar, baseband, timestamp = get_data(no_targets=no_targets, target_angles=target_angles, load_stub="15")

    # perform range doppler processing
    found_target = range_doppler(radar=radar, baseband=baseband)

    if found_target:
        err = abs(5 - found_target)
        err_percent = err / 5 * 100
        print(f"Found target: {found_target}")
        print("Err: " + str(err))
        print("Err %: " + str(err_percent))