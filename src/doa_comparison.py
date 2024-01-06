from typing import Dict, List
import numpy as np
from scipy import signal, fft, linalg
import plotly.graph_objs as go
from radarsimpy.simulator import simc
import radarsimpy

from radar_proc import range_doppler_fft, doa_music, root_music, espirit
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
        Optional[float, ndarray]: Found angle and covariance matrix. Only in single mode.
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

    # get peaks to return as targets
    local_maxima, _ = signal.find_peaks(fft_spec)
    doa_idx = local_maxima[np.argsort(fft_spec[local_maxima])[-no_targets:]]

    angles = np.arcsin(np.linspace(-1, 1, 1024, endpoint=False)) / np.pi * 180
    doa_idx = [angles[idx] for idx in doa_idx]

    return doa_idx, covmat

def music(covmat, no_targets):
    angle_scan = np.arange(-90, 90, 0.1)
    np_angle_scan = np.array(angle_scan)
    music_doa, music_idx, ps_db = doa_music(covmat, no_targets, np_angle_scan, 0.5) 

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=angle_scan,
                            y=ps_db,
                            name='Pseudo Spectrum')
                )

    fig.add_trace(go.Scatter(x=music_doa,
                            y=ps_db[music_idx],
                            mode='markers',
                            name='Estimated DoA')
                )
    fig.update_layout(
        title='MUSIC',
        yaxis=dict(title='Amplitude (dB)'),
        xaxis=dict(title='Angle (deg)'),
        margin=dict(l=10, r=10, b=10, t=40),
    )

    # save plot
    fig.write_html(f'{EXPORT_PATH}music.html')

    return music_doa

def show_prompt(target, found):
    err = abs(target - found)
    err_percent = err / target * 100
    print(f"Found target: {found}")
    print("Err: " + str(err))
    print("Err %: " + str(err_percent))

def show_prompts(algorithm_promptname, targets, founds):
    print(f"{algorithm_promptname}\nTargets: {targets}")

    for i in range(len(targets)):
        show_prompt(targets[i], founds[i])

if __name__ == "__main__":
    # get radar sim data
    no_targets = 1
    target_angles = [5]
    radar, baseband, timestamp = get_data(no_targets=no_targets, target_angles=target_angles, load_stub="15")

    # # perform range doppler processing
    found_targets,  covmat = range_doppler(radar=radar, baseband=baseband)
    show_prompts("Range-Dopler FFT", target_angles, found_targets)

    music_found_targets = music(covmat, no_targets)
    show_prompts("MUSIC", target_angles, music_found_targets)

    rootmusic_found_targets = root_music(covmat, no_targets, 0.5)
    show_prompts("ROOT MUSIC", target_angles, rootmusic_found_targets)

    espirit_found_targets = espirit(covmat, no_targets, 0.5)
    show_prompts("ESPIRIT", target_angles, espirit_found_targets)
