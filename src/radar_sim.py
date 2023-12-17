import numpy as np
from scipy import signal, fft, linalg
from radarsimpy import Radar, Transmitter, Receiver
import plotly.graph_objs as go
from radarsimpy.simulator import simc
from radar_proc import range_doppler_fft
from utils import EXPORT_PATH
from typing import Dict, List

def create_radar(show_plot: bool = False):
    """
    Create a radar object.

    Args:
        show_plot (bool, optional): Plot the transmitter and receiver array. Defaults to False.

    Returns:
        Radar: radar object
    """
    # configure a MIMO array with 2 transmitter channels and 64 receiver channels
    # standard wavelength c / f
    wavelength = 3e8 / 60.5e9

    # receiver
    N_rx = 64

    tx_channels = []
    tx_channels.append(
        dict(
            location=(0, -N_rx/2*wavelength/2, 0),
        ))

    tx_channels.append(
        dict(
            location=(0, wavelength*N_rx/2-N_rx/2*wavelength/2, 0),
        ))

    rx_channels = []
    for idx in range(0, N_rx):
        rx_channels.append(
            dict(
                location=(0, wavelength/2*idx-(N_rx-1) * wavelength/4, 0),
            ))

    tx = Transmitter(f=[61e9, 60e9],
                    t=[0, 16e-6],
                    tx_power=15,
                    prp=40e-6,
                    pulses=512,
                    channels=tx_channels)

    rx = Receiver(fs=20e6,
                noise_figure=8,
                rf_gain=20,
                load_resistor=500,
                baseband_gain=30,
                channels=rx_channels)

    # finally, create radar object
    radar = Radar(transmitter=tx, receiver=rx)

    # plot transmitters and receivers
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=radar.radar_prop["transmitter"].txchannel_prop["locations"][:, 1]/wavelength,
                y=radar.radar_prop["transmitter"].txchannel_prop["locations"][:, 2]/wavelength,
                mode='markers',
                name='Transmitter',
                opacity=0.7,
                marker=dict(size=10)))

    fig.add_trace(
        go.Scatter(x=radar.radar_prop["receiver"].rxchannel_prop["locations"][:, 1]/wavelength,
                y=radar.radar_prop["receiver"].rxchannel_prop["locations"][:, 2]/wavelength,
                mode='markers',
                opacity=1,
                name='Receiver'))

    fig.update_layout(
        title='Array configuration',
        xaxis=dict(title='y (λ)'),
        yaxis=dict(title='z (λ)',
                scaleanchor="x",
                scaleratio=1),
    )

    # uncomment this to display interactive plot
    if show_plot:
        fig.show()

    return radar

def create_targets(num_targets: int, target_angles: List[int]):
    """
    Create targets to simulate for the radar.

    Args:
        num_targets (int): number of targets
        target_angles (list[int]): azimuth angles of the targtes

    Returns:
        List[int]: created targets list
    """
    targets = []

    for i in range(num_targets):
        target = dict(location=(40*np.cos(np.radians(target_angles[i])), 40*np.sin(np.radians(target_angles[i])),
                    0), speed=(0, 0, 0), rcs=10, phase=0)
        targets.append(target)

    return targets

def simulate(radar: Radar, targets: List[int]):
    """
    Simulate signals received from targets.
    This operation can take some time so it's better to cache this data, and reuse it as much as you can.

    Args:
        radar (Radar): radar object 
        targets (List[int]): targets 

    Returns:
        {numpy.3darray,numpy.3darray}: simulated data 
    """
    return simc(radar, targets,  noise=True)

def save_data(data: Dict[np.ndarray, np.ndarray], stub_name: str):
    """
    Save data to numpy file.

    Args:
        data (Dict[np.ndarray, np.ndarray]): data to save
        stub_name (str): name to add to recognize your data 
    """
    np.save(f"{EXPORT_PATH}baseband_{stub_name}.npy", data["baseband"])
    np.save(f"{EXPORT_PATH}timestamp_{stub_name}.npy", data["timestamp"])

def load_data(stub_name: str):
    """
    Load data from numpy file.

    Args:
        stub_name (str): name to recognize your data  

    Returns:
        numpy.ndarray, numpy.ndarray: data (baseband, timestamp)
    """
    baseband = np.load(f"{EXPORT_PATH}baseband_{stub_name}.npy")
    timestamp = np.load(f"{EXPORT_PATH}timestamp_{stub_name}.npy")

    return baseband, timestamp

if __name__ == "__main__":
    radar = create_radar(False)
    # targets = create_targets(3, [-5, -4, 45])
    targets = create_targets(1, [5])
    # data = simulate(radar=radar, targets=targets)

    # # cache data for further use
    # np.save("baseband_15.npy", data["baseband"])
    # np.save("timestamp_15.npy", data["timestamp"])

    baseband, timestamp = load_data("15")
