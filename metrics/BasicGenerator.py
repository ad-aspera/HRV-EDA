import numpy as np
import pandas as pd


class BasicGenerator:
    """Allows to generate basic (sin derived signal)"""


def create_single_frequency(
    freq: float, magnitude: float = 1, sampling_freq=100, duration=300
) -> pd.Series:
    "Generates 5 minutes of signal"
    t = np.arange(0, duration, 1 / sampling_freq)
    signal = pd.Series(
        magnitude * np.sin(2 * np.pi * freq * t), index=pd.Index(t, name="Time")
    )
    signal = signal - signal.mean()
    return signal


def generate_combined_sines(
    frequencies: list[float], magnitudes:list[float]=None, sampling_freq=100, duration=300
) -> pd.Series:
    """Combines multiple sine signals into a single signal"""
    signal = 0
    if magnitudes is None:
        magnitudes = [1] * len(frequencies)

    for freq, mag in zip(frequencies, magnitudes):
        signal += create_single_frequency(freq, mag, sampling_freq, duration)
    signal = signal / signal.max()

    return signal


def generate_sin_HRV(
    signal,
    inactive_refractive_freq=3,
    active_refractive_freq=3,
    threshold1=-0.2,
    threshold2=-0.7,
):
    """Loosely generates a signal from sines"""
    sampled_signal = pd.Series(dtype=float)
    last_sample_time = (
        -10
    )  # Initialize to a value that ensures the first sample can be taken

    for time, value in signal.items():
        if value > threshold1 and (
            (time - last_sample_time) >= 1 / inactive_refractive_freq
        ):
            sampled_signal.at[time] = value
            last_sample_time = time
        elif value > threshold2 and (
            (time - last_sample_time)
            >= 1 / inactive_refractive_freq + 1 / active_refractive_freq
        ):
            sampled_signal.at[time] = value
            last_sample_time = time

    return sampled_signal
