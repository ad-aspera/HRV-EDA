import pandas as pd
import numpy as np
import seaborn as sns
from scipy.signal import welch

def _old_signal_to_PSD(signal: pd.Series, sampling_freq=100, min_5_limit = True, window='hann', window_fraction=1/16):
    if window:
        signal = window_func(signal, window, window_fraction)

    if signal.mean()>100:
        signal = signal/1000
    signal = signal-signal.mean()

    fft = np.fft.fft(signal)

    freq = np.fft.fftfreq(len(fft), 1/sampling_freq)  # noqa: E501
    PSD = pd.Series(abs(fft)**2, index=freq, name='Energy')
    PSD.index.name = 'Frequency'
   # if min_5_limit:
        #PSD = PSD[PSD.index >= 1/300]
        #None
    return PSD


def signal_to_PSD(signal: pd.Series, sampling_freq=250, n_per_seg=4*1024,**args):
   # if n_per_seg> len(signal):
    #    n_per_seg = int(len(signal)/4)
    # https://dsp.stackexchange.com/questions/81640/trying-to-understand-the-nperseg-effect-of-welch-method
    n_per_seg = min(n_per_seg, len(signal))
    f, S = welch(signal, fs=sampling_freq, nperseg=n_per_seg)
    return pd.Series(S, index=f, name='Energy')

def _signal_shorten(signal:pd.Series):
  
    length = len(signal)
    power_of_two_length = 2**int(np.log2(length))
    return signal.iloc[:power_of_two_length]


def sinc_interpolate(signal: pd.Series):
    """Apply sinc interpolation to a signal.
    Uses sin(x)/x convolution to correctly interpolate."""
    T = (signal.index.max() - signal.index.min()) / len(signal)
    constant_grid = np.arange(len(signal)) * T
    interpolated = pd.Series(0, index=constant_grid)
    for point, magnitude in signal.items():
        interpolated += magnitude * np.sinc((point - constant_grid) / T)
    return interpolated

def sinc_and_psd(signal: pd.Series, window=None, window_fraction=1/16):
    """SINC Interpolates and gets the PSD of a signal.
    Returns signal and PSD
    Window options: 'hann', 'sin'; Fraction 1, windows the full data """
    signal = sinc_interpolate(signal)

    return signal, signal_to_PSD(signal, 1 / np.mean(np.diff(signal.index)), window_fraction=window_fraction)

def window_func(signal: pd.Series, window_type='hann', window_fraction=1/16):
    signal = signal.copy()
    window_length = int(len(signal) * window_fraction)
    if window_type == 'hann':
        subwindow1 = np.hanning(window_length)[:window_length // 2]
        subwindow2 = np.hanning(window_length)[window_length // 2:]
    elif window_type == 'sin':
        subwindow1 = np.sin(np.linspace(0, np.pi / 2, window_length // 2))
        subwindow2 = np.sin(np.linspace(np.pi / 2, np.pi, window_length // 2))
    else:
        raise ValueError("Unsupported window type")
    signal.iloc[:len(subwindow1)] *= subwindow1
    signal.iloc[-len(subwindow2):] *= subwindow2
    return signal

if __name__ == '__main__':
    import metrics.BasicGenerator as BasicGenerator
    frequencies = [0.15, 0.35, 0.45]
    magnitudes = [1, 0.5, 0.3]
    signal = BasicGenerator.generate_combined_sines(frequencies, magnitudes)
    sampled_HRV_signal = BasicGenerator.generate_sin_HRV(signal)
    FFT_input_signal, PSD = sinc_and_psd(signal, window='hann', window_fraction=1/16)
    print(PSD)