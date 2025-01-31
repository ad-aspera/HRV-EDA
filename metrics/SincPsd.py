import pandas as pd
import numpy as np
import seaborn as sns

def signal_to_PSD(signal:pd.Series, sampling_freq = 100):


    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(fft), 1/sampling_freq)

    PSD = pd.Series(abs(fft)**2, index=freq, name='Energy')
    PSD.index.name = 'Frequency'
    return PSD

def sinc_interpolate(signal:pd.Series):
    """Apply sinc interpolation to a signal.
    Uses sin(x)/x convolution to correctly """
    T = (signal.index.max()-signal.index.min())/len(signal)
    constant_grid = np.arange(len(signal)) * T
    interpolated = pd.Series(0, index=constant_grid)
    for point, magnitude in signal.items():
        interpolated += magnitude * np.sinc((point-constant_grid) / T)

    return interpolated

def sinc_and_psd(signal:pd.Series, window=None, window_fraction=1/16):
    """SINC Interpolates and gets the PSD of a signal.
    Returns signal and PSD
    Window options: 'hann', 'sin'; Fraction 1, windows the full data """
    signal = sinc_interpolate(signal)

    if window:
        signal = window_func(signal, window, window_fraction)

   

    return signal, signal_to_PSD(signal, 1/np.mean(np.diff(signal.index)))


def window_func(signal:pd.Series, window_type='hann', window_fraction=1/16):
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

    #print(f"{window_type} window applied over:")
    #print(f"left: index 0 to {len(subwindow1)-1}")
    #print(f"right: index {len(signal)-len(subwindow2)} to {len(signal)}")
    
    return signal

if __name__ == '__main__':
    import metrics.BasicGenerator as BasicGenerator
    frequencies = [0.15, 0.35, 0.45]
    magnitudes = [1, 0.5, 0.3]

    signal = BasicGenerator.generate_combined_sines(frequencies, magnitudes)

    sampled_HRV_signal = BasicGenerator.generate_sin_HRV(signal)

    FFT_input_signal, PSD = sinc_and_psd(signal, window = 'hann', window_fraction = 1/16)

    print(PSD)