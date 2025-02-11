"""
Created on Wed Jan 29 15:35:35 2025

@author: henryhollingworth

Edited by Povilas Saucivuienas on 

Based on the review: 2025/01/31, and 2025-02-05

Shaffer, F. and Ginsberg, J.P., 2017. An overview of heart rate variability metrics and norms. Frontiers in Public Health, 5, p.258. Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC5624990/ [Accessed 28 Jan. 2025].

Priority for time domain measures: RMSSD, SDNN, pNN50
Priority for Frequency domain measures: LF Bands, HF Bands, LF/HF
Non-linear measures: S, SD1, SD2, SD1/SD2, ApEn, SampEn, DFA α1, DFA α2, D2
"""
# Dependencies
import numpy as np
import pandas as pd
import metrics.SincPsd as SincPsd
from typing import Dict
from scipy.spatial.distance import cdist
from numpy.linalg import svd

def get_all_metrics(signal: pd.Series) -> Dict[str, float]:
    td_metrics = TD_metrics(signal).get_all_metrics()
    fd_metrics = FD_metrics(signal).get_all_metrics()
    nl_metrics = NL_metrics(signal).get_all_metrics()
    combined_metrics = {**td_metrics, **fd_metrics, **nl_metrics}
    return combined_metrics

class TD_metrics:
    """class calculates time domain metrics for a pd.series type list of RR intervals"""
    def __init__(self, data: pd.Series):
        if not isinstance(data, pd.Series):  # establish correct type should be pd.series
            raise TypeError(f"Expected a pandas Series, but got {type(data).__name__}")
        self.data = data.dropna().values  # drop NaN values

    def SDRR(self) -> float:
        """Standard deviation of RR intervals"""
        return np.std(self.data, ddof=1)

    def pNN50(self) -> float:
        """Percentage of successive RR intervals that differ by more than 50 ms"""
        diff_rr = np.abs(np.diff(self.data))
        return np.sum(diff_rr > 50) / len(diff_rr) * 100

    def RMSSD(self) -> float:
        """Root mean square of successive RR interval differences"""
        diff_rr = np.diff(self.data)
        return np.sqrt(np.mean(diff_rr ** 2))

    def mean_hr(self) -> float:
        """mean HR in bpm"""
        mean_rr = np.mean(self.data)
        return 60000 / mean_rr  # Convert ms to bpm

    def get_all_metrics(self) -> Dict[str, float]:
        """Dictionary of time domain metrics (all vals as should be np.float64)"""
        return {
            "SDRR": self.SDRR(),
            "RMSSD": self.RMSSD(),
            "pNN50 (%)": self.pNN50(),
            "Mean HR (bpm)": self.mean_hr()
        }

class FD_metrics:
    """class calculates frequency domain metrics for a pd.series type list of RR intervals"""
    def __init__(self, data: pd.Series, sampling_frequency: int = 250):
        if not isinstance(data, pd.Series):
            raise TypeError(f"Expected a pandas Series, but got {type(data).__name__}")
        self.data = data

        signal, self.freq_domain_data = SincPsd.sinc_and_psd(self.data, window='hann')

    def _get_band_power(self, low_freq: float, high_freq: float) -> float:
        """Helper method to calculate power in a specific frequency band"""
        mask = (self.freq_domain_data.index >= low_freq) & (self.freq_domain_data.index <= high_freq)
        band_data = self.freq_domain_data[mask]
        if len(band_data) == 0:
            return 0.0
        return np.trapz(band_data.values, band_data.index)

    def _get_peak_frequency(self, low_freq: float, high_freq: float) -> float:
        """Helper method to find peak frequency in a specific band"""
        mask = (self.freq_domain_data.index >= low_freq) & (self.freq_domain_data.index <= high_freq)
        band_data = self.freq_domain_data[mask]
        if len(band_data) == 0:
            return np.nan
        peak_idx = band_data.values.argmax()
        return band_data.index[peak_idx]

    def ULF_power(self) -> float:
        """Ultra low frequency power (≤0.003 Hz)"""
        return self._get_band_power(0, 0.003)

    def ULF_peak(self) -> float:
        """Peak frequency in ULF band (≤0.003 Hz)"""
        return self._get_peak_frequency(0, 0.003)

    def VLF_power(self) -> float:
        """Very low frequency power (0.003-0.04 Hz)"""
        return self._get_band_power(0.003, 0.04)

    def VLF_peak(self) -> float:
        """Peak frequency in VLF band (0.003-0.04 Hz)"""
        return self._get_peak_frequency(0.003, 0.04)

    def LF_power(self) -> float:
        """Low frequency power (0.04-0.15 Hz)"""
        return self._get_band_power(0.04, 0.15)

    def LF_peak(self) -> float:
        """Peak frequency in LF band (0.04-0.15 Hz)"""
        return self._get_peak_frequency(0.04, 0.15)

    def HF_power(self) -> float:
        """High frequency power (0.15-0.4 Hz)"""
        return self._get_band_power(0.15, 0.4)

    def HF_peak(self) -> float:
        """Peak frequency in HF band (0.15-0.4 Hz)"""
        return self._get_peak_frequency(0.15, 0.4)

    def LF_HF_ratio(self) -> float:
        """Ratio of LF to HF power"""
        try:
            return self.LF_power() / self.HF_power()
        except ZeroDivisionError:
            return np.NaN

    def get_all_metrics(self) -> Dict[str, float]:
        """Dictionary of frequency domain metrics"""
        return {
            "ULF Power": self.ULF_power(),
            "ULF Peak Frequency": self.ULF_peak(),
            "VLF Power": self.VLF_power(),
            "VLF Peak Frequency": self.VLF_peak(),
            "LF Power": self.LF_power(),
            "LF Peak Frequency": self.LF_peak(),
            "HF Power": self.HF_power(),
            "HF Peak Frequency": self.HF_peak(),
            "LF/HF Ratio": self.LF_HF_ratio()
        }

class NL_metrics:
    """Class calculates non-linear metrics for a pd.series type list of RR intervals"""
    def __init__(self, data: pd.Series):
        if not isinstance(data, pd.Series):
            raise TypeError(f"Expected a pandas Series, but got {type(data).__name__}")
        self.data = data.dropna().values

    def _create_poincare_plot(self) -> tuple:
        """Creates Poincaré plot data points"""
        x = self.data[:-1]  # RR(n) x should be the current RR interval
        y = self.data[1:]   # RR(n+1) y should be the next RR interval
        return x, y

    def SD1(self) -> float:
        """Poincaré plot standard deviation perpendicular to line of identity"""
        x, y = self._create_poincare_plot() #take the existing poincare plot data
        sd1 = np.sqrt(np.var(y - x) / 2) #work out the standard deviation of the data perpendicular to the line of identity
        return sd1

    def SD2(self) -> float:
        """Poincaré plot standard deviation along line of identity"""
        x, y = self._create_poincare_plot()
        sd2 = np.sqrt(np.var(y + x) / 2)#work out the standard deviation of the data along the line of identity
        return sd2

    def SD1_SD2_ratio(self) -> float:
        """Ratio of SD1 to SD2"""
        try:
            return self.SD1() / self.SD2() #works out the ratio
        except ZeroDivisionError:
            return np.nan #covering the case where SD2 is 0 - shouldnt really happen

    def S(self) -> float:
        """Area of the ellipse representing total HRV"""
        return np.pi * self.SD1() * self.SD2() #calculates elipse based on SD1 and SD2 as radii

    def ApEn(self, m: int = 2, r: float = 0.2) -> float: #i cannot lie i couldnt work out how to do this so this function is a chatgpt special
        """Approximate entropy
        m: embedding dimension
        r: tolerance (typically 0.2 * std of the data)"""
        N = len(self.data)
        if N == 0:
            return np.nan
            
        r = r * np.std(self.data)
        
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            x = [[self.data[j] for j in range(i, i + m)] for i in range(N - m + 1)]
            C = [len([1 for j in range(len(x)) if _maxdist(x[i], x[j]) <= r]) / (N - m + 1.0) 
                 for i in range(len(x))]
            return (N - m + 1.0)**(-1) * sum(np.log(C))
        
        return abs(_phi(m) - _phi(m + 1))

    def SampEn(self, m: int = 2, r: float = 0.2) -> float:
        """Sample entropy
        m: embedding dimension
        r: tolerance (typically 0.2 * std of the data)"""
        N = len(self.data)
        if N == 0:
            return np.nan
            
        r = r * np.std(self.data)
        
        def _count_matches(m):
            template = np.array([self.data[i:i+m] for i in range(N-m+1)])
            dist = cdist(template, template, 'chebyshev')
            return np.sum(dist <= r) - (N-m+1)  # Exclude self-matches
            
        A = _count_matches(m+1)
        B = _count_matches(m)
        
        try:
            return -np.log(A/B)
        except (ValueError, ZeroDivisionError):
            return np.nan

    def DFA(self, scale_min: int = 4, scale_max: int = None) -> tuple:
        """Detrended Fluctuation Analysis
        Returns α1 (short-term) and α2 (long-term) scaling exponents"""
        # Prepare the data by integrating the time series
        x = np.cumsum(self.data - np.mean(self.data))
        
        if scale_max is None:
            scale_max = len(x) // 4
        
        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), 20, dtype=int)
        fluct = np.zeros(len(scales))
        
        # Calculate fluctuation for each scale
        for i, scale in enumerate(scales):
            # Calculate local trends
            segments = len(x) // scale
            if segments == 0:
                continue
                
            y = np.reshape(x[:segments*scale], (segments, scale))
            t = np.arange(scale)
            v = np.zeros(len(y))
            
            for j in range(segments):
                p = np.polyfit(t, y[j], 1)
                v[j] = np.sqrt(np.mean((y[j] - np.polyval(p, t))**2))
            
            fluct[i] = np.sqrt(np.mean(v**2))
        
        # Calculate slopes (α1 and α2)
        scales_log = np.log10(scales)
        fluct_log = np.log10(fluct)
        
        # Split into short-term and long-term
        idx_split = len(scales) // 2
        
        # Calculate α1 (short-term)
        p1 = np.polyfit(scales_log[:idx_split], fluct_log[:idx_split], 1)
        alpha1 = p1[0]
        
        # Calculate α2 (long-term)
        p2 = np.polyfit(scales_log[idx_split:], fluct_log[idx_split:], 1)
        alpha2 = p2[0]
        
        return alpha1, alpha2

    def D2(self, m: int = 10, r: float = 0.2) -> float:
        """Correlation Dimension (D2)
        m: embedding dimension
        r: radius for neighborhood search"""
        N = len(self.data)
        if N == 0:
            return np.nan
            
        r = r * np.std(self.data)
        
        # Create embedded vectors
        Y = np.array([self.data[i:i+m] for i in range(N-m+1)])
        
        # Calculate distances between all pairs
        D = cdist(Y, Y, 'euclidean')
        
        # Calculate correlation sum
        C = np.sum(D <= r) / (N * (N-1))
        
        try:
            return np.log(C) / np.log(r)
        except (ValueError, ZeroDivisionError):
            return np.nan

    def get_all_metrics(self) -> Dict[str, float]:
        """Dictionary of non-linear metrics"""
        alpha1, alpha2 = self.DFA()
        return {
            "S": self.S(),
            "SD1": self.SD1(),
            "SD2": self.SD2(),
            "SD1/SD2": self.SD1_SD2_ratio(),
            "ApEn": self.ApEn(),
            "SampEn": self.SampEn(),
            "DFA α1": alpha1,
            "DFA α2": alpha2,
            "D2": self.D2()
        }
    