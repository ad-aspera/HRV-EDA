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
from typing import Dict, Tuple, Optional, List, Any
from scipy.spatial.distance import cdist, pdist, squareform
from functools import lru_cache
from numpy.linalg import svd

def get_all_metrics(signal: pd.Series) -> Dict[str, float]:
    # Initialize all metric calculators at once to avoid redundant computations
    td_calc = TD_metrics(signal)
    fd_calc = FD_metrics(signal)
    nl_calc = NL_metrics(signal)
    
    # Get metrics from each calculator
    td_metrics = td_calc.get_all_metrics()
    fd_metrics = fd_calc.get_all_metrics()
    nl_metrics = nl_calc.get_all_metrics()
    
    # Combine all metrics
    combined_metrics = {**td_metrics, **fd_metrics, **nl_metrics}
    return combined_metrics

class TD_metrics:
    """class calculates time domain metrics for a pd.series type list of RR intervals"""
    def __init__(self, data: pd.Series):
        if not isinstance(data, pd.Series):  # establish correct type should be pd.series
            raise TypeError(f"Expected a pandas Series, but got {type(data).__name__}")
        self.data = data.dropna().values  # drop NaN values
        # Pre-compute common values
        self._diff_rr = np.diff(self.data)
        self._mean_rr = np.mean(self.data)

    def SDRR(self) -> float:
        """Standard deviation of RR intervals"""
        return np.std(self.data, ddof=1)

    def pNN50(self) -> float:
        """Percentage of successive RR intervals that differ by more than 50 ms"""
        # Use pre-computed diff_rr
        return np.sum(np.abs(self._diff_rr) > 50) / len(self._diff_rr) * 100

    def RMSSD(self) -> float:
        """Root mean square of successive RR interval differences"""
        # Use pre-computed diff_rr
        return np.sqrt(np.mean(self._diff_rr ** 2))

    def mean_hr(self) -> float:
        """mean HR in bpm"""
        # Use pre-computed mean_rr
        return 60000 / self._mean_rr  # Convert ms to bpm

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

        # Calculate PSD only once
        signal, self.freq_domain_data = SincPsd.sinc_and_psd(self.data, window='hann')
        # Limit domain to 1Hz
        self.freq_domain_data = self.freq_domain_data[self.freq_domain_data.index <= 1]
        # Normalize power
        self.freq_domain_data /= np.sum(self.freq_domain_data.values)
        
        # Define frequency bands for reuse
        self.freq_bands = {
            "ULF": (0, 0.003),
            "VLF": (0.003, 0.04),
            "LF": (0.04, 0.15),
            "HF": (0.15, 0.4)
        }
        
        # Pre-compute band powers and peaks
        self._band_results = {}
        for band_name, (low, high) in self.freq_bands.items():
            # Get band data once
            mask = (self.freq_domain_data.index >= low) & (self.freq_domain_data.index <= high)
            band_data = self.freq_domain_data[mask]
            
            # Store results for reuse
            if len(band_data) > 0:
                power = np.trapz(band_data.values, band_data.index)
                peak_idx = band_data.values.argmax() if len(band_data) > 0 else None
                peak_freq = band_data.index[peak_idx] if peak_idx is not None else np.nan
                peak_power = band_data.values.max() if len(band_data) > 0 else 0.0
            else:
                power, peak_freq, peak_power = 0.0, np.nan, 0.0
                
            self._band_results[band_name] = {
                "power": power,
                "peak_freq": peak_freq,
                "peak_power": peak_power
            }

    def _get_band_power(self, band_name: str) -> float:
        """Get pre-computed power for a specific frequency band"""
        return self._band_results[band_name]["power"]

    def _get_peak_frequency(self, band_name: str) -> float:
        """Get pre-computed peak frequency for a specific band"""
        return self._band_results[band_name]["peak_freq"]

    def _get_peak_power(self, band_name: str) -> float:
        """Get pre-computed peak power for a specific band"""
        return self._band_results[band_name]["peak_power"]

    def ULF_power(self) -> float:
        """Ultra low frequency power (≤0.003 Hz)"""
        return self._get_band_power("ULF")

    def ULF_peak(self) -> float:
        """Peak frequency in ULF band (≤0.003 Hz)"""
        return self._get_peak_frequency("ULF")

    def ULF_peak_power(self) -> float:
        """Peak power in ULF band (≤0.003 Hz)"""
        return self._get_peak_power("ULF")

    def VLF_power(self) -> float:
        """Very low frequency power (0.003-0.04 Hz)"""
        return self._get_band_power("VLF")

    def VLF_peak(self) -> float:
        """Peak frequency in VLF band (0.003-0.04 Hz)"""
        return self._get_peak_frequency("VLF")

    def VLF_peak_power(self) -> float:
        """Peak power in VLF band (0.003-0.04 Hz)"""
        return self._get_peak_power("VLF")

    def LF_power(self) -> float:
        """Low frequency power (0.04-0.15 Hz)"""
        return self._get_band_power("LF")

    def LF_peak(self) -> float:
        """Peak frequency in LF band (0.04-0.15 Hz)"""
        return self._get_peak_frequency("LF")

    def LF_peak_power(self) -> float:
        """Peak power in LF band (0.04-0.15 Hz)"""
        return self._get_peak_power("LF")

    def HF_power(self) -> float:
        """High frequency power (0.15-0.4 Hz)"""
        return self._get_band_power("HF")

    def HF_peak(self) -> float:
        """Peak frequency in HF band (0.15-0.4 Hz)"""
        return self._get_peak_frequency("HF")

    def HF_peak_power(self) -> float:
        """Peak power in HF band (0.15-0.4 Hz)"""
        return self._get_peak_power("HF")

    def LF_HF_ratio(self) -> float:
        """Ratio of LF to HF power"""
        lf = self._get_band_power("LF")
        hf = self._get_band_power("HF")
        return lf / hf if hf > 0 else np.nan

    def get_all_metrics(self) -> Dict[str, float]:
        """Dictionary of frequency domain metrics - get all metrics at once for efficiency"""
        metrics = {}
        
        # Add all band metrics in a loop
        for band_name in self.freq_bands:
            band_key = band_name
            metrics[f"{band_key} Power"] = self._get_band_power(band_name)
            metrics[f"{band_key} Peak Frequency"] = self._get_peak_frequency(band_name)
            metrics[f"{band_key} Peak Power"] = self._get_peak_power(band_name)
        
        # Add ratio
        metrics["LF/HF Ratio"] = self.LF_HF_ratio()
        
        return metrics

class NL_metrics:
    """Class calculates non-linear metrics for a pd.series type list of RR intervals"""
    def __init__(self, data: pd.Series):
        if not isinstance(data, pd.Series):
            raise TypeError(f"Expected a pandas Series, but got {type(data).__name__}")
        self.data = data.dropna().values
        # Pre-compute standard deviation for reuse
        self._std = np.std(self.data)
        # Cache Poincaré plot data
        self._poincare_data = None
        # Cache for distance matrices
        self._distance_cache = {}
        self._sd1_sd2 = None

    def _create_poincare_plot(self) -> tuple:
        """Creates Poincaré plot data points"""
        if self._poincare_data is None:
            x = self.data[:-1]  # RR(n) x should be the current RR interval
            y = self.data[1:]   # RR(n+1) y should be the next RR interval
            self._poincare_data = (x, y)
        return self._poincare_data

    def _calculate_sd1_sd2(self) -> Tuple[float, float]:
        """Calculate both SD1 and SD2 together"""
        if self._sd1_sd2 is None:
            x, y = self._create_poincare_plot()
            sd1 = np.sqrt(np.var(y - x) / 2)
            sd2 = np.sqrt(np.var(y + x) / 2)
            self._sd1_sd2 = (sd1, sd2)
        return self._sd1_sd2

    def SD1(self) -> float:
        """Poincaré plot standard deviation perpendicular to line of identity"""
        return self._calculate_sd1_sd2()[0]

    def SD2(self) -> float:
        """Poincaré plot standard deviation along line of identity"""
        return self._calculate_sd1_sd2()[1]

    def SD1_SD2_ratio(self) -> float:
        """Ratio of SD1 to SD2"""
        sd1, sd2 = self._calculate_sd1_sd2()
        return sd1 / sd2 if sd2 > 0 else np.nan

    def S(self) -> float:
        """Area of the ellipse representing total HRV"""
        sd1, sd2 = self._calculate_sd1_sd2()
        return np.pi * sd1 * sd2


    def S(self) -> float:
        """Area of the ellipse representing total HRV"""
        # Reuse already computed SD1 and SD2
        return np.pi * self.SD1() * self.SD2()

    def _get_distance_matrix(self, m: int) -> np.ndarray:
        """Compute and cache distance matrix for embeddings of dimension m"""
        if m not in self._distance_cache:
            N = len(self.data)
            if N <= m:
                return np.array([])
                
            # Create embedded vectors efficiently
            embedded = np.array([self.data[i:i+m] for i in range(N-m+1)])
            # Compute distance matrix using pdist for efficiency
            dist_matrix = squareform(pdist(embedded, 'chebyshev'))
            self._distance_cache[m] = dist_matrix
            
        return self._distance_cache[m]

    def ApEn(self, m: int = 2, r: float = 0.2) -> float:
        """Approximate entropy - optimized implementation
        m: embedding dimension
        r: tolerance (typically 0.2 * std of the data)"""
        N = len(self.data)
        if N <= m + 1:
            return np.nan
            
        r = r * self._std
        
        # Get distance matrices from cache
        dist_m = self._get_distance_matrix(m)
        dist_m1 = self._get_distance_matrix(m + 1)
        
        if len(dist_m) == 0 or len(dist_m1) == 0:
            return np.nan
        
        # Count matches using vectorized operations
        count_m = np.sum(dist_m <= r, axis=1)
        count_m1 = np.sum(dist_m1 <= r, axis=1)
        
        # Calculate phi values directly
        phi_m = np.mean(np.log(count_m / (N - m + 1.0)))
        phi_m1 = np.mean(np.log(count_m1 / (N - m)))
        
        return abs(phi_m - phi_m1)

    def SampEn(self, m: int = 2, r: float = 0.2) -> float:
        """Sample entropy - optimized implementation
        m: embedding dimension
        r: tolerance (typically 0.2 * std of the data)"""
        N = len(self.data)
        if N <= m + 1:
            return np.nan
            
        r = r * self._std
        
        # Get distance matrices from cache
        dist_m = self._get_distance_matrix(m)
        dist_m1 = self._get_distance_matrix(m + 1)
        
        if len(dist_m) == 0 or len(dist_m1) == 0:
            return np.nan
        
        # Remove self-matches by setting diagonal to infinity
        np.fill_diagonal(dist_m, np.inf)
        np.fill_diagonal(dist_m1, np.inf)
        
        # Count matches using vectorized operations
        A = np.sum(dist_m1 <= r)
        B = np.sum(dist_m <= r)
        
        # Calculate SampEn
        if B == 0:
            return np.nan
        return -np.log(A / B)

    def DFA(self, scale_min: int = 4, scale_max: int = None) -> tuple:
        """Detrended Fluctuation Analysis - optimized implementation
        Returns α1 (short-term) and α2 (long-term) scaling exponents"""
        # Prepare the data by integrating the time series
        x = np.cumsum(self.data - np.mean(self.data))
        N = len(x)
        
        if scale_max is None:
            scale_max = N // 4
        
        # Generate logarithmically spaced scales
        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), 20, dtype=int)
        # Ensure unique scales
        scales = np.unique(scales)
        fluct = np.zeros(len(scales))
        
        # Calculate fluctuation for each scale using vectorized operations where possible
        for i, scale in enumerate(scales):
            # Skip if scale is too large
            if N < scale:
                fluct[i] = np.nan
                continue
                
            # Number of segments
            segments = N // scale
            if segments == 0:
                fluct[i] = np.nan
                continue
            
            # Reshape data into segments
            y = np.reshape(x[:segments*scale], (segments, scale))
            
            # Create time array once
            t = np.arange(scale)
            
            # Calculate local trends and fluctuations for all segments at once
            v_squared = np.zeros(segments)
            for j in range(segments):
                p = np.polyfit(t, y[j], 1)
                v_squared[j] = np.mean((y[j] - np.polyval(p, t))**2)
                
            fluct[i] = np.sqrt(np.mean(v_squared))
        
        # Calculate slopes (α1 and α2)
        valid_idx = ~np.isnan(fluct)
        if np.sum(valid_idx) < 4:  # Need at least 4 points for reliable fits
            return np.nan, np.nan
            
        scales_log = np.log10(scales[valid_idx])
        fluct_log = np.log10(fluct[valid_idx])
        
        # Split into short-term and long-term
        idx_split = len(scales_log) // 2
        
        if idx_split > 0:
            # Calculate α1 (short-term)
            p1 = np.polyfit(scales_log[:idx_split], fluct_log[:idx_split], 1)
            alpha1 = p1[0]
        else:
            alpha1 = np.nan
            
        if len(scales_log) - idx_split > 0:
            # Calculate α2 (long-term)
            p2 = np.polyfit(scales_log[idx_split:], fluct_log[idx_split:], 1)
            alpha2 = p2[0]
        else:
            alpha2 = np.nan
        
        return alpha1, alpha2

    def D2(self, m: int = 10, r: float = 0.2) -> float:
        """Correlation Dimension (D2) - optimized implementation
        m: embedding dimension
        r: radius for neighborhood search"""
        N = len(self.data)
        if N <= m:
            return np.nan
            
        r = r * self._std
        
        # Get distance matrix from cache
        dist_matrix = self._get_distance_matrix(m)
        
        if len(dist_matrix) == 0:
            return np.nan
            
        # Remove self-matches
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Calculate correlation sum more efficiently
        C = np.sum(dist_matrix <= r) / (N * (N-1))
        
        if C <= 0:
            return np.nan
            
        return np.log(C) / np.log(r)

    def get_all_metrics(self) -> Dict[str, float]:
        """Dictionary of non-linear metrics - get all metrics at once for efficiency"""
        # Calculate SD1 and SD2 only once
        sd1 = self.SD1()
        sd2 = self.SD2()
        
        # Calculate DFA exponents
        alpha1, alpha2 = self.DFA()
        
        return {
            "SD1": sd1,
            "SD2": sd2,
            "SD1/SD2": sd1 / sd2 if sd2 > 0 else np.nan,
            "S": np.pi * sd1 * sd2,
            "ApEn": self.ApEn(),
            "SampEn": self.SampEn(),
            "DFA α1": alpha1,
            "DFA α2": alpha2,
            "D2": self.D2()
        }
