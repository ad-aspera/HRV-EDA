"""
Created on Wed Jan 29 15:35:35 2025

@author: henryhollingworth

Based on the review:

Shaffer, F. and Ginsberg, J.P., 2017. An overview of heart rate variability metrics and norms. Frontiers in Public Health, 5, p.258. Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC5624990/ [Accessed 28 Jan. 2025].

Priority for time domain measures: RMSSD, SDNN, pNN50
Priority for Frequency domain measures: LF Bands, HF Bands, LF/HF
"""
#Dependencies
import numpy as np
import pandas as pd
import metrics.SincPsd as SincPsd

class TD_metrics:
    """class calculates time domain metrics for a pd.series type list of RR intervals"""
    def __init__(self, data: pd.Series):
        if not isinstance(data, pd.Series):#establish correct type should be pd.series
            raise TypeError(f"Expected a pandas Series, but got {type(HRV_data).__name__}")
        self.data = data.dropna().values #drop Nan values


    def SDRR(self):
        """Standard deviation of RR intervals"""
        return np.std(self.data, ddof=1)
    def pNN50(self):
        """Percentage of successive RR intervals that differ by more than 50 ms"""
        diff_rr = np.abs(np.diff(self.data))
        return np.sum(diff_rr > 50) / len(diff_rr) * 100
    def RMSSD(self):
        """Root mean square of successive RR interval differences"""
        diff_rr = np.diff(self.data)
        return np.sqrt(np.mean(diff_rr ** 2))
    def mean_hr(self):
        """mean HR in bpm"""
        mean_rr = np.mean(self.data)
        return 60000 / mean_rr  # Convert ms to bpm
    def get_all_metrics(self):
        """Dictionary of time domain metrics (all vals as should be np.float64)"""
        return {
            "SDRR": self.SDRR(),
            "RMSSD": self.RMSSD(),
            "pNN50 (%)": self.pNN50(),
            "Mean HR (bpm)": self.mean_hr()
        }
    
class FD_metrics:
    """class calculates frequency domain metrics for a pd.series type list of RR intervals"""
    def __init__(self, data: pd.Series, sampling_frequency=100):
        if not isinstance(data, pd.Series):
            raise TypeError(f"Expected a pandas Series, but got {type(HRV_data).__name__}")
        self.data = data.dropna().values
        self.freq_domain_data = SincPsd.signal_to_PSD(self.data, sampling_frequency)
    
    def _get_band_power(self, low_freq, high_freq):
        """Helper method to calculate power in a specific frequency band"""
        mask = (self.freq_domain_data.index >= low_freq) & (self.freq_domain_data.index <= high_freq)
        band_data = self.freq_domain_data[mask]
        if len(band_data) == 0:
            return 0.0
        return np.trapz(band_data.values, band_data.index)
    
    def _get_peak_frequency(self, low_freq, high_freq):
        """Helper method to find peak frequency in a specific band"""
        mask = (self.freq_domain_data.index >= low_freq) & (self.freq_domain_data.index <= high_freq)
        band_data = self.freq_domain_data[mask]
        if len(band_data) == 0:
            return np.nan
        peak_idx = band_data.values.argmax()
        return band_data.index[peak_idx]
    
    def ULF_power(self):
        """Ultra low frequency power (≤0.003 Hz)"""
        return self._get_band_power(0, 0.003)
    
    def ULF_peak(self):
        """Peak frequency in ULF band (≤0.003 Hz)"""
        return self._get_peak_frequency(0, 0.003)
    
    def VLF_power(self):
        """Very low frequency power (0.003-0.04 Hz)"""
        return self._get_band_power(0.003, 0.04)
    
    def VLF_peak(self):
        """Peak frequency in VLF band (0.003-0.04 Hz)"""
        return self._get_peak_frequency(0.003, 0.04)
    
    def LF_power(self):
        """Low frequency power (0.04-0.15 Hz)"""
        return self._get_band_power(0.04, 0.15)
    
    def LF_peak(self):
        """Peak frequency in LF band (0.04-0.15 Hz)"""
        return self._get_peak_frequency(0.04, 0.15)
    
    def HF_power(self):
        """High frequency power (0.15-0.4 Hz)"""
        return self._get_band_power(0.15, 0.4)
    
    def HF_peak(self):
        """Peak frequency in HF band (0.15-0.4 Hz)"""
        return self._get_peak_frequency(0.15, 0.4)
    
    def LF_HF_ratio(self):
        """Ratio of LF to HF power"""
        return self.LF_power() / self.HF_power()
    
    def get_all_metrics(self):
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
    

