#Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BasicGenerator
import HRV_Metrics

"""example of how to use HRV_metrics module"""
frequencies = [0.15, 0.35, 0.45]
magnitudes = [1, 0.5, 0.3]

signal = BasicGenerator.generate_combined_sines(frequencies, magnitudes)

sampled_HRV_signal = BasicGenerator.generate_sin_HRV(signal)

measure_TD = HRV_Metrics.TD_metrics(sampled_HRV_signal)
measure_FD = HRV_Metrics.FD_metrics(sampled_HRV_signal)

print(measure_TD.get_all_metrics())
print(measure_FD.get_all_metrics())
