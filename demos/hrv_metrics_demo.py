import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from project_utils.rr_to_metrics import time_portion_signal, patients_metrics

# Generate a synthetic RR interval time series
def generate_synthetic_rr(duration_seconds=600, mean_hr_bpm=60, hr_variation=0.1):
    # Mean RR interval in milliseconds
    mean_rr = 60000 / mean_hr_bpm
    
    # Generate RR intervals with variation
    n_beats = int(duration_seconds / (mean_rr / 1000))
    rr_intervals = np.random.normal(mean_rr, mean_rr * hr_variation, n_beats)
    
    # Ensure all RR intervals are positive
    rr_intervals = np.maximum(rr_intervals, 300)
    
    return pd.Series(rr_intervals)

if __name__ == "__main__":
    # Generate a synthetic RR interval series for 10 minutes
    rr_series = generate_synthetic_rr(duration_seconds=600, mean_hr_bpm=70, hr_variation=0.15)
    
    print("RR intervals series:")
    print(rr_series.head())
    print(f"Total duration: {rr_series.sum()/1000:.2f} seconds")
    
    # Segment the RR intervals into 5-minute (300s) segments
    segments, segments_info = time_portion_signal(rr_series, fragment_s=300)
    
    print("\nSegmentation results:")
    for i, (segment, (t_start, t_end)) in enumerate(zip(segments, segments_info)):
        print(f"Segment {i+1}:")
        print(f"  Start time: {t_start:.2f}s")
        print(f"  End time: {t_end:.2f}s")
        print(f"  Duration: {t_end-t_start:.2f}s")
        print(f"  Number of RR intervals: {len(segment)}")
    
    # Calculate HRV metrics
    metrics_df = patients_metrics(rr_series)
    
    print("\nHRV Metrics:")
    print(metrics_df)
    
    # Plot the results
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Original RR intervals
    rr_cumsum = np.cumsum(rr_series) / 1000  # Convert to seconds
    axes[0].plot(rr_cumsum, rr_series, 'b.-')
    axes[0].set_title('RR Intervals')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('RR Interval (ms)')
    
    # Plot 2: Segments visualization
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, ((t_start, t_end), color) in enumerate(zip(segments_info, colors)):
        axes[1].axvspan(t_start, t_end, alpha=0.3, color=color, label=f'Segment {i+1}')
    
    axes[1].plot(rr_cumsum, rr_series, 'k.-')
    axes[1].set_title('RR Intervals with Segments')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('RR Interval (ms)')
    axes[1].legend()
    
    # Plot 3: Time domain metrics by segment
    metrics_to_plot = ['RMSSD', 'SDRR', 'Mean HR (bpm)']
    for metric in metrics_to_plot:
        axes[2].plot(metrics_df['t_start'], metrics_df[metric], '.-', label=metric)
    
    axes[2].set_title('HRV Metrics by Segment')
    axes[2].set_xlabel('Segment Start Time (s)')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('hrv_metrics_demo.png')
    plt.show()
