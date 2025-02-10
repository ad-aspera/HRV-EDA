import pandas as pd
from typing import Union
import metrics.HRV_Metrics as HRV_Metrics
Numeric = Union[float, int]
import xlwings as xw


def signal_as_series_enforcer(func):
    def wrapper(signal: pd.Series | list[Numeric] | tuple[Numeric], *args, **kwargs):
        """Forces signal into series with RR_interval values in milliseconds and index as seconds"""
        signal = pd.Series(signal).astype(float)
        signal.index = signal.cumsum()

        # Make sure that indexing is in seconds rather than milliseconds
        if signal.mean() >= 10:
            signal.index = signal.index / 1000

        # Make sure that RR_intervals are in milliseconds
        if signal.mean() < 10:
            signal = signal * 1000

        return func(signal, *args, **kwargs)
    return wrapper

@signal_as_series_enforcer
def time_portion_signal(signal: pd.Series, fragment_s: float = 300):
    """Divides signal into 5 minutes internals.
    Signal index has to be in seconds"""
    five_min = []

    
    start_idx = 0
    while start_idx < len(signal):
        end_idx = start_idx
        while end_idx < len(signal) and (signal.index[end_idx] - signal.index[start_idx]) < fragment_s:
            end_idx += 1

        if end_idx == len(signal):
            break

        segment = signal.iloc[start_idx:end_idx]
        segment.index = segment.index - segment.index[0]

        if segment.index[-1] < fragment_s / 2:
            break

        five_min.append(segment)
        start_idx = end_idx

    if len(five_min) == 0:
        raise ValueError(f"The signal is too short to fraction into fragments of {fragment_s}s")

    return five_min


if __name__ == "__main__":
    import pickle

    # Load the pickled file
    with open('actionable_data/data.pkl', 'rb') as f:
        peaks = pickle.load(f)

    # Select 3 patients
    selected_patients = list(peaks.keys())[:3]
    DS_RR = {}
    for patient_id in selected_patients:
        patient_ds = peaks[patient_id]['DS'][0]
        patient_ds = time_portion_signal(patient_ds)

        print(patient_ds)


def patients_metrics(signal:pd.Series, sub_signal_duration_s=300)->pd.Series:
    metrics = pd.DataFrame()
    for i, subsignal in enumerate(time_portion_signal(signal, sub_signal_duration_s)):

        metrics_dict = HRV_Metrics.get_td_and_fd_metrics(subsignal)

        metrics_dict = {'t_start': i*300, 't_end': (i+1)*300, **metrics_dict}

        metrics = pd.concat([metrics, pd.DataFrame([metrics_dict])], ignore_index=True)

    return metrics



def df_from_excel(path):
    wb = xw.Book(path)
    wb.app.api.Visible = False
    wb.app.calculate()
    wb.save()
    #wb.close()
    wb.app.quit()
    return pd.read_excel(path)
        


