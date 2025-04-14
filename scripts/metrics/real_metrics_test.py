import metrics.HRV_Metrics as HRV_Metrics
import pandas as pd
import pickle

def load_trial_data(filepath):
    with open(filepath, 'rb') as file:
        trial = pickle.load(file)
    return trial

def process_interval(signal):

    measure_TD = HRV_Metrics.TD_metrics(signal)
    measure_FD = HRV_Metrics.FD_metrics(signal)

    td_metrics = measure_TD.get_all_metrics()
    fd_metrics = measure_FD.get_all_metrics()

    combined_metrics = {**td_metrics, **fd_metrics}
    return combined_metrics

def main(filepath):
    signal = load_trial_data(filepath)
    print(signal)
    metrics = pd.DataFrame()
    
    for lower in range(0, int(signal.index.max()), 300):
        five_min = signal[(signal.index > lower) & (signal.index < lower+300)]
        print("f", five_min)
        combined_metrics = process_interval(five_min)
        if combined_metrics:
            metrics = pd.concat([metrics, pd.DataFrame([combined_metrics])], ignore_index=True)
    return metrics

if __name__ == "__main__":
    filepath = 'metrics/organic_HRV_sample.pkl'  # Replace with the actual file path
    metrics = main(filepath)
    print(metrics)