from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
if __name__ == "__main__":
    import SincPsd
else:
    import metrics.SincPsd as SincPsd


class SignalPsdPlotter():
    """This is a plotter used to simplify plotting and exploring a paired signal and its psd"""
    def __init__(self, title:str = None, alpha =0.7, upper_labels = ['Time (s)', 'HRV (ms)'], lower_labels = ['Frequency (Hz)', 'Power (AU)', ]):
        self.fig = make_subplots(rows=2, cols=1, subplot_titles=('<b>Signal</b>', '<b>PSD</b>'))
        self.fig.update_layout(height=600, margin=dict(l=20, r=20, t=50, b=20), title_text="Signal and PSD")
        self.alpha = alpha
        self.upper_labels = upper_labels
        self.lower_labels = lower_labels



        d_vert = 0.06
        self.fig.update_layout(yaxis2=dict(domain=[0, 0.5-d_vert]), yaxis=dict(domain=[0.5+d_vert, 1]))
        
        self.fig.update_annotations(font=dict(size=14))  # Ensure subplot titles don't move
        self.fig['layout']['annotations'][1]['y']+=0.06

        title = title or "HRV Signal and PSD plot"
        self.fig.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'} ,
            title_font=dict(family='Arial Black', size=16))

    def plot_signal_and_psd(self, signal: pd.Series, psd: pd.Series, label: str, color: str):
        """Plots the signal and its PSD"""
        self.plot_signal(signal, label, color)
        self.plot_psd(psd, label, color)
        

    def plot_signal(self, signal: pd.Series, label: str, color: str):
        """Plots the signal on the first subplot"""
        self.fig.add_trace(go.Scatter(x=signal.index, y=signal.values, mode='lines', name=label, 
            legendgroup=label,
            line=dict(color=color, width=2), opacity=self.alpha), row=1, col=1)
        self.fig.update_xaxes(title_text=self.upper_labels[0], range=[0, 100], row=1, col=1)
        self.fig.update_yaxes(title_text=self.upper_labels[1], row=1, col=1)

    def plot_psd(self, psd: pd.Series, label: str, color: str, range = [0,1]):
        """Plots the PSD on the second subplot"""
        psd = psd[(psd.index >= range[0]) & (psd.index <= range[1])]

        self.fig.add_trace(go.Scatter(x=psd.index, y=psd.values / psd.max(), mode='lines', name=label,
            legendgroup=label,
            line=dict(color=color), opacity=self.alpha), row=2, col=1)
        self.fig.update_xaxes(title_text=self.lower_labels[0], range=range, row=2, col=1)
        self.fig.update_yaxes(title_text=self.lower_labels[1], row=2, col=1, )

    def show(self, domains=[[0,300], [0,1]]):
        def _remove_legend_duplicates():
            """Removes one of the paired traces"""
            unique_labels = set()
            for trace in self.fig['data']:
                if trace['name'] not in unique_labels:
                    unique_labels.add(trace['name'])
                    trace['legendgroup'] = trace['name']
                else:
                    trace['showlegend'] = False
            self.fig.update_layout(legend_tracegroupgap=4)
        _remove_legend_duplicates()

        self.fig.update_layout(legend=dict(title='Signals', y=1, xanchor='right'))
        self.fig.update_layout(xaxis1=dict(domain=[0, .85]))
        self.fig.update_xaxes(range=domains[0], row=1, col=1)
        self.fig.update_xaxes(range=domains[1], row=2, col=1)
        

        

        

        self.fig.show()

    def calc_and_plot(self, signal:pd.Series, label='', colour= None):
        """Uses input signal to interpolate, window and calculate psd"""
        if colour is None:
            colour = 'blue'
            # Choose a random color
            colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
            colour = np.random.choice(colours)

        # Interpolate the signal
        new_signal, psd = SincPsd.sinc_and_psd(signal, 'hann')


        # Plot the signal and its PSD
        self.plot_signal_and_psd(signal, psd, label, colour)


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
        patient_ds = pd.Series(patient_ds, index=patient_ds.cumsum() / 1000)
        DS_RR[patient_id] = patient_ds

    # Create a plotter instance
    plotter = SignalPsdPlotter()

    # Plot the data for each patient
    for patient_id, patient_ds in DS_RR.items():
        color = f'#{np.random.randint(0, 0xFFFFFF):06x}'
        plotter.calc_and_plot(patient_ds, label=f'{patient_id} DS', colour=color)

    # Show the plot
    plotter.show()


