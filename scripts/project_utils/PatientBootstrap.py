import numpy as np
import pandas as pd
import os
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

class PatientBootstrap():
    """Bootstraps a distribution of a metric for a group of ids.
    Bootstraps mean by repeatedly sampling 1 datum from each patient id;
    then a mean for DPN and control groups is calculated.
    Tries to Automatically load memory from a pickle file if it exists.
    Args:
        data (pd.DataFrame): The data to sample from
        n (int): Number of means to sample
        file_name (str): Name of the file to save the memory of the bootstrapper
        save_folder (str): Folder to save the memory of the bootstrapper
    """

    def __init__(self, data:pd.DataFrame, n:int=5000, file_name ="bayes_pickle.pkl", save_folder = "processed_data/bayesian_bootstrap"):
        assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
        self.data = data
        assert isinstance(n, int), "n must be an integer"
        self.n = n

        assert isinstance(file_name, str), "file_name must be a string"
        assert isinstance(save_folder, str), "save_folder must be a string"
        self.file_name = file_name
        self.save_folder = save_folder

        self.memory = {}

    def sample_metric(self, metric:str, id:str):
        """Sample values a metric with replacement given id"""
        return self.data.loc[self.data['id'] == id, metric].sample(self.n, replace=True).values

    def get_distribution(self, metric:str, ids:str, agg_func=np.mean):
        """Get a distribution of a metric for a group of ids"""
        values = [self.sample_metric(metric, ID) for ID in ids]

        values = list(zip(*values))
        distribution = [agg_func(group) for group in values]
        return distribution
    
    def sample_dpn_control(self, metric:str, dpn_id:list, control_id:list, agg_func=np.mean):
        """RENAME, get the distribution of a metric for both DPN and control group"""
        self.memory[metric] ={'DPN': self.get_distribution(metric, dpn_id),
                                'Control':self.get_distribution(metric, control_id, agg_func=np.mean)}
        return self.memory[metric]
    
    def access_memory_metric(self, metric:str):
        """Returns the memory of a metric"""
        return self.memory[metric]
    
    def pickle_memory(self):
        """Saves memory to a pickle file"""

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        path = os.path.join(self.save_folder,self.file_name)
        with open(path, 'wb') as f:
            pkl.dump(self.memory, f)

    def load_memory(self):
        """Loads memory from a pickle file"""
        path = os.path.join(self.save_folder,self.file_name)
        try:
            with open(path, 'rb') as f:
                self.memory = pkl.load(f)
        except FileNotFoundError:
            print("No memory file found")

    def _eval_bayes_metric(self, control, dpn, metric):
        """Evaluates the bayesian distribution for a given metric"""
        control = pd.Series(control)
        dpn = pd.Series(dpn)

        bayes_p = np.mean(control < dpn)
        results = {
            'Metric': metric,
            'Bayes_p': min(bayes_p, 1-bayes_p),
            'DPN_Mean': np.mean(dpn),
            'DPN_Std': np.std(dpn),
            'Control_Mean': np.mean(control),
            'Control_Std': np.std(control)
        }
        results_df = pd.DataFrame([results])
        return results_df

    def evaluate_bayes_metrics(self):
        """Returns the bayes metrics evaluated as a DataFrame"""
        bayes_table = pd.DataFrame()
        for metric in self.memory.keys():
            dpn = self.memory[metric]['DPN']
            control = self.memory[metric]['Control']
            res = self._eval_bayes_metric(control, dpn, metric)
            bayes_table = pd.concat ([bayes_table, res])
        return bayes_table
    

    def draw_metric(self, control, dpn, metric):
        plt.figure(figsize=(8, 4))

        def histplot(data, colour):
            sns.histplot(
                data, alpha=0.5, color=colour, element="step", fill=True, stat="density"
            )

        histplot(control, "blue")
        histplot(dpn, "red")

        plt.title(f"Bootstrapped (n={self.n}) Mean Distribution of {metric}")
        plt.legend(["Control", "DPN"])
        plt.xlabel(metric)
        plt.show()


    def show_bayes_metrics(self,metrics: list = None):  
        bayes_table = self.evaluate_bayes_metrics()
        """Simplifies the process of displaying the bayes metrics and the corresponding density plots"""
        if metrics is None:
            metrics = list(self.memory.keys)
        if not isinstance(metrics, list):
            metrics = [metrics]

        for metric in metrics:
            dpn = self.memory[metric]["DPN"]
            control = self.memory[metric]["Control"]
            display(bayes_table[bayes_table["Metric"] == metric])
            self.draw_metric(control, dpn, metric)