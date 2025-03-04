import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Cleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.copy = data.copy()
        self.removed = pd.DataFrame()

    def filter_threshold(self, metric: str, threshold: float, comparison: str = 'upper', ids = None) -> None:
        """
        Remove values based on threshold comparison in a metric for a given set of ids
        
        Parameters:
        - metric: column name in the dataframe
        - threshold: threshold value for comparison
        - comparison: 'upper' to remove values > threshold, 'lower' to remove values < threshold
        - ids: list of IDs to apply the filter to (default: all IDs)
        """
        if ids is None:
            ids = self.data.id.unique()
        
        id_mask = self.data["id"].isin(ids)
        
        if comparison.lower() == 'upper':
            metric_mask = self.data[metric] > threshold
        elif comparison.lower() == 'lower':
            metric_mask = self.data[metric] < threshold
        else:
            raise ValueError("comparison must be either 'upper' or 'lower'")
            
        outliers = (id_mask & metric_mask)
        
        self.removed = pd.concat([self.removed, self.data[outliers]])
        self.data = self.data[~outliers]
        
        return self
    
    def upper(self, metric: str, threshold: float, ids = None) -> None:
        "Remove values passing an upper threshold in a metric for a given set of ids"
        return self.filter_threshold(metric, threshold, 'upper', ids)
    
    def lower(self, metric: str, threshold: float, ids = None) -> None:
        "Remove values passing a lower threshold in a metric for patients in ids"
        return self.filter_threshold(metric, threshold, 'lower', ids)
 

    
    def upper_array(self, metric: str, array: list) -> None:
        """Call cleaner.upper on array"""
        for threshold, ids in array:
            self.upper(metric, threshold, ids)

    def lower_array(self, metric: str, array: list) -> None:
        for threshold, ids in array:
            self.lower(metric, threshold, ids)
    
    def draw_metric(self, metric: str):
        
        self.data.sort_values(by=["id", metric], inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.stripplot(x='id', y=metric, hue='DPN', data=self.data, jitter=True, alpha=0.5, ax=ax)
        ax.set_title(f'Distribution of {metric} by ID and DPN Status')
        ax.set_xlabel('ID')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=90)
        ax.grid(axis='y', linestyle='dashed', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        return self