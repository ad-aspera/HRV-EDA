import kaggle
import seaborn as sns
import plotly.express as px
import matplotlib as mpl 
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from IPython.display import display
from scipy import stats

def kaggle_download(url:str, path="."):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        url, path=path, unzip=True
    )

def pd_display_settings(significant_figures = 4):
    def _display_rule(x:float|int)->str:
        if int(x) == x or x>=1000:
            return f'{x:.0f}'
        return f'{x:.{significant_figures}g}'
    pd.options.display.float_format = _display_rule


class default_plot_format():
    def __init__(self):
        """Makes default plot style uniform between seaborn and plotly"""
        self.set_palette()
        self.bold_title_labels()

    @staticmethod
    def set_palette():
        "sets Seaborn palette to plotly default"

        palette = px.colors.qualitative.Plotly
        

        palette, _ = px.colors.convert_colors_to_same_type(palette, colortype='tuple')
        palette = [mpl.colors.rgb2hex(color) for color in palette]
        px.defaults.color_discrete_sequence = palette

        sns.set_palette(sns.color_palette(palette))

        continuous_palette = px.colors.sequential.Plasma
        #sns.set_palette(sns.color_palette(continuous_palette, as_cmap=True))
        #sns.set_palette(sns.color_palette(continuous_palette, as_cmap=True))

        return palette, continuous_palette
        
    @staticmethod
    def bold_title_labels():
        """Sets titles and labels to be bold by default"""
        mpl.rcParams['axes.titleweight'] = 'bold'
        mpl.rcParams['axes.labelweight'] = 'bold'
        sns.set_context("notebook", rc={"axes.labelweight": "bold"})

def chi_squared_test(data, row_col, col_col):
    """Performs a chi-squared test of independence between two categorical variables.
    Outputs the results in a nicely formatted table."""

    print(f"Evaluation of random assignment between {row_col} and {col_col} results:")

    contingency_table = pd.crosstab(data[row_col], data[col_col])
    
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    expected = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
    
    results_df = pd.DataFrame({
        "Chi-Squared Test Statistic": [chi2],
        "P-value": [p],
        "Degrees of freedom": [dof]
    })
    results_df.index=['']

    display(results_df.transpose())

    merged_table = pd.concat([contingency_table, expected], axis=1, keys=['Observed', 'Expected'])


    merged_table.insert(len(contingency_table.columns), ('', '', ), '')
    merged_table.insert(len(contingency_table.columns) + 1, (' ', ' ', ), ' ')
    
    print("\n","The Observed and Expected distributions of instances:")
    display(merged_table)

    return contingency_table, p, row_col, col_col



if __name__ == "__main__":
    df = pd.read_csv("fast_food/WA_Marketing-Campaign.csv")

    chi_squared_test(df, 'MarketID', 'Promotion');
