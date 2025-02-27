import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from IPython.display import display
import numpy as np
from scipy import stats
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")


class CleanerHelper:

    def __init__(self, data: pd.DataFrame, abs_decimals=0, percent_decimals=0):

        self.data = data
        self.percent_decimals = percent_decimals
        self.abs_decimals = abs_decimals
        self.palette = sns.color_palette()

    def plot_numeric_column(self, col: str):
        """displays a numeric column given a title"""
        fig, axes = plt.subplots(3, 1, height_ratios=(1, 2, 2), figsize=(10, 4))

        sns.boxplot(x=self.data[col], ax=axes[0])
        sns.kdeplot(x=self.data[col], ax=axes[1], fill=True)
        sns.stripplot(x=self.data[col], ax=axes[2], jitter=0.4, alpha=0.4, edgecolor="black"
        )

        axes[0].set_ylabel('Boxplot', fontweight='bold')
        axes[1].set_ylabel('KDE', fontweight='bold')
        axes[2].set_ylabel('Stripplot', fontweight='bold')

        for ax in axes:
            ax.yaxis.set_visible(False)

        fig.suptitle(
            f"Numeric Distribution of {col} (n = {len(self.data[col])})",
            fontsize=10,
            fontweight="bold",
        )

        min_xlim, max_xlim = zip(*(a.get_xlim() for a in axes))
        min_xlim, max_xlim = min(min_xlim), max(max_xlim)

        for ax in axes[:-1]:
            ax.set_xlim(min_xlim, max_xlim)
            #ax.xaxis.grid(True)
            ax.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            ax.set_xlabel(None)

        axes[-1].set_xlim(min_xlim, max_xlim)
        #axes[-1].xaxis.grid(True)

        axes[-1].set_xlabel(col.title())

        #plt.tight_layout()
        

    @staticmethod
    def try_display(df: pd.DataFrame):
        try:
            display(df)
        except NameError:
            print(df)

    def plot_numerics(self):
        self.numeric_cols = self.data.select_dtypes(
            include=["int64", "float64"]
        ).columns
        for col in self.numeric_cols:
            self.try_display(self.data[col].describe())
            self.plot_numeric_column(col)
            plt.show()

    def _plot_categorical_column(self, col: str, max_bars=10):
        """Given column name, plot categorical as a countplot"""
        plt.figure(
            figsize=(10, 4),
        )
        df = self.data[col].value_counts()

        self.try_display(df)

        if len(df) > max_bars:
            print(f"Too many values ({len(df)}). Displaying 10 largest")
            df = df.sort_values(ascending=False)
            df = pd.concat(
                [df.iloc[:max_bars], pd.Series(df[max_bars:].sum(), index=["others"])]
            )

        df = df.sort_index()
        ax = sns.barplot(df)

        lbls = [f"{p[0]} ({p[1]:.0f}%)" for p in zip(df, 100 * df / df.sum())]
        ax.bar_label(container=ax.containers[0], labels=lbls)
        # ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_xlabel(col.title(), fontweight="bold")

        plt.xticks(rotation=20)
        #plt.grid(axis="y", linestyle="--", alpha=0.7)


        plt.title(f"Distribution of {col} (n={df.sum()})", fontweight="bold")
        plt.show()

    def plot_categoricals(self):
        """Plots all the categorical within the df"""
        self.categorical_cols = self.data.select_dtypes(
            exclude=["int64", "float64"]
        ).columns
        for col in self.categorical_cols:
            self._plot_categorical_column(col)

    def hue_plot(self, data_col, hue_col):
        sns.stripplot(
            data=self.data,
            x=data_col,
            hue=hue_col,
            jitter=0.4,
            alpha=0.4,
            edgecolor="black",
        )
        plt.show()

    def plot_group_agg(
        self,
        group_col: str,
        val_col: str,
        agg_func: str,
        ax=None,
        figsize=(10, 5),
        hue=None,
        title_n = False
    ):
        """Given x and y column names, aggregates and bar plots data. Allows hue"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        errorbar = ("ci", 95) if agg_func == "mean" else None

        sns.barplot(
            data=self.data,
            x=group_col,
            y=val_col,
            ax=ax,
            hue=hue,
            estimator=agg_func,
            errorbar=errorbar,
            palette= sns.color_palette(self.palette, n_colors=len(self.data[hue].unique())) if hue else None,
        )
        self._label_containers(ax, agg_func)
        n = len(self.data[[val_col, group_col]].dropna(inplace=False))
        title_text = f"{agg_func.title()}" + ("(n={n})" if title_n else '')

        if agg_func == "mean":
            title_text += "; Bootstrapped 95% CI"

        ax.set_title(title_text, fontweight="bold")
        ax.set_xlabel(group_col, fontweight="bold")
        ax.set_ylabel(f"{agg_func.title()} of {val_col}", fontweight="bold")

       # ax.yaxis.grid(False, alpha=0.5)

    def _label_containers(self, axis, agg_func):
        """Adds container data value labels.
        Special case for mean [95% CI]"""

        for container in axis.containers:
            labels = [
                f"{v:.{self.abs_decimals}f}\n{100 * v / sum(container.datavalues):.{self.percent_decimals}f}%"
                for v in container.datavalues
            ]
            if agg_func in ["mean", 'median']:
                axis.bar_label(container, labels=labels, label_type="center")

            else:
                axis.bar_label(container, labels=labels)


    def plot_side_agg(
        self,
        group_col: str,
        val_col: str,
        agg_functions: list[str],
        hue: str = None,
        figsize=(16, 5),
        ratios: list[int] = None,
    ):
        if ratios is not None and len(agg_functions) != len(ratios):
            raise IndexError("Ratios and agg_functions length mismatch")

        fig, axes = plt.subplots(
            ncols=len(agg_functions), width_ratios=ratios, figsize=figsize
        )

        n = len(self.data[[val_col, group_col]].dropna(inplace=False))
        fig.suptitle(
            f"{val_col} Aggregations by {group_col} (n={n})", fontweight="bold"
        )

        for func, axis in zip(agg_functions, axes):
            self.plot_group_agg(group_col, val_col, func, ax=axis, hue=hue)

        plt.tight_layout()


    def _gaussian_ci(self, data: pd.Series, confidence: float):
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        return mean, mean-h, mean+h
    
    def _bootstrap_ci(self, data: pd.Series, confidence: float, n_bootstrap=1000):
        bootstrapped_means = []
        for _ in range(n_bootstrap):
            sample = data.sample(frac=1, replace=True)
            bootstrapped_means.append(sample.mean())
        lower_bound = np.percentile(bootstrapped_means, (1 - confidence) / 2 * 100)
        upper_bound = np.percentile(bootstrapped_means, (1 + confidence) / 2 * 100)
        return np.mean(bootstrapped_means), lower_bound, upper_bound

    def plot_kde_confidence(self, group_col, val_col, confidence=0.95, figsize = (10,4), display_statistics= False, method = "bootstrap"):
        fig, ax = plt.subplots(figsize=figsize)

        unique_groups = self.data[group_col].unique()
        color_dict = dict(zip(unique_groups, sns.color_palette(self.palette, len(unique_groups))))

        sns.kdeplot(data=self.data, x=val_col, hue=group_col, ax=ax, fill=True, alpha=0.2, palette=color_dict)

        display_stats = pd.DataFrame()

        for kde in self.data[group_col].unique():
            sub_data = self.data[self.data[group_col] == kde][val_col]

            method = self._bootstrap_ci if method == "bootstrap" else self._gaussian_ci
            mean, ci_lower, ci_upper = method(sub_data, confidence)
            display_stats = pd.concat([display_stats, pd.DataFrame({group_col: [kde], 'mean': [mean], 'ci_lower': [ci_lower], 'ci_upper': [ci_upper]})], ignore_index=True)
            
            ax.axvline(mean, linestyle='--', color=color_dict[kde], label=f'{kde} Mean')
            ax.fill_betweenx(ax.get_ylim(), ci_lower, ci_upper, color=color_dict[kde], alpha=0.3, label=f'{kde} 95% CI')

        ax.legend(title=group_col)

        ax.set_title(f'KDE Plot of {val_col.title()} with Mean and 95% CI for Each {group_col.title()}', fontweight='bold')
        ax.set_xlabel('Sales in Thousands', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')

        ax.legend(title=f'{group_col.title()} Groups')
        #ax.xaxis.grid('x', linestyle='-', linewidth=0.5)
        plt.show()

        if display_statistics:
            self.try_display(display_stats)



if __name__ == "__main__":
    data = pd.read_csv(r"fast_food/WA_Marketing-Campaign.csv")
    helper = CleanerHelper(data)
    helper.plot_side_agg(
        "MarketID",
        "SalesInThousands",
        ["sum", "mean"],
        hue="MarketSize",
        ratios=[1, 1.6],
    )
    
    helper.plot_side_agg("week", "SalesInThousands", ["mean", "sum", "max", "min"])
    
    helper.plot_kde_confidence('Promotion', 'SalesInThousands', display_statistics=True)

    plt.show()
