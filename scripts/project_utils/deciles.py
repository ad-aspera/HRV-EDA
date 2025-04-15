import pandas as pd
from scripts.general_utils.OverallDetailComparisonPlot import OverallDetailComparisonPlot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def make_decile_column(data: pd.DataFrame, column: str):
    """For each id, splits the data into deciles based on the column
    decile column name: {column}_decile"""
    data[f"{column}_decile"] = data.groupby("id")[column].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False)
    )
    return data


def agg_metric_deciles(data, metric_column, aggfunc="median"):
    """Uses prepared metric decile to aggregates each metric by it"""
    decile_agg_data = data.groupby(["DPN", "id", f"{metric_column}_decile"]).agg(
        {metric_column: aggfunc}
    )
    decile_agg_data = decile_agg_data.reset_index()
    decile_agg_data = decile_agg_data.rename(columns={metric_column: "value", f"{metric_column}_decile": "decile"})
    return decile_agg_data


def pivot_metric_decile(data, metric_column, aggfunc='median'):
    """Calculates the deciles and aggregates each metric by it"""
    data = make_decile_column(data, metric_column)
    decile_agg_data = agg_metric_deciles(data, metric_column, aggfunc)
    pivot = pd.pivot_table(decile_agg_data, values='value', index='id', columns='decile', aggfunc=aggfunc)
    pivot = pivot.join(data[['id', 'DPN']].drop_duplicates().set_index('id'))
    #display(pivot)
    return pivot


def plot_p_value_heatmap(channels, threshold = 0.1,  figsize=(8, 4)):
    channel_pivot = pd.pivot_table(
        channels, values="p_value", index="metric", columns="decile"
    )

    plt.figure(figsize=figsize)
    white_cmap = ListedColormap(["white"])

    ax1 = sns.heatmap(channel_pivot, cmap=white_cmap, cbar=False, annot=True, fmt=".2f")
    ax2 = sns.heatmap(
        channel_pivot[channel_pivot < threshold], cmap="rocket", annot=True, fmt=".2f", cbar_kws={"label": "P-value"}
    )

    for ax in [ax1, ax2]:
        if ax is None:
            continue
        for text in ax.texts:
            current_text = text.get_text()
            if current_text.startswith("0"):
                text.set_text(current_text[1:])

    plt.axvline(x=5, color="red", linestyle="--", linewidth=2)
    plt.title(f"P-values (p<{threshold}) of Metrics by Medians of Deciles")
    plt.xlabel("Decile")
    plt.ylabel("Metric")
    plt.tight_layout()


def plot_decile_dist(channels, metric, dec_range=None):
   # dec = man_whitney_channels[metric]
    agg_decile = channels[channels['metric'] == metric]
    plt.figure(figsize=(8, 5))
    sns.stripplot(
        data=agg_decile,
        x="decile",
        y="value",
        hue="DPN",
        dodge=True,
        size=4
    )
    if dec_range is not None:
        plt.gca().add_patch(
            plt.Rectangle(
                (dec_range[0]-0.4, agg_decile['value'].min()*0.5), 
                dec_range[-1] - dec_range[0] + 0.8, 
                agg_decile['value'].max() - agg_decile['value'].min()*0.5, 
                edgecolor='red', 
                facecolor='none', 
                linestyle='--', 
                linewidth=2
            )
        )

    plt.title(f"{metric} distribution by deciles")
    plt.show()



def draw_by_decile(data, column, x= 1000, y= 400):
    title = f"{column} in PDR patients"
    plotter = OverallDetailComparisonPlot(
        data=data.sort_values(["DPN", 'id'], ascending=True),
        x_column="id",
        y_column=column,
        hue_column=f"{column}_decile",
        title=title,
        category_orders={f"{column}_decile": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
        detail_mode="strip",
    )


    plotter.fig.add_shape(
        type="rect",
        x0=len(data["id"].unique())-4.5,
        x1=len(data["id"].unique()),
        y0=data[column].min()-data[column].max()*0.05,
        y1=data[column].max()*0.93,
        line=dict(color="red", width=2),
        fillcolor="rgba(255, 0, 0, 0.2)",
        name="Highlight",
        row=1,
        col=2,
    )

    plotter.update_legend()

    plotter.setup_general_layout(x, y, 1 / 4)
    plotter.show()

def produce_median_melt(data, metrics):
    median_df = data.groupby(['id', 'DPN'])[metrics].median()
    median_df = median_df.reset_index()


    median_melted = pd.melt(
        median_df, 
        id_vars=['id', 'DPN'], 
        value_vars=metrics,
        var_name='metric', 
        value_name='value'
    ).reset_index(drop=True)

    return median_melted