import pandas as pd
from general_utils.OverallDetailComparisonPlot import OverallDetailComparisonPlot

def make_decile_column(data: pd.DataFrame, column: str):
    """For each id, splits the data into deciles based on the column
    decile column name: {column}_decile"""
    data[f"{column}_decile"] = data.groupby("id")[column].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False)
    )
    return data

def draw_by_decile(data, column, x= 1000, y= 400):
    title = f"{column} in PDR patients"
    plotter = OverallDetailComparisonPlot(
        data=data.sort_values("DPN", ascending=True),
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