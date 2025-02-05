"""Helper Function to plot population Metrics
By Povilas Sauciuvienas; @Sauciu1"""

from plotly.subplots import make_subplots

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def population_metric_plotter(data, y_column, color_column):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

    ratio = 1/6
    d_hor= 0.01
    fig.update_layout(xaxis1=dict(domain=[0.05, ratio-d_hor]), xaxis2=dict(domain=[ratio+d_hor, 1]))

    fig1 = px.violin(
        data,
        y=y_column,
        x="Diabetic peripheral neuropathy",
        color=color_column,
        box=True,
        hover_data=data.columns,
        category_orders={"Diabetic peripheral neuropathy": [False, True]},
        violinmode="group",
    )
    fig1.update_layout(
        title=f"Overall {y_column} Distribution",
        xaxis_title="Diabetic peripheral neuropathy",
        yaxis_title=y_column,
    )

    fig1.update_traces(width = 2)
    # Create box plot for y_column distribution by id
    fig2 = px.box(
        data,
        y=y_column,
        x="id",
        color=color_column,
        hover_data=data.columns,
        category_orders={"Diabetic peripheral neuropathy": [False, True]},
    )
    fig2.update_layout(
        title=f"{y_column} Distribution by id", xaxis_title="id"
    )

    # Add traces to the subplot
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)

    for trace in fig2.data:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        title=f"<b>{y_column} Distribution in Diabetic Peripheral Neuropathy (DPN)</b>",
        width=1200,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        violinmode="group",
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=1,
            title=dict(text="<b>Overall</b>"),
            showticklabels=False,
        ),
        xaxis2=dict(
            tickmode="linear",
            tick0=1,
            dtick=1,
            title=dict(text="<b>id</b>"),
        ),
    )

    fig.update_layout(
        legend_title_text=color_column,
        legend=dict(
            x=1,
            y=1,
            xanchor="right",
            yanchor="top",
            orientation="h",
            traceorder="normal",
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="Black",
            borderwidth=1,
        ),
    )

    fig.update_yaxes(title_text=f"<b>{y_column}</b>", row=1, col=1)

    #fig.show()
    pio.write_html(
        fig,
        file=f"html_plots/dpn_{y_column}_dashboard.html",
        auto_open=False,
        full_html=False,
        include_plotlyjs="cdn",
    )