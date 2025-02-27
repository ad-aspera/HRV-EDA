"""
Helper Function to plot graphics side by side for comparison. Adapted for Multiple UseCases
By Povilas Sauciuvienas; @Sauciu1
Last edited on 2025-02-08
"""

from plotly.subplots import make_subplots
import os
import plotly.express as px
import plotly.io as pio
from typing import Literal
import plotly.graph_objects as go


class OverallDetailComparisonPlot:
    def __init__(self, data, x_column, y_column, hue_column, title, detail_mode:Literal["violin", "box", "strip"]='box', points:Literal["all", None]=None, category_orders=None):
        """Creates a plot that compares overall and detailed data side by side. Adapted for Multiple UseCases"""
        self.data = data
        self.y_column = y_column
        self.x_column = x_column
        self.hue_column = hue_column
        self.title = title
        self.category_orders = category_orders

        self.fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
        self.setup_general_layout()


        self.overall_plot = self._create_overall_plot()
        self.detailed_plot = self._create_detail_plot(plot_mode=detail_mode, points=points)

        self._add_all_traces()
        self._update_layout()


        

    def setup_general_layout(self, width=1200, height=600, ratio=1/6, d_hor=0.005):
        """Sets up the layout of the figure itself"""
        self.fig.update_layout(xaxis1=dict(domain=[0.05, ratio-d_hor]), xaxis2=dict(domain=[ratio+d_hor, 1]))
        self.fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=10, r=10, t=40, b=20),
            title=dict(
                text=self.bolden(self.title),
                x=0.5,
                xanchor='center',
                yanchor='top'
            )
        )

    def _create_overall_plot(self):
        """Creates the overall (hue_group only) plot on the left"""
        self.fig1 = px.violin(
            self.data,
            x=self.hue_column,
            y=self.y_column,
            color=self.hue_column,
            box=True,
            hover_data=self.data.columns,
            category_orders=self.category_orders,
            violinmode="group",
        )

             
        self.fig1.update_traces(width=1.5,spanmode="hard", side = 'positive')
        
        #Traces added separately to allow overriding data and hover
        return self.fig1



    def _create_detail_plot(self, plot_mode: Literal["violin", "box", "strip"] = "box", points:Literal["all", None]=None):
        """Creates the detailed (x_column) plot on the right"""
        plot_func = {
            "violin": px.violin,
            "box": px.box,
            "strip": px.strip
        }.get(plot_mode, px.box)

        plot_args = dict(
            violin=dict(box=True, violinmode="group", points=points),
            box=dict(boxmode="group",points=points),
            strip=dict()
        ).get(plot_mode, {})

        self.fig2 = plot_func(
            self.data,
            x=self.x_column,
            y=self.y_column,
            color=self.hue_column,
            category_orders=self.category_orders,
            **plot_args
        )

        if plot_mode == "violin":
            self.fig2.update_traces(spanmode="hard", side = 'positive')

        #Traces added separately to allow overriding data and hover
        return self.fig2
    

    def _add_all_traces(self):
        def _add_traces(fig, row, col):
            """Adds traces to the main figure"""
            for trace in fig.data:
                self.fig.add_trace(trace, row=row, col=col)

        
        _add_traces(self.fig1, row=1, col=1)
        _add_traces(self.fig2, row=1, col=2)

      

        self.update_legend()


    def override_hover_data(self, hover_data: list):
        """Overrides the hover data for the detail plot"""
        for trace in self.fig2.data:
            trace.hovertemplate = '<br>'.join([f'{col}: %{{customdata[{i}]}}' for i, col in enumerate(hover_data)])
            trace.customdata = self.data[hover_data].values

        self._add_all_traces()


    def _update_layout(self):
        #Necessary to update after adding traces.
        self._update_layout_overall_plot()
        self._update_layout_detailed_plot2()
        self.update_legend()

        self.fig.update_layout(boxmode='group', violinmode='group')



    def _update_layout_overall_plot(self):
        """Updates the layout of the overall plot"""
        self.fig.update_layout(
            xaxis=dict(
                tickmode="linear",
                tick0=0,
                dtick=1,
                title=dict(text=self.bolden("Overall")),
                showticklabels=False,
            ),
        )

        def _set_y_range():
            """Sets the y range for the overall plot"""

            y_min = self.data[self.y_column].min()
            y_max = self.data[self.y_column].max()
            length = y_max - y_min

            y_min = y_min - 0.05*length if y_min != 0 else 0

            self.fig.update_yaxes(range=[ y_min, y_max+0.05*length], row=1, col=1)
        #_set_y_range()





    def _update_layout_detailed_plot2(self)->None:
        """Updates the layout of the detailed plot"""
        self.fig.update_layout(
            xaxis2=dict(
                tickmode="linear",
                tick0=1,
                dtick=1,
                title=dict(text=self.bolden(f"By {self.x_column}")),
                type="category",
            ),
        )

    def update_legend(self, legend_title:str=None, hue_labels:dict=None)->None:
        """Updates the legend of the plot"""

        def _remove_legend_duplicates():
            """Remove duplicate legend items for hue"""
            unique_legend_items = {}
            for trace in self.fig.data:
                if trace.name not in unique_legend_items:
                    unique_legend_items[trace.name] = trace
                else:
                    trace.showlegend = False

        _remove_legend_duplicates()

        legend_title = legend_title or self.hue_column

        if hue_labels:
            for trace in self.fig.data:
                if trace.name in hue_labels:
                    trace.name = hue_labels.get(trace.name, trace.name)

        self.fig.update_layout(
            legend_title_text=legend_title,
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
        self.fig.update_yaxes(title_text=self.bolden(self.y_column), row=1, col=1)
    
    def override_axes_labels(self, x_label:str= None, y_label:str=None)->None:
        """Overrides the axes labels of the plot"""
        
        if x_label:
            self.fig2.update_xaxes(title_text=self.bolden(x_label))

        if y_label:
            self.fig1.update_yaxes(title_text=self.bolden(y_label))




    def show(self):
        """Shows the plot in the browser or Jupyter notebook"""
        self.fig.show()

    def save(self, parent_folder="html_plots", save_name=None)->str:
        """Saves the plot to an html file"""
        

        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)
        
        save_name = (save_name or self.title).replace(" ", "_")
        output_path = os.path.join(parent_folder, f"{save_name}.html")

        pio.write_html(
            self.fig,
            file=output_path,
            auto_open=False,
            full_html=False,
            include_plotlyjs="cdn",
        )
        return output_path

    @staticmethod
    def bolden(text):
        """Wraps text in bold HTML tags"""
        return f"<b>{text}</b>"



def _generate_test_data():
    import numpy as np
    import pandas as pd
    import random

    np.random.seed(42)
    random.seed(42)

    data = {
        "SalesInThousands": np.random.normal(100, 20, 1000),
        "MarketID": np.random.choice([1, 2, 3], 1000),
        "Promotion": np.random.choice([1, 2, 3], 1000),
        "LocationID": np.random.choice([1, 2, 3], 1000),
    }

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":    
    import pandas as pd
    df = _generate_test_data()

    df["Promotion"] = pd.Categorical(df["Promotion"], categories=[1, 2, 3], ordered=True)
    df["MarketID"] = pd.Categorical(df["MarketID"], ordered=True)

    
    
    plotter = OverallDetailComparisonPlot(
        df, 
        y_column="SalesInThousands", 
        x_column="MarketID", 
        hue_column="Promotion",
        title ="Hello",
        detail_mode="violin",
        points="all",
        category_orders={"Promotion": [1, 2, 3]}
    )

    #plotter.update_legend(legend_title="Market Size", hue_labels={"Small": "S", "Medium": "M"})
    plotter.override_axes_labels(x_label="Market ID", y_label="Sales (in Thousands)")
    plotter.setup_general_layout(ratio=1/4)

    plotter.override_hover_data(["MarketID", "Promotion", "SalesInThousands", "LocationID"])




    plotter.show()