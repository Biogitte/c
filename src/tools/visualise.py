#!/usr/bin/python3
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import math
import numpy as np


class Plotter:
    def __init__(self, df: pd.DataFrame, style: str):
        """
        Class for creating visualisations of data.
        :param df: Input dataframe
        """
        self.df = df
        self.style = style

    def histogram(self, feat: str):
        """
        Create single-feature histogram.
        style can be "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"
        :param feat: Feature to plot
        """
        x_values = self.df[feat]
        fig = go.Figure(go.Histogram(x=x_values, histnorm='probability'))
        fig.update_layout(
            template=self.style, title_text=feat,
            xaxis_title_text='Value', yaxis_title_text='Count',
            bargap=0.1, height=350, width=400)
        fig.update_traces(marker_color='grey', opacity=0.75)
        return fig.show()

    def histogram_panel(self):
        """ Plot histogram of all features in a dataframe."""
        features = self.df.columns.to_list()
        for feat in features:
            self.histogram(feat)
        return

    def density_plot(self, n_cols: int):
        """ Create density plot.
        :param n_cols: Set the number of graph columns.
        """
        features = self.df.columns.to_list()
        n_rows = math.ceil(len(features) / n_cols)
        fig = self.df[features].plot(kind='density', subplots=True, sharex=False,
                                     layout=(n_rows, n_cols), figsize=(10, 20))
        return fig

    def corr_plot(self):
        """Correlation plot of numerical values"""
        numerical = self.df.select_dtypes([np.number]).columns
        corr_matrix = self.df[numerical].corr()
        return sns.heatmap(corr_matrix)
