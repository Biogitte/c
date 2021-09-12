#!/usr/bin/python3
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import math
import numpy as np
import matplotlib as plt


class CategoricalPlotter:

    def __init__(self, df, label):
        self.df = df
        self.label = label
        self.long = pd.melt(self.df, id_vars=[self.label])

        self.x = self.long['variable']
        self.y = self.long['value']

        self.dims = (10, 8)

        self.col = 'vlag'

        sns.set_style('ticks')
        self.fig, self.ax = plt.subplots(figsize=self.dims)

    def box_basic(self):
        sns.boxplot(ax=self.ax, x=self.y, y=self.x, data=self.long, hue=self.label,
                    width=.6, palette=self.col)

    def box_swarm(self):
        sns.boxplot(ax=self.ax, x=self.x, y=self.y, hue=self.label, data=self.long, palette=self.col)
        sns.swarmplot(x=self.x, y=self.y, hue=self.label, dodge=True, data=self.long, alpha=.8, color='grey', s=4)

    def violin_basic(self):
        sns.violinplot(ax=self.ax, x=self.x, y=self.y, hue=self.label, data=self.long,
                       palette=self.col, split=False, scale='count',
                       scale_hue=False, bw=.2)
        sns.despine(left=True)

    def violin_split(self):
        sns.violinplot(ax=self.ax, x=self.long['variable'], y=self.long['value'],
                       hue=self.label, data=self.long, palette='rainbow',
                       split=True, scale='count', inner='stick', linewidth=1,
                       scale_hue=False, bw=.2)
        sns.despine(left=True)

    def swarm(self):
        sns.swarmplot(ax=self.ax, x=self.long['variable'], y=self.long['value'],
                      data=self.long, hue=self.label,
                      dodge=True, palette='rainbow')

    def boxen(self):
        """
        Recommended for large datasets with > 10,000 records.
        """
        sns.boxenplot(ax=self.ax, x=self.long['variable'], y=self.long['value'],
                      hue=self.label, data=self.long, palette='rainbow')

    def point(self):
        sns.pointplot(ax=self.ax, x=self.long['variable'], y=self.long['value'],
                      hue=self.label, data=self.long, palette='rainbow')

    def strip(self):
        sns.stripplot(ax=self.ax, x=self.long['variable'], y=self.long['value'],
                      data=self.long, jitter=True, hue=self.label,
                      dodge=True, palette='rainbow')

    def cat_scatter(self):
        sns.catplot(ax=self.ax, x=self.long['value'], y=self.long['variable'],
                    hue=self.label, kind='swarm', data=self.long)


class CountPlotter:

    def __init__(self, df, label):
        self.df = df
        self.label = label
        self.long = pd.melt(self.df, id_vars=[self.label])

        self.plt.figure(figsize=(10, 8))
        self.fig, self.ax = plt.subplots(figsize=self.dims)

    def count(self):
        sns.countplot(ax=self.ax, x=self.long['variable'], y=self.long['value'],
                      hue=self.label, data=self.long, palette='rainbow')

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


class CorrelationPlotter:

    def __init__(self, df, label):
        self.df = df
        self.label = label
        self.long = pd.melt(self.df, id_vars=[self.label])

        self.plt.figure(figsize=(10, 8))
        self.fig, self.ax = plt.subplots(figsize=self.dims)

    def num_corr_plot(self):
        """Correlation plot of numerical values"""
        numerical = self.df.select_dtypes([np.number]).columns
        corr_matrix = self.df[numerical].corr()
        return sns.heatmap(corr_matrix)

    def scatter(self):
        # TODO
        pass

    def connect_scatter(self):
        # TODO
        pass

    def heatmap(self):
        # TODO
        pass

    def correlogram(self):
        # TODO
        pass

    def bubble(self):
        # TODO
        pass

    def two_dim_density(self):
        # TODO
        pass


class DistributionPlotter:
    def __init__(self, df, label):
        self.df = df
        self.label = label
        self.long = pd.melt(self.df, id_vars=[self.label])

        self.plt.figure(figsize=(10, 8))
        self.fig, self.ax = plt.subplots(figsize=self.dims)
        sns.set_theme(style='whitegrid')

    def violin(self):
        sns.violinplot(ax=self.ax, x=self.long['variable'], y=self.long['value'],
                       hue=self.label, data=self.long, palette="Set2",
                       split=True, scale='count', inner='stick', linewidth=1,
                       scale_hue=False, bw=.2)
        sns.despine(left=True)

    def density(self, n_cols: int):
        """ Create density plot.
        :param n_cols: Set the number of graph columns.
        """
        features = self.df.columns.to_list()
        n_rows = math.ceil(len(features) / n_cols)
        fig = self.df[features].plot(kind='density', subplots=True, sharex=False,
                                     layout=(n_rows, n_cols), figsize=(10, 20))
        return fig

    def histogram(self):
        # TODO
        pass

    def box(self):
        # TODO
        pass

    def ridge(self):
        # TODO
        pass


class RankPlotter:
    def __init__(self, df):
        self.df = df

    def bar(self):
        # TODO
        pass

    def circle_bar(self):
        # TODO
        pass

    def spider(self):
        # TODO
        pass

    def word_cloud(self):
        # TODO
        pass

    def parallel(self):
        # TODO
        pass

    def lollipop(self):
        # TODO
        pass
