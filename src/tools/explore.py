#!/usr/bin/python3
import pandas as pd
import missingno as msno
import pandas_profiling


class Explore:

    def __init__(self, df: pd.DataFrame):
        """
        Class for data exploration.
        :param df: Input dataframe
        """
        self.df = df

    def print_dims(self) -> str:
        """
        Print the dimensions and features of a dataframe.
        :return: String with data dimensions
        """
        rows = self.df.shape[0]
        cols = self.df.shape[1]
        print(f'Dataframe features:')
        for c in self.df.columns:
            print(c)
        return f'The dataframe consist of {cols} features and {rows} records'

    def top_unique_values(self, top_number: int):
        """
        Print counts and data types of the top # unique values of a dataframe.
        :param top_number: Number of top results to print
        :return: print top # unique values
        """
        counter = 0
        for i in self.df.columns:
            x = self.df.loc[:, i].unique()
            print(counter, i, type(self.df.loc[0, i]), len(x), x[0:top_number])
        counter += 1
        return

    def missing_values(self):
        """
        Analyse content of missing values in a dataframe.
        :return: Missing value co-occurrence
        """
        return msno.matrix(self.df, figsize=(6, 5), fontsize=10)

    def data_profiling(self, out_path: str) -> pandas_profiling.profile_report.ProfileReport:
        """
        Perform pandas profiling on dataframe.
        :param out_path: Path with filename for output (.html extension)
        :return: HTML summary report and a string
        """
        profile = pandas_profiling.ProfileReport(self.df)
        profile.to_file(output_file=out_path)
        print(f'(Data profiling completed. HTML file can be found here: {out_path})')
        return profile.to_widgets()
