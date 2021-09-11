#!/usr/bin/python3
import os
import glob
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
import time


class CleanData:
    """
    Pre-processing and data cleaning of CSV files.
    """

    def __init__(self, path: str, out_path: str, unwanted_cols=None, num_how='mean', cat_how='mode'):
        """
        :param path: Input path to directory with CSVs or single CSV file.
        :param out_path: Output directory or file path.
        :param unwanted_cols: List of columns to remove.
        """
        self.path = path
        self.out_path = out_path
        self.unwanted_cols = unwanted_cols
        self.df = self.load_data()
        self.num_how = num_how
        self.num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.to_list()
        self.cat_how = cat_how
        self.cat_cols = self.df.select_dtypes(include=['object', 'bool']).columns.to_list()

        self.timestr = time.strftime('%Y%m%d')

    def merge_multiple_csv(self) -> pd.DataFrame:
        """ Merge multiple CSV files present in the same directory."""
        ext = 'csv'
        files = [i for i in glob.glob(self.path + '/*.{}'.format(ext))]
        combined_df = pd.concat([pd.read_csv(f) for f in files])
        return combined_df

    def load_data(self) -> pd.DataFrame:
        """ Import and merge multiple CSV files OR import a single CSV file. """
        if os.path.isfile(self.path):
            self.df = pd.read_csv(self.path)
            return self.df
        elif os.path.isdir(self.path):
            self.df = self.merge_multiple_csv()
            return self.df
        else:
            print('File or directory does not exist')

    def get_base_name(self) -> str:
        """ Get the base name of a file or directory. """
        if os.path.isfile(self.path):
            name = os.path.splitext(os.path.basename(self.path))[0]
            return name
        elif os.path.isdir(self.path):
            name = os.path.basename(self.path)
            return name
        else:
            print('File or directory does not exist')

    def create_qc_report(self, out_name: str):
        """ Create a Pandas Profiling QC HTML report from a dataframe. """
        title = out_name + ' - QC report'
        qc = ProfileReport(self.df, title=title, explorative=True)
        qc.to_file(f"{self.out_path}/{self.timestr}_{out_name}.html")

    def drop_unwanted_columns(self):
        """
        Remove unwanted columns from a Pandas dataframe.
        Inputs a Pandas dataframe and a list of columns to be removed.
        """
        if self.unwanted_cols is None:
            return self.df
        else:
            self.df = self.df.drop(self.unwanted_cols, axis=1)
            return self.df

    def df_str_to_lowercase(self):
        self.df.columns = self.df.columns.str.lower()  # column names to lower case
        self.df = self.df.apply(lambda x: x.astype(str).str.lower() if (x.dtype == 'object') else x)
        return self.df

    def remove_duplicate_rows(self):
        """ Remove duplicate rows of a dataframe."""
        print("There are", len(self.df) - len(self.df.drop_duplicates()), "duplicate rows to remove.")
        self.df = self.df.drop_duplicates()
        return self.df

    def impute_missing_numeric_values(self):
        """
        Impute missing numeric values.
        num_how options are 'mean', 'median', 'mode' or a specified numeric value
        cols is a list of columns
        """
        if self.num_how == 'mean':
            for i in self.num_cols:
                print("Imputing {0} with mean: {1}".format(i, self.df.loc[:, i].mean()))
                self.df.loc[:, i] = self.df.loc[:, i].fillna(self.df.loc[:, i].mean())
            return self.df

        elif self.num_how == 'median':
            for i in self.num_cols:
                print("Imputing {0} with median: {1}".format(i, self.df.loc[:, i].median()))
                self.df.loc[:, i] = self.df.loc[:, i].fillna(self.df.loc[:, i].median())
            return self.df

        elif type(self.num_how) == int or type(self.num_how) == float:
            for i in self.num_cols:
                print("Imputing {0} with value: {1}".format(i, self.num_how))
                self.df.loc[:, i] = self.df.loc[:, i].fillna(self.num_how)
            return self.df

        elif self.num_how == 'mode':
            for i in self.num_cols:
                print("Imputing {0} with mode: {1}".format(i, self.df.loc[:, i].mode()[0]))
                self.df.loc[:, i] = self.df.loc[:, i].fillna(self.df.loc[:, i].mode()[0])
            return self.df

        else:
            return "Imputing cannot be completed"

    def impute_missing_categorical_values(self):
        """
        Impute missing categorical values.
        cat_how: Options are 'mode' or a specified numeric or string value
        """
        if self.cat_how == 'mode':
            for i in self.cat_cols:
                print('Imputing {0} with mode: {1}'.format(i, self.df.loc[:, i].mode()[0]))
                self.df.loc[:, i] = self.df.loc[:, i].fillna(self.df.loc[:, i].mode()[0])
            return self.df

        elif type(self.cat_how) == str:
            for i in self.cat_cols:
                print("Imputing {0} with value: {1}".format(i, self.cat_how))
                self.df.loc[:, i] = self.df.loc[:, i].fillna(self.cat_how)
            return self.df

        elif type(self.cat_how) == int or type(self.cat_how) == float:
            for i in self.cat_cols:
                print("Imputing {0} with value: {1}".format(i, self.cat_how))
                self.df.loc[:, i] = self.df.loc[:, i].fillna(str(self.cat_how))
            return self.df

        else:
            return "Imputing cannot be completed"

    def clean_csv_data(self):
        """
        1) load the file(s).
        2) Get basename of file or directory.
        3) Define QC report name for raw input data (i.e., not preprocessed).
        4) Create QC HTML report of the raw data.
        5) Remove duplicate rows.
        6) Impute numeric columns with one of the following options: 'mean', 'median', 'mode' or
         a specified numeric value.
        7) Impute categorical columns with one of the following options: 'mode' or a specified numeric or string value.
        8) Remove unwanted columns.
        9) Convert strings to lowercase.
        10) Convert column names to lowercase and change spaces to underscores.
        11) Reset the dataframe index.
        12) Create QC HTML report of the preprocessed data.
        13) Create output CSV file containing preprocessed data.
        14) Return pandas dataframe containing preprocessed data.
        """
        self.df = self.load_data()
        name = self.get_base_name()
        raw_qc_name = name + '_qc_raw'
        self.create_qc_report(raw_qc_name)
        self.df = self.remove_duplicate_rows()
        self.df = self.impute_missing_numeric_values()
        self.df = self.impute_missing_categorical_values()
        self.df = self.drop_unwanted_columns()
        self.df = self.df_str_to_lowercase()
        self.df.columns = self.df.columns.str.replace(' ', '_')
        self.df.reset_index(drop=True, inplace=True)
        clean_qc_name = name + '_qc_clean'
        self.create_qc_report(clean_qc_name)
        out_file = self.out_path + '/' + self.timestr + '_' + name + '_clean.csv'
        self.df.to_csv(out_file, index=False)
        return self.df


class Encoders:

    def __init__(self, df, label_col):
        self.df = df
        self.label_col = label_col

    def one_hot(self):
        """ Apply One-Hot encoding to all categorical columns in a dataframe except the label column. """
        features = self.df.loc[:, self.df.columns != self.label_col].select_dtypes(include=['object']).columns.to_list()
        for feat in features:
            self.df[feat] = pd.Categorical(self.df[feat])
            df_dummies = pd.get_dummies(self.df[feat], prefix=feat, drop_first=True)
            self.df.drop(feat, axis=1, inplace=True)
            self.df = pd.concat([self.df, df_dummies], axis=1)
        return self.df

    def binary_encoder(self, features):
        """
        Encode binary categorical values in specific columns with the LabelEncoder from sklearn.
        """
        le = LabelEncoder()
        self.df[features] = self.df[features].apply(le.fit_transform)
        return self.df
