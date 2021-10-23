#!/usr/bin/python3
"""
This module allows a simple and quick comparison of classifiers with binary label categories.

Example usage:
    $ python3 <path>/binary_classifiers.py --df <path to input data> --label_col <binary label column> --out_dir <output directory>
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import argparse
import glob
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
import time
import warnings

warnings.filterwarnings("ignore")
timestr = time.strftime('%Y%m%d')


def import_latest_encoded_df(data_dir: str) -> pd.DataFrame:
    """
    Import the latest encoded CSV file with the extension '_encoded.csv' from a specified data directory.
    """
    files = glob.glob(f"{data_dir}/*_encoded.csv")
    df_path = max(files, key=os.path.getmtime)
    df = pd.read_csv(df_path)
    return df


def box_plot(data: pd.DataFrame, plot_title: str, out_dir: str, out_name: str):
    """
    Create boxplot with model performance metrics or time metrics.
    """
    sns.set_theme(style='white')
    fig, ax = plt.subplots(figsize=(10, 6))
    fliers = dict(marker='o', markersize=3, markerfacecolor='white', linestyle='none', markeredgecolor='grey')
    sns.boxplot(ax=ax, x='values', y='model', hue='metrics', data=data, palette='bright',
                linewidth=0.5, flierprops=fliers)
    ax.grid(b=True, which='both', axis='both', color='lightgrey', linestyle='--')
    for spine in ['top', 'bottom', 'right', 'left']:
        ax.spines[spine].set_alpha(0.2)
    plt.title(plot_title, pad=20)
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='lower center', ncol=5)
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.savefig(f"{out_dir}/{timestr}_{out_name}.png", dpi=300, bbox_inches='tight')
    return plt.show()


def compare_binary_classifiers(data_dir: str, label_col: str, out_dir: str) -> pd.DataFrame:
    """
    Simple and quick comparison of classifiers with binary label categories.
    The following classifiers will be compared for a input dataset and label:
    Logistic regression (LogReg), random forest classifier (RF), K-Nearest Neighbors classifier (KNN),
    Support Vector Machines (SVM), Gaussian Naive Bayes algorithm (GNB), Decision Tree (DecTree),
    MLP Classifier (MLP), Gradient Boosting Classifier (GBC), Ada Boost Classifier (Ada),
    BaggingClassifier (Bagging), and XGBClassifier (XGB).
    """
    df = import_latest_encoded_df(data_dir)

    labels = pd.unique(df[label_col]).tolist()
    list_string = map(str, labels)
    labels = list(list_string)

    x = df.drop(label_col, axis=1)
    y = df[label_col]

    scaler = StandardScaler()
    X = scaler.fit_transform(x)

    stratified_shuffle = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    for train_index, test_index in stratified_shuffle.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    dfs = []
    results = []
    names = []

    models = [
        ('LogReg', LogisticRegression()),
        ('RF', RandomForestClassifier()),
        ('KNN', KNeighborsClassifier()),
        ('SVM', SVC()),
        ('GNB', GaussianNB()),
        ('DecTree', DecisionTreeClassifier()),
        ('MLP', MLPClassifier()),
        ('GBC', GradientBoostingClassifier()),
        ('Ada', AdaBoostClassifier()),
        ('Bagging', BaggingClassifier()),
        ('XGB', XGBClassifier(eval_metric='logloss'))
    ]

    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']

    for name, model in models:
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)
        print(classification_report(y_test, y_pred, target_names=labels))

        results.append(cv_results)
        names.append(name)

        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)

    final = pd.concat(dfs, ignore_index=True)

    # metrics
    bootstraps = []

    for model in list(set(final.model.values)):
        model_df = final.loc[final.model == model]
        bootstrap = model_df.sample(n=30, replace=True)
        bootstraps.append(bootstrap)

    bootstrap_df = pd.concat(bootstraps, ignore_index=True)
    results_long = pd.melt(bootstrap_df, id_vars=['model'], var_name='metrics', value_name='values')
    time_metrics = ['fit_time', 'score_time']  # fit time metrics

    # performance metrics
    performance = results_long.loc[~results_long['metrics'].isin(time_metrics)]  # get df without fit data
    performance = performance.sort_values(by='values')

    # time metrics
    time_met = results_long.loc[results_long['metrics'].isin(time_metrics)]  # df with fit data
    time_met = time_met.sort_values(by='values')

    # box plots
    box_plot(performance, 'Comparison of Models by Classification Metrics', out_dir, "performance_metrics")
    box_plot(time_met, 'Comparison of Models by Classification Metrics', out_dir, "time_metrics")

    # extended metrics info
    perf_metrics = list(set(performance.metrics.values))
    ext_perf_metrics = bootstrap_df.groupby(['model'])[perf_metrics].agg([np.std, np.mean])

    time_metrics =  list(set(time_met.metrics.values))
    ext_time_met = bootstrap_df.groupby(['model'])[time_metrics].agg([np.std, np.mean])

    return final, bootstrap_df, performance, time_met, ext_perf_metrics, ext_time_met


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--df', help='Input directory.')
    parser.add_argument('--label_col', help='Column with target labels.')
    parser.add_argument('--out_dir', help='Output directory.')
    args = parser.parse_args()
    compare_binary_classifiers(args.df, args.label_col, args.out_dir)
