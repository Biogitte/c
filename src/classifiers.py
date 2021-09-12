#!/usr/bin/python3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")


class Classification:

    def __init__(self, df, label_col):
        self.df = df
        self.label_col = label_col

        labels = pd.unique(self.df[self.label_col]).tolist()
        list_string = map(str, labels)
        self.labels = list(list_string)

        self.x = self.df.drop(self.label_col, axis=1)
        self.y = self.df[self.label_col]

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.x)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.3, random_state=42)

    def compare_binary_clfs(self):
        """
        Simple and quick comparison of classifiers with binary label categories.
        The following classifiers will be compared for a input dataset and label:
        Logistic regression (LogReg), random forest classifier (RF), K-Nearest Neighbors classifier (KNN),
        Support Vector Machines (SVM), and Gaussian Naive Bayes algorithm (GNB).
        """

        dfs = []
        results = []
        names = []

        models = [
            ('LogReg', LogisticRegression()),
            ('RF', RandomForestClassifier()),
            ('KNN', KNeighborsClassifier()),
            ('SVM', SVC()),
            ('GNB', GaussianNB())
        ]

        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']

        for name, model in models:
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = model_selection.cross_validate(model, self.X_train, self.y_train, cv=kfold, scoring=scoring)
            clf = model.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            print(name)
            print(classification_report(self.y_test, y_pred, target_names=self.labels))

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

        return final, bootstrap_df, performance, time_met


class ClassificationPlotter:

    def __init__(self, df, label_col):
        self.df = df
        self.label_col = label_col

        clf_obj = Classification(self.df, self.label_col)
        self.final, self.bootstrap_df, self.performance, self.time_met = clf_obj.compare_binary_clfs()

        self.dims = (10, 8)
        self.fig, self.ax = plt.subplots(figsize=self.dims)

        sns.set_style('ticks')
        sns.set(font_scale=1.2)

    def performance_plot(self):
        sns.boxplot(ax=self.ax, x="model", y="values", hue="metrics", data=self.performance, palette="Set3")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
        plt.title('Comparison of Models by Classification Metrics')
        # TODO:
        # if save_img==True:
        # plt.savefig('path/timestr_benchmark_models_performance.png', dpi=300)
        # etc.

    def time_metrics_plot(self):
        # FIXME
        sns.boxplot(ax=self.ax, x="model", y="values", hue="metrics", data=self.time_met, palette="Set3")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
        plt.title('Comparison of Model by Fit and Score Time')
        # TODO:
        # if save_img==True:
        # plt.savefig('path/timestr_benchmark_models_time.png',dpi=300)
        # etc.

    def extended_performance_metrics(self):
        """ Get the extended performance metrics info. """
        metrics = list(set(self.performance.metrics.values))
        ext_df = self.bootstrap_df.groupby(['model'])[metrics].agg([np.std, np.mean])
        return ext_df

    def extended_time_metrics(self):
        """ Get the extended time metrics info. """
        metrics_time = list(set(self.time_met.metrics.values))
        ext_df = self.bootstrap_df.groupby(['model'])[metrics_time].agg([np.std, np.mean])
        return ext_df
