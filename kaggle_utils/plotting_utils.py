# Methods for plotting

import numpy as np
import pandas as pd
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt


def plot_countplot(data, feature, dimension):
    """
    Plot counts for categorical features with small
    number of categories.

    :param data: pandas dataframe
    :param feature: string, feature to plot
    :param dimension: dimension to group counts
    """
    plt.figure(figsize=(10, 5))
    sns.countplot(x=feature, hue=dimension, data=data)
    plt.title(feature)
    plt.show()


def plot_hist(data, cols_to_plot, xlabel, ylabel, title,
              min_value, max_value, num_bins=50, percs=None):
    """
    Plot hist with percentile vertical lines
    """
    bins_ = np.linspace(min_value, max_value, num_bins)

    plt.figure(figsize=(12, 5))
    for col in cols_to_plot:
        # plot histograms
        plt.hist(data[col], bins=bins_, alpha=0.5, label=col)
        # plot percentile lines
        if percs is not None:
            # get colors
            colors = plt.cm.tab10.colors[:len(percs)]
            percs_values = {p: {'value': np.percentile(data[col], p),
                                'color': c}
                            for (p, c) in zip(percs, colors)}
            for p in percs_values:
                plt.axvline(x=percs_values[p]['value'],
                            label=f"{p} percentile={round(percs_values[p]['value'], 1)}",
                            alpha=0.5, linestyle='--', linewidth=2,
                            color=percs_values[p]['color'])

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(which='major', linestyle='--', linewidth=1, alpha=0.3)
    plt.legend()
    plt.show()


def plot_2_hist(data, feature, num_bins):
    """
    Plot 2 hists of 1 feature on 1 condition

    :param data: pandas dataframe
    :param feature: string, feature to plot
    :param num_bins: number of bins to plot on histogram
    """
    bins_ = np.linspace(data[feature].min(), data[feature].max(), num_bins)

    plt.figure(figsize=(10, 5))
    plt.hist(data[data['churn'] == 0][feature], bins=bins_, alpha=0.5, label='churn = 0')
    plt.hist(data[data['churn'] == 1][feature], bins=bins_, alpha=0.5, label='churn = 1')
    plt.title(feature + ' distribution')
    plt.legend()
    plt.show()


def plot_lgb_feature_importance(lgb_model, max_features):
    """
    Method to plot LightGBM feature importance

    :param lgb_model: LightGBM trained model
    :param  max_features: number of features to plot
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    lgb.plot_importance(lgb_model, max_num_features=max_features, height=0.8, ax=ax, importance_type='gain')
    ax.grid(False)
    plt.title("LightGBM - Feature Importance", fontsize=20)
    plt.show()


def plot_box_plot(data, x, y, title, ylabel, xlabel):
    """
    Method for plotting seaborn boxplot
    :param data: pandas DataFrame
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(x=x, y=y, data=data)
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(which='major', linestyle='--', linewidth=1, alpha=0.3)
    plt.show()
