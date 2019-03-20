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
    lgb.plot_importance(lgb_model, max_num_features=max_features, height=0.8, ax=ax)
    ax.grid(False)
    plt.title("LightGBM - Feature Importance", fontsize=20)
    plt.show()
