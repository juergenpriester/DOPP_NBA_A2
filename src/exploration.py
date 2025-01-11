import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging as log

from utils import check_create_dir

log.basicConfig(level=log.INFO)


def plot_correlation_matrix(df: pd.DataFrame):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_columns]
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig("plots/correlation_matrix.png")


def plot_correlation_to_target(df: pd.DataFrame, target_column: str):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_columns]
    corr = df.corr()
    corr_target = corr[target_column].abs().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 10))
    corr_target.plot(kind='bar')
    plt.savefig("plots/correlation_to_target.png")


def main():
    check_create_dir('plots')
    df = pd.read_csv('data/nba_data_combined.csv')
    log.info("Percentage of home wins")
    log.info(np.round(df['WL'].mean(), 4))

    nba_data = pd.read_csv('data/nba_data_converted.csv')
    plot_correlation_matrix(nba_data)
    plot_correlation_to_target(nba_data, 'WL')


if __name__ == '__main__':
    main()
