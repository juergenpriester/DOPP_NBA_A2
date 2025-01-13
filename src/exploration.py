import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging as log

from utils import check_create_dir
from constants import DATA_DIR, PLOTS_DIR

EXPLORE_DIR = PLOTS_DIR + '/exploration'
check_create_dir(EXPLORE_DIR)


log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')


def plot_correlation_matrix(df: pd.DataFrame):
    df = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_columns]
    df = df.drop(columns=['SEASON_YEAR', 'TEAM_ID', 'GAME_ID'], inplace=False)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig(EXPLORE_DIR + "/correlation_matrix.png")
    plt.close()


def plot_correlation_to_target(df: pd.DataFrame, target_column: str):
    df = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_columns]
    df = df.drop(columns=['SEASON_YEAR', 'TEAM_ID', 'GAME_ID'], inplace=False)
    corr = df.corr()
    corr_target = corr[target_column].abs().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(20, 10))
    corr_target.plot(kind='bar')
    plt.title(f"Correlation to target variable: {target_column}")
    plt.xticks(range(len(corr_target)), corr_target.index, rotation='vertical')
    plt.xlabel('Feature')
    plt.ylabel('Correlation')
    plt.savefig(EXPLORE_DIR + "/correlation_to_target.png")
    plt.close()


def plot_wins_over_teams(df: pd.DataFrame):
    df = df.copy()
    wins = df.groupby(['TEAM_ABBREVIATION'])['WL'].mean()
    wins = wins.sort_values(ascending=False)
    wins.plot(kind='bar', stacked=True, figsize=(20, 10))
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title("Wins over teams")
    plt.xlabel('Team')
    plt.ylabel('Win percentage')
    plt.savefig(EXPLORE_DIR + "/wins_over_teams.png")
    plt.close()


def main():
    check_create_dir(PLOTS_DIR)
    df = pd.read_csv('data/nba_data_combined.csv')
    log.info("Percentage of home wins")
    log.info(np.round(df['WL'].mean(), 4))

    nba_data = pd.read_csv('data/nba_data_full_converted.csv')
    plot_correlation_matrix(nba_data)
    plot_correlation_to_target(nba_data, 'WL')
    plot_wins_over_teams(nba_data)


if __name__ == '__main__':
    main()
