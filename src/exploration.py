import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging as log
from functools import reduce


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


def factors(n):
    return set(reduce(
        list.__add__,
        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def get_subplot_sizes(n):
    factors_list = np.array(list(factors(8)))
    max_cols = 4
    factors_list = factors_list[factors_list <= max_cols]
    if len(factors_list) == 1:
        return factors_list[0], 1
    else:
        return factors_list[-1], n//factors_list[-1]


def plot_scatter(df: pd.DataFrame):
    df = df.copy()
    df = df[['WL', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB', 'TOV', 'STL', 'BLK', 'PLUS_MINUS']]

    num_columns = len(df.columns)
    figsize = (4 * num_columns, 4 * num_columns)  # Adjust figure size for clarity
    fig, axs = plt.subplots(num_columns, num_columns, figsize=figsize)

    # Plot scatter plots
    for i, col in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            ax = axs[j, i]
            df.plot.scatter(x=col, y=col2, ax=ax, alpha=0.5)
            ax.set_xlabel('' if j != num_columns - 1 else col, fontsize=40)  # Label only the bottom row
            ax.set_ylabel('' if i != 0 else col2, fontsize=40)  # Label only the left column
            ax.set_xticks([])
            ax.set_yticks([])

    # Add global labels for rows and columns
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    # Save the plot and close
    plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])  # Adjust layout to fit global labels
    plt.savefig(EXPLORE_DIR + "/scatter_matrix.png")
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
    plot_scatter(nba_data)


if __name__ == '__main__':
    main()
