import numpy as np
import pandas as pd
import logging as log

from utils import check_create_dir, load_from_csv

from constants import DATA_DIR, SEASONS, DEFAULT_COLUMNS, NUMERIC_COLUMNS, AGG_WINDOW_SIZE

log.basicConfig(level=log.INFO)


def create_team_mapping(df: pd.DataFrame):
    pass


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df['SEASON_YEAR'] = df['SEASON_YEAR'].str[:4].astype(int)
    df['TEAM_NAME'] = df['TEAM_NAME'].astype('string')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['HOME'] = df['MATCHUP'].str.contains(r'\bvs\b').astype(int)
    df.drop(columns=['MATCHUP'], inplace=True)
    df['WL'] = df['WL'].map({'W': 1, 'L': 0}).astype(int)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            if (df[col].dtype == 'int64' or df[col].dtype == 'float64') and col in df.columns:
                df[col] = df[col].astype(float)
    return df


def aggregate_team_stats(df: pd.DataFrame, window_size) -> pd.DataFrame:
    # sort by date
    df = df.sort_values(by=['GAME_DATE'])

    # group by team and season
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df.groupby(['TEAM_ID', 'SEASON_YEAR'])[col].transform(
                lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())

    # add number of wins

    # Drop first window_size rows for each team
    df = df.groupby(['TEAM_ID', 'SEASON_YEAR']).apply(
        lambda x: x.iloc[window_size:],
        include_groups=False
    ).reset_index(['TEAM_ID', 'SEASON_YEAR'], drop=False)

    return df


def combine_on_gameid(df) -> pd.DataFrame:
    if 'TEAM_NAME' in df.columns:
        df = df.drop(columns=['TEAM_NAME'], inplace=False)

    df_HOME = df[df['HOME'] == 1]
    df_AWAY = df[df['HOME'] == 0]
    df_HOME = df_HOME.drop(columns=['HOME'], inplace=False)
    df_AWAY = df_AWAY.drop(columns=['HOME', 'SEASON_YEAR', 'GAME_DATE', 'WL'], inplace=False)

    df_merged = df_HOME.merge(df_AWAY, on='GAME_ID', suffixes=('_HOME', '_AWAY'))

    df_merged.set_index(['SEASON_YEAR', 'GAME_ID', 'GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY'], inplace=True)

    return df_merged


def main():
    log.info("Loading data from csv")
    df = load_from_csv('data/nba_data.csv')
    log.info(df.head())

    log.info("GP_Rank")
    log.info(df['GP_RANK'])

    COLUMNS = DEFAULT_COLUMNS + NUMERIC_COLUMNS
    df = df[COLUMNS]
    log.info(df.head())
    log.info("Checking for missing values")
    missing_values = df.isnull().sum()
    log.info(missing_values)

    log.info("Data types before conversion")
    log.info(df.dtypes)
    df = convert_dtypes(df)
    df.sort_values(by=['GAME_DATE', 'GAME_ID'], inplace=True)
    log.info("Data types after conversion")
    log.info(df.dtypes)

    df.to_csv(DATA_DIR + 'nba_data_coverted.csv', index=False)

    df = aggregate_team_stats(df, AGG_WINDOW_SIZE)

    df.to_csv(DATA_DIR + 'nba_data_aggregated.csv', index=False)

    df = combine_on_gameid(df)
    log.info(df.head())

    df.to_csv(DATA_DIR + 'nba_data_combined.csv', index=True)


if __name__ == '__main__':
    main()
