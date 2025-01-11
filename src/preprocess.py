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


def combine_on_gameid(df) -> pd.DataFrame:
    df_home = df[df['HOME'] == 1]
    df_away = df[df['HOME'] == 0]
    df_home = df_home.drop(columns=['HOME'], inplace=False)
    df_away = df_away.drop(columns=['HOME', 'SEASON_YEAR', 'GAME_DATE'], inplace=False)

    df = df_home.merge(df_away, on='GAME_ID', suffixes=('_home', '_away'))
    df.set_index(['GAME_ID', 'GAME_DATE'], inplace=True)

    return df


def aggregate_team_stats(df: pd.DataFrame, window_size) -> pd.DataFrame:
    # sort by date
    df = df.sort_values(by=['GAME_DATE'])

    # group by team and season
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df.groupby(['TEAM_ID', 'TEAM_NAME', 'SEASON_YEAR'])[col].transform(
                lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())

    log.info(df.head())

    # Drop first window_size rows for each team
    df = df.groupby(['TEAM_ID', 'TEAM_NAME', 'SEASON_YEAR']).apply(
        lambda x: x.iloc[window_size:]
    ).reset_index(drop=True)

    return df


def main():
    log.info("Loading data from csv")
    df = load_from_csv('data/nba_data.csv')
    log.info(df.head())

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
    log.info(df.head())

    df.to_csv(DATA_DIR + 'nba_data_coverted.csv', index=False)

    # df = combine_on_gameid(df)
    # log.info(df.head())

    df = aggregate_team_stats(df, AGG_WINDOW_SIZE)

    log.info(df.head())

    df.to_csv(DATA_DIR + 'nba_data_cleaned.csv', index=False)


if __name__ == '__main__':
    main()