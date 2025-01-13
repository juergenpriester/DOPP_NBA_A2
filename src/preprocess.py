import numpy as np
import pandas as pd
import logging as log
import os
import kagglehub

from utils import check_create_dir, load_from_csv

from constants import DATA_DIR, TEAMLOG_DATA, PLAYERLOG_DATA, INJURY_DATA, DEFAULT_COLUMNS, NUMERIC_COLUMNS, AGG_WINDOW_SIZES, WIN_PCT_COLUMN

log.basicConfig(level=log.INFO)


def create_team_mapping(df: pd.DataFrame):
    pass


def convert_dtypes_teamlogs(df: pd.DataFrame, numeric_cols=None) -> pd.DataFrame:
    df = df.copy()
    df['SEASON_YEAR'] = df['SEASON_YEAR'].str[:4].astype(int)
    df['TEAM_NAME'] = df['TEAM_NAME'].astype('string')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['HOME'] = df['MATCHUP'].str.contains(r'\bvs\b').astype(int)
    df.drop(columns=['MATCHUP'], inplace=True)
    df['WL'] = df['WL'].map({'W': 1, 'L': 0}).astype(int)
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                if (df[col].dtype == 'int64' or df[col].dtype == 'float64') and col in df.columns:
                    df[col] = df[col].astype(float)
        return df
    else:
        return df


def convert_dtypes_playerlogs(df: pd.DataFrame, numeric_cols=None) -> pd.DataFrame:
    df = df.copy()
    df = df[['SEASON_YEAR', 'PLAYER_ID', 'PLAYER_NAME', 'GAME_DATE', 'MIN', 'PTS', 'PLUS_MINUS']]
    df['SEASON_YEAR'] = df['SEASON_YEAR'].str[:4].astype(int)
    df['PLAYER_ID'] = df['PLAYER_ID'].astype(int)
    df['PLAYER_NAME'] = df['PLAYER_NAME'].astype('string')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['MIN'] = df['MIN'].astype(float)
    df['PTS'] = df['PTS'].astype(float)
    df['PLUS_MINUS'] = df['PLUS_MINUS'].astype(float)
    return df


def compute_rolling_values(df: pd.DataFrame,
                           window_size: int,
                           agg_function=lambda x: np.mean(x),
                           group: list[str] = ['TEAM_ID', 'SEASON_YEAR']) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            colname = col + '_AGG_' + str(window_size)
            df[colname] = df.groupby(group)[col].transform(
                lambda x: x.shift(1).rolling(window=window_size, min_periods=1).agg(agg_function))
    return df


def aggregate_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    # sort by date
    df = df.sort_values(by=['GAME_DATE'])

    for window_size in AGG_WINDOW_SIZES:
        df = compute_rolling_values(df, window_size, agg_function=lambda x: np.mean(x))

    df = df.drop(columns=NUMERIC_COLUMNS, inplace=False)
    # Calculate season long win percentage
    df[WIN_PCT_COLUMN] = df.groupby(['TEAM_ID', 'SEASON_YEAR'])['WL'].transform(
        lambda x: x.shift(1).rolling(window=9999, min_periods=1).mean())

    # Drop first window_size rows for each team
    df = df.groupby(['TEAM_ID', 'SEASON_YEAR']).apply(
        lambda x: x.iloc[window_size:],
        include_groups=False
    ).reset_index(['TEAM_ID', 'SEASON_YEAR'], drop=False)

    return df


def aggregate_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=['GAME_DATE', 'PLAYER_NAME'])
    df = compute_rolling_values(df, 9999, agg_function=lambda x: np.mean(x), group=['PLAYER_ID', 'SEASON_YEAR'])

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


def preprocess_teamlogs():
    log.info("Loading data from csv")
    df = load_from_csv(os.path.join(TEAMLOG_DATA, "team_data.csv"))
    log.info(df.head())

    log.info(df['TEAM_NAME'].unique())

    # drop variables that end with "_RANK"
    df = df.loc[:, ~df.columns.str.endswith('_RANK')]
    df = df.drop(columns=['AVAILABLE_FLAG'], inplace=False)

    # create data with all columns converted to appropriate data types
    df_full_converted = convert_dtypes_teamlogs(df)
    df_full_converted.to_csv(os.path.join(TEAMLOG_DATA, 'team_data_full_converted.csv'), index=False)

    COLUMNS = DEFAULT_COLUMNS + NUMERIC_COLUMNS
    df = df[COLUMNS]

    log.info("Checking for missing values")
    missing_values = df.isnull().sum()
    log.info(missing_values)

    df = convert_dtypes_teamlogs(df, NUMERIC_COLUMNS)
    df.sort_values(by=['GAME_DATE', 'GAME_ID'], inplace=True)
    df.to_csv(os.path.join(TEAMLOG_DATA, 'team_data_converted.csv'), index=False)

    df = aggregate_team_stats(df)
    df.to_csv(os.path.join(TEAMLOG_DATA, 'team_data_aggregated.csv'), index=False)

    df = combine_on_gameid(df)
    df.to_csv(os.path.join(TEAMLOG_DATA, 'team_data_combined.csv'), index=True)


def preprocess_player_logs():
    log.info("Loading data from csv")
    df = load_from_csv(os.path.join(PLAYERLOG_DATA, "player_data.csv"))
    log.info(df.head())
    create_player_mapping(df)

    # drop variables that end with "_RANK"
    df = convert_dtypes_playerlogs(df)

    # create data with all columns converted to appropriate data types
    df.to_csv(os.path.join(PLAYERLOG_DATA, 'player_data_converted.csv'), index=False)

    df = aggregate_player_stats(df)
    df.to_csv(os.path.join(PLAYERLOG_DATA, 'player_data_aggregated.csv'), index=False)


def create_player_mapping(df: pd.DataFrame):
    # create json file containing player name to player id mapping
    player_mapping = df[['PLAYER_NAME', 'PLAYER_ID']].drop_duplicates()
    player_mapping.to_json(os.path.join(DATA_DIR, 'player_mapping.json'), orient='records')


def preprocess_injury_data():
    df = pd.read_csv(os.path.join(INJURY_DATA, "injury_data.csv"), index_col="Unnamed: 0")
    df['Date'] = pd.to_datetime(df['Date'])
    # keep dates greater than 2016-10-01
    df = df.loc[df['Date'] >= '2016-10-01']
    # add player_id column from mapping
    log.info(df.head())

    team_mapping = pd.read_json(os.path.join(DATA_DIR, 'team_mapping.json'))
    player_mapping = pd.read_json(os.path.join(DATA_DIR, 'player_mapping.json'))

    log.info(df.shape)
    df = df.merge(team_mapping, left_on='Team', right_on='Team', how='left')
    df = df.drop(columns=['Team'], inplace=False)
    log.info(df.shape)

    for col in ['Relinquished', 'Acquired']:
        df = df.merge(player_mapping, left_on=col, right_on='PLAYER_NAME', how='left')
        df = df.drop(columns=[col, 'PLAYER_NAME'], inplace=False)
        df.rename(columns={'PLAYER_ID': col + '_ID'}, inplace=True)
        log.info(df.shape)
    log.info(df.head())

    df.to_csv(os.path.join(INJURY_DATA, 'injury_data_converted.csv'), index=False)


def main():
    check_create_dir(DATA_DIR)
    preprocess_teamlogs()
    preprocess_player_logs()
    preprocess_injury_data()


if __name__ == '__main__':
    main()
