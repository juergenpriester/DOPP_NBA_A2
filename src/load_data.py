import numpy as np
import pandas as pd
import logging as log
import os
from nba_api.stats.endpoints import TeamGameLogs, PlayerGameLogs
import kagglehub

from utils import check_create_dir
from constants import DATA_DIR, SEASONS, TEAMLOG_DATA, PLAYERLOG_DATA, INJURY_DATA


log.basicConfig(level=log.INFO)


def load_injury_data() -> pd.DataFrame:
    # Download latest version
    path = kagglehub.dataset_download("jacquesoberweis/2016-2025-nba-injury-data")

    print("Path to dataset files:", path)
    df = pd.read_csv(os.path.join(path, "injury_data.csv"))
    df.to_csv(os.path.join(INJURY_DATA, "injury_data.csv"), index=False)


def load_playerlogs(seasons: list[str]) -> pd.DataFrame:
    df = pd.DataFrame()
    for season in seasons:
        gamedatapull = PlayerGameLogs(
            league_id_nullable="00",  # nba 00, g_league 20, wnba 10
            season_type_nullable="Regular Season",  # Regular Season, Playoffs, Pre Season
            season_nullable=season,
        )

        df_season = gamedatapull.get_data_frames()[0]
        df = pd.concat([df, df_season])
    return df


def load_teamlogs(seasons: list[str]) -> pd.DataFrame:
    df = pd.DataFrame()
    for season in seasons:
        gamedatapull = TeamGameLogs(
            league_id_nullable="00",  # nba 00, g_league 20, wnba 10
            season_type_nullable="Regular Season",  # Regular Season, Playoffs, Pre Season
            season_nullable=season,
        )

        df_season = gamedatapull.get_data_frames()[0]
        df = pd.concat([df, df_season])
    return df


def main():
    load_injury_data()

    check_create_dir(DATA_DIR)
    check_create_dir(TEAMLOG_DATA)
    team_load_data = load_teamlogs(SEASONS)
    team_load_data.to_csv(os.path.join(TEAMLOG_DATA, "team_data.csv"), index=False)
    log.info("Team data loaded and saved to nba_data.csv")

    check_create_dir(PLAYERLOG_DATA)
    player_log_data = load_playerlogs(SEASONS)
    player_log_data.to_csv(os.path.join(PLAYERLOG_DATA, "player_data.csv"), index=False)
    log.info("Player data loaded and saved to nba_player_data.csv")


if __name__ == "__main__":
    main()
