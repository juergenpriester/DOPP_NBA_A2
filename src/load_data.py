import numpy as np
import pandas as pd
from nba_api.stats.endpoints import TeamGameLogs

from utils import check_create_dir
from constants import DATA_DIR, SEASONS


def load_from_api(seasons: list[str]) -> pd.DataFrame:
    df = pd.DataFrame()
    for season in seasons:
        gamedatapull = TeamGameLogs(
            league_id_nullable='00',  # nba 00, g_league 20, wnba 10
            season_type_nullable='Regular Season',  # Regular Season, Playoffs, Pre Season
            season_nullable=season
        )

        df_season = gamedatapull.get_data_frames()[0]
        df = pd.concat([df, df_season])
    return df


def main():
    check_create_dir(DATA_DIR)
    api_data = load_from_api(SEASONS)
    api_data.to_csv(DATA_DIR + 'nba_data.csv', index=False)


if __name__ == '__main__':
    main()
