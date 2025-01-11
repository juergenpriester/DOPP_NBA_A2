import numpy as np
import pandas as pd
from nba_api.stats.endpoints import TeamGameLogs

from utils import check_create_dir
from constants import DATA_DIR



def load_from_api():
    gamedatapull = TeamGameLogs(
            league_id_nullable ='00', # nba 00, g_league 20, wnba 10
            team_id_nullable = '', # can specify a specific team_id
#            season_nullable = '2023-24',
            season_type_nullable = 'Regular Season' # Regular Season, Playoffs, Pre Season
        )
        
    df_season = gamedatapull.get_data_frames()[0]     
    print(df_season.head())
    return df_season


def main():
    # check_create_dir('data')
    check_create_dir(DATA_DIR)
    api_data = load_from_api()
    api_data.to_csv(DATA_DIR + 'nba_data.csv', index=False)




if __name__ == '__main__':
    main()