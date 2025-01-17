import os

DATA_DIR = "data/"
TEAMLOG_DATA = os.path.join(DATA_DIR, "team_data")
PLAYERLOG_DATA = os.path.join(DATA_DIR, "player_data")
INJURY_DATA = os.path.join(DATA_DIR, "injury_data")
PLOTS_DIR = "plots/"
SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
DEFAULT_COLUMNS = ['SEASON_YEAR', 'TEAM_ID', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'WL', 'MATCHUP']
# NUMERIC_COLUMNS = ['PTS', 'PLUS_MINUS', 'FG_PCT', 'FGM', 'OREB', 'DREB', 'AST', 'BLK']
NUMERIC_COLUMNS = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB',
                   'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']

WIN_PCT_COLUMN = 'WIN_PCT'
AGG_WINDOW_SIZES = [5, 15]
