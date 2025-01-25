import pandas as pd
import os
from constants import INJURY_DATA


def preprocess_injury_data():
    # Read the CSV without filtering quotes initially
    df_injury = pd.read_csv(os.path.join(INJURY_DATA, "injury_data.csv"))

    df_injury["Date"] = pd.to_datetime(
        df_injury["Date"], format="%Y-%m-%d", errors="coerce"
    )

    players = []
    if "Relinquished" in df_injury.columns:
        players = df_injury["Relinquished"].dropna().unique().tolist()

    injury_start = df_injury[df_injury["Relinquished"].notna()][
        ["Date", "Team", "Relinquished", "Notes"]
    ].rename(columns={"Date": "Injury_Start", "Relinquished": "Player"})

    injury_end = df_injury[df_injury["Acquired"].notna()][
        ["Date", "Team", "Acquired"]
    ].rename(columns={"Date": "Injury_End", "Acquired": "Player"})

    result = []
    for player, group in injury_start.groupby("Player"):
        injury_group = group.sort_values(by="Injury_Start")
        recovery_group = injury_end[injury_end["Player"] == player].sort_values(
            by="Injury_End"
        )
        recovery_dates = iter(recovery_group["Injury_End"])

        current_recovery = None
        try:
            current_recovery = next(recovery_dates)
        except StopIteration:
            pass

        for i, (index, injury_row) in enumerate(injury_group.iterrows()):
            injury_note = injury_row["Notes"] if pd.notna(injury_row["Notes"]) else ""
            current_injury = injury_row["Injury_Start"]

            # Calculate season-ending date if applicable
            injury_end_date = pd.NaT
            if "(out for season)" in injury_note.lower():
                current_year = current_injury.year
                august_15_current_year = pd.Timestamp(f"{current_year}-08-15")

                if current_injury > august_15_current_year:
                    injury_end_date = pd.Timestamp(f"{current_year + 1}-08-15")
                else:
                    injury_end_date = august_15_current_year

            if i + 1 < len(injury_group):
                next_injury = injury_group.iloc[i + 1]["Injury_Start"]
                if current_recovery and current_recovery > next_injury:
                    result.append(
                        {
                            "Player": player,
                            "Team": injury_row["Team"],
                            "Injury_Start": current_injury,
                            "Injury_End": injury_end_date,
                            "Injury_Notes": injury_note,
                        }
                    )
                else:
                    end_date = (
                        injury_end_date
                        if not pd.isna(injury_end_date)
                        else current_recovery
                    )
                    result.append(
                        {
                            "Player": player,
                            "Team": injury_row["Team"],
                            "Injury_Start": current_injury,
                            "Injury_End": end_date,
                            "Injury_Notes": injury_note,
                        }
                    )
                    try:
                        current_recovery = next(recovery_dates)
                    except StopIteration:
                        current_recovery = None
            else:
                if current_recovery and current_recovery > current_injury:
                    end_date = current_recovery
                else:
                    end_date = (
                        injury_end_date if not current_recovery else current_recovery
                    )

                result.append(
                    {
                        "Player": player,
                        "Team": injury_row["Team"],
                        "Injury_Start": current_injury,
                        "Injury_End": end_date,
                        "Injury_Notes": injury_note,
                    }
                )

    # Convert to DataFrame and clean up
    result_df = pd.DataFrame(result)

    # Iterating backwards to combine injuries that are consecutive, and fixing those that are season ending without having knowledge of it at the time
    for i in range(len(result_df) - 1, -1, -1):
        # Check if current row has NaT in Injury_End
        if pd.isna(result_df.iloc[i]["Injury_End"]):
            current_injury_start = result_df.iloc[i]["Injury_Start"]
            current_player = result_df.iloc[i]["Player"]

            # Get the next August 15 after Injury_Start
            current_year = current_injury_start.year
            august_15_current_year = pd.Timestamp(f"{current_year}-08-15")

            if current_injury_start > august_15_current_year:
                default_injury_end = pd.Timestamp(f"{current_year + 1}-08-15")
            else:
                default_injury_end = august_15_current_year

            # Check if there's a row below (i < len(df) - 1) and if names match
            if (
                i < len(result_df) - 1
                and current_player == result_df.iloc[i + 1]["Player"]
            ):
                next_injury_start = result_df.iloc[i + 1]["Injury_Start"]
                next_august_15 = pd.Timestamp(f"{current_injury_start.year}-08-15")
                if current_injury_start > next_august_15:
                    next_august_15 = pd.Timestamp(
                        f"{current_injury_start.year + 1}-08-15"
                    )

                # Check if next injury starts before next August 15
                if next_injury_start <= next_august_15:
                    # Combine rows by taking end date from next row
                    result_df.at[i, "Injury_End"] = result_df.iloc[i + 1]["Injury_End"]
                    # Combine injury notes if they exist
                    if pd.notna(result_df.iloc[i + 1]["Injury_Notes"]):
                        result_df.at[i, "Injury_Notes"] = (
                            str(result_df.iloc[i]["Injury_Notes"])
                            + " ; "
                            + str(result_df.iloc[i + 1]["Injury_Notes"])
                        )
                    # Drop the next row as it's now combined
                    result_df = result_df.drop(index=result_df.index[i + 1])
                else:
                    # Set end date to next August 15
                    result_df.at[i, "Injury_End"] = default_injury_end
            else:
                # Set end date to next August 15
                result_df.at[i, "Injury_End"] = default_injury_end

    result_df["Team"] = result_df["Team"].astype(str)
    result_df["Player"] = result_df["Player"].astype(str).str.strip()
    result_df["Injury_Notes"] = result_df["Injury_Notes"].astype(str).str.strip()

    # Load mappings
    team_name_mapping = pd.read_json(os.path.join('mappings', 'team_name_mapping.json'))
    team_id_mapping = pd.read_json(os.path.join('mappings', 'team_id_mapping.json'))
    player_mapping = pd.read_json(os.path.join('mappings', 'player_mapping.json'))

    # Mapping Team Name to Team ID
    result_df = result_df.merge(team_name_mapping, left_on='Team', right_on='Team', how='left')
    result_df = result_df.drop(columns=['Team'], inplace=False)
    result_df.rename(columns={'TEAM_NAME': 'TEAM'}, inplace=True)
    result_df = result_df.merge(team_id_mapping, left_on='TEAM', right_on='TEAM_NAME', how='left')
    result_df = result_df.drop(columns=['TEAM', 'TEAM_NAME'], inplace=False)

    # from "Player" column remove string that are contained in brackets ()
    result_df["Player"] = result_df["Player"].str.replace(r"\(.*\)", "", regex=True)
    # Split a row where the name contatins "/" into two rows
    result_df = result_df.assign(
        Player=result_df["Player"].str.split("/")).explode("Player")
    result_df["Player"] = result_df["Player"].str.strip()

    # turn into datetime
    result_df["Injury_Start"] = pd.to_datetime(result_df["Injury_Start"])
    result_df["Injury_End"] = pd.to_datetime(result_df["Injury_End"])

    # rename columns into Caps
    result_df.columns = result_df.columns.str.upper()

    result_df = result_df.sort_values(["PLAYER", "INJURY_START"])
    result_df = result_df.reset_index(drop=True)

    # players = list(set([entry["PLAYER"] for entry in result]))

    result_df.to_csv(os.path.join(INJURY_DATA, "injury_data_cleaned.csv"), index=False)
