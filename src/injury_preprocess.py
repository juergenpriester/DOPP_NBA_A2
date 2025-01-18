import pandas as pd
import os
from utils import load_from_csv
from constants import INJURY_DATA, PLAYERLOG_DATA


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

    result_df = result_df.sort_values(["Player", "Injury_Start"])
    result_df = result_df.reset_index(drop=True)

    players = list(set([entry["Player"] for entry in result]))
    return players, result_df


def preprocess_advanced_stats():
    df_advanced = load_from_csv(os.path.join(PLAYERLOG_DATA, "player_advanced.csv"))
    df_advanced.drop(columns=["Rk", "AS", "Pos"], inplace=True)
    df_advanced["Season"] = df_advanced["Season"].apply(
        lambda x: int(x.split("-")[1]) + 2000
    )
    df_advanced["Team"] = df_advanced["Team"].astype(str)
    df_advanced["Player"] = df_advanced["Player"].astype(str)
    return df_advanced


if __name__ == "__main__":
    players, result = preprocess_injury_data()

    ##significant injuries missing: Chet Holmgren, Luka Doncic, Franz Wagner
    player_name = "Nikola Jokic"  # Replace with the name you want to search for
    player_injury = result[result["Player"] == player_name]

    df = preprocess_advanced_stats()
    player_name = "Nikola Jokić"  # Replace with the name you want to search for
    player_data = df[df["Player"] == player_name]
    player_df = player_data[["Player", "Season", "WS/48", "BPM"]]
