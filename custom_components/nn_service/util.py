import numpy as np
import os
import logging
import math
from .data_fetcher import fetch_data
import pandas as pd


def merge_data(data):
    # Convert the data list into a pandas DataFrame
    # Convert the state objects to dictionaries and combine them into a single list
    data_list = []
    for entity_id, states in data.items():
        for state in states:
            state_dict = state.as_dict()
            state_dict["entity_id"] = entity_id
            data_list.append(state_dict)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data_list)

    # Convert the last_updated column to a datetime type
    df["last_updated"] = pd.to_datetime(df["last_updated"])

    # Set the last_updated column as the index
    df.set_index("last_updated", inplace=True)

    # Combine the temperature and humidity measurements into a single DataFrame
    combined_df = df.pivot(columns="entity_id", values="state")

    # Forward fill the missing values
    combined_df.fillna(method="ffill", inplace=True)

    combined_df = combined_df[
        ~combined_df.isin(["unavailable", "unknown", None, "", np.nan]).any(axis=1)
    ]

    # Convert the index back to a column and rename it
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={"last_updated": "timestamp"}, inplace=True)

    return combined_df
