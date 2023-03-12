import pdb
import numpy as np
import pandas as pd

data = [
    {
        "last_updated_ts": "2022-10-05T11:15:00Z",
        "entity_id": "input_number.humidity",
        "state": 40.2,
    },
    {
        "last_updated_ts": "2022-10-05T12:00:00Z",
        "entity_id": "input_number.temperature",
        "state": 23.7,
    },
    {
        "last_updated_ts": "2022-10-05T12:30:00Z",
        "entity_id": "input_number.temperature",
        "state": 25.3,
    },
    {
        "last_updated_ts": "2022-10-05T13:00:00Z",
        "entity_id": "input_number.humidity",
        "state": 41.5,
    },
    {
        "last_updated_ts": "2022-10-05T13:45:00Z",
        "entity_id": "input_number.humidity",
        "state": 39.7,
    },
    {
        "last_updated_ts": "2022-10-05T15:00:00Z",
        "entity_id": "input_number.humidity",
        "state": 38.9,
    },
    {
        "last_updated_ts": "2022-10-05T16:30:00Z",
        "entity_id": "input_number.temperature",
        "state": 26.1,
    },
]

data = {
    "input_number.temperature": [
        {
            "last_updated_ts": "2022-10-05T12:00:00Z",
            "state": 23.7,
        },
        {
            "last_updated_ts": "2022-10-05T12:30:00Z",
            "state": 25.3,
        },
        {
            "last_updated_ts": "2022-10-05T16:30:00Z",
            "state": 26.1,
        },
    ],
    "input_number.humidity": [
        {
            "last_updated_ts": "2022-10-05T11:15:00Z",
            "state": 40.2,
        },
        {
            "last_updated_ts": "2022-10-05T13:00:00Z",
            "state": 41.5,
        },
        {
            "last_updated_ts": "2022-10-05T13:45:00Z",
            "state": 39.7,
        },
        {
            "last_updated_ts": "2022-10-05T15:00:00Z",
            "state": 38.9,
        },
    ],
}

# data = {
#     "input_number.temperature": [
#         <state input_number.temperature=0.0; initial=None, unit_of_measurement=Fahrenheit, friendly_name=temperature @ 2023-03-06T14:28:24.528154-08:00>,
#         <state input_number.temperature=68.3; initial=None, unit_of_measurement=Fahrenheit, friendly_name=temperature @ 2023-03-06T14:29:12.802015-08:00>
# }


# Convert the data dictionary to a pandas DataFrame
df = pd.concat([pd.DataFrame(v).assign(entity_id=k) for k, v in data.items()])
pdb.set_trace()
# Convert the last_updated_ts column to a datetime type
df["last_updated_ts"] = pd.to_datetime(df["last_updated_ts"])

# Set the last_updated_ts column as the index
df.set_index("last_updated_ts", inplace=True)

# Combine the temperature and humidity measurements into a single DataFrame
combined_df = df.pivot(columns="entity_id", values="state")

# Forward fill the missing values
combined_df.fillna(method="ffill", inplace=True)

# Drop any rows that still contain missing values
combined_df.dropna(inplace=True)

# Convert the index back to a column and rename it
combined_df.reset_index(inplace=True)
combined_df.rename(columns={"last_updated_ts": "timestamp"}, inplace=True)

# Convert the DataFrame to a numpy array
data_array = combined_df.values

# Print the resulting numpy array
print(data_array)
