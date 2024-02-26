# %%

from itertools import chain

import pandas as pd


def main(path: str = None):
    train = pd.read_csv("data/predict-energy-behavior-of-prosumers/train.csv")
    gas_df = pd.read_csv("data/predict-energy-behavior-of-prosumers/gas_prices.csv")
    electricity_df = pd.read_csv(
        "data/predict-energy-behavior-of-prosumers/electricity_prices.csv"
    )
    client_df = pd.read_csv("data/predict-energy-behavior-of-prosumers/client.csv")
    fw_df = pd.read_csv(
        "data/predict-energy-behavior-of-prosumers/forecast_weather.csv"
    )
    hw_df = pd.read_csv(
        "data/predict-energy-behavior-of-prosumers/historical_weather.csv"
    )
    # locations is customize data sets which are used by other coders in the competition
    # to help in
    # getting more consolidated merged data set to work on
    locations = pd.read_csv(
        "data/predict-energy-behavior-of-prosumers/county_lon_lats.csv"
    )

    data = train
    client = client_df
    hist_weather = hw_df
    forecast_weather = fw_df
    electricity = electricity_df
    gas = gas_df
    locations = locations

    counties = pd.read_json(
        "data/predict-energy-behavior-of-prosumers/county_id_to_name_map.json",
        orient="index",
    ).reset_index()
    counties.columns = ["county", "county_name"]

    # Dropping (target) nan values
    data = data[data["target"].notnull()]

    # Converting (datetime) column to datetime
    data["datetime"] = pd.to_datetime(data["datetime"], utc=True)

    # Renaming (forecast_date) to (datetime) for merging with the train data later
    electricity = electricity.rename(columns={"forecast_date": "datetime"})

    # Converting (datetime) column to datetime
    electricity["datetime"] = pd.to_datetime(electricity["datetime"], utc=True)

    # Decreasing (data_block_id) in client data because it's 2 steps ahead from train's
    # data (data_block_id)
    client["data_block_id"] -= 2

    """locations is a custom data that will help replace (latitude) and (longitude)
    columns by the counties for each coordination | you can find the data in Input """
    locations = locations.drop("Unnamed: 0", axis=1)

    # Rounding the (latitude) and (longitude) for 1 decimal fraction
    forecast_weather[["latitude", "longitude"]] = (
        forecast_weather[["latitude", "longitude"]].astype(float).round(1)
    )

    # Merging counties in locations data with the coordinations in the forecast_weather
    #  data
    forecast_weather = forecast_weather.merge(
        locations, how="left", on=["longitude", "latitude"]
    )

    # dropping nan values
    forecast_weather.dropna(axis=0, inplace=True)

    # Converting (county) column to integer
    forecast_weather["county"] = forecast_weather["county"].astype("int64")

    # Dropping the columns we won't need | We will use the (forecast_datetime) column
    # instead of the (origin_datetime)
    forecast_weather.drop(
        ["origin_datetime", "latitude", "longitude", "hours_ahead", "data_block_id"],
        axis=1,
        inplace=True,
    )

    # Renaming (forecast_datetime) to (datetime) for merging with the train data later
    forecast_weather.rename(columns={"forecast_datetime": "datetime"}, inplace=True)

    # Converting (datetime) column to datetime
    forecast_weather["datetime"] = pd.to_datetime(
        forecast_weather["datetime"], utc=True
    )

    """Grouping all forecast_weather columns mean values by hour, So each hour
    will have the mean values of the forecast_weather columns"""
    forecast_weather_datetime = (
        forecast_weather.groupby([forecast_weather["datetime"].dt.to_period("h")])[
            list(forecast_weather.drop(["county", "datetime"], axis=1).columns)
        ]
        .mean()
        .reset_index()
    )

    # After converting the (datetime) column to hour period for the groupby we convert
    # it back to datetime
    forecast_weather_datetime["datetime"] = pd.to_datetime(
        forecast_weather_datetime["datetime"].dt.to_timestamp(), utc=True
    )

    """Grouping all forecast_weather columns mean values by hour and county, So each
    hour and county will have the mean values of the forecast_weather columns for
    each county"""
    forecast_weather_datetime_county = (
        forecast_weather.groupby(
            ["county", forecast_weather["datetime"].dt.to_period("h")]
        )[list(forecast_weather.drop(["county", "datetime"], axis=1).columns)]
        .mean()
        .reset_index()
    )

    # After converting the (datetime) column to hour period for the groupby we convert
    # it  back to datetime
    forecast_weather_datetime_county["datetime"] = pd.to_datetime(
        forecast_weather_datetime_county["datetime"].dt.to_timestamp(), utc=True
    )

    # Rounding the (latitude) and (longitude) for 1 decimal fraction
    hist_weather[["latitude", "longitude"]] = (
        hist_weather[["latitude", "longitude"]].astype(float).round(1)
    )

    # Merging counties in locations data with the coordinations in the
    # historical_weather data
    hist_weather = hist_weather.merge(
        locations, how="left", on=["longitude", "latitude"]
    )

    # Dropping nan values
    hist_weather.dropna(axis=0, inplace=True)

    # Dropping the columns we won't need
    hist_weather.drop(["latitude", "longitude"], axis=1, inplace=True)

    # Converting (county) to integer
    hist_weather["county"] = hist_weather["county"].astype("int64")

    # Converting (datetime) column to datetime
    hist_weather["datetime"] = pd.to_datetime(hist_weather["datetime"], utc=True)

    """Grouping all historical_weather columns mean values by hour, So each hour
    will have the mean values of the historical_weather columns"""
    hist_weather_datetime = (
        hist_weather.groupby([hist_weather["datetime"].dt.to_period("h")])[
            list(
                hist_weather.drop(
                    ["county", "datetime", "data_block_id"], axis=1
                ).columns
            )
        ]
        .mean()
        .reset_index()
    )

    # After converting the (datetime) column to hour period for the groupby we convert
    # it back to datetime
    hist_weather_datetime["datetime"] = pd.to_datetime(
        hist_weather_datetime["datetime"].dt.to_timestamp(), utc=True
    )

    # Merging (data_block_id) back after dropping it in the last step |
    # (data_block_id will  be used to merge with train data)
    hist_weather_datetime = hist_weather_datetime.merge(
        hist_weather[["datetime", "data_block_id"]], how="left", on="datetime"
    )

    """Grouping all historical_weather columns mean values by hour and county, So each
    hour will have the mean values of the historical_weather columns for each county"""
    hist_weather_datetime_county = (
        hist_weather.groupby(["county", hist_weather["datetime"].dt.to_period("h")])[
            list(
                hist_weather.drop(
                    ["county", "datetime", "data_block_id"], axis=1
                ).columns
            )
        ]
        .mean()
        .reset_index()
    )

    # After converting the (datetime) column to hour period for the groupby we convert
    # it back to datetime
    hist_weather_datetime_county["datetime"] = pd.to_datetime(
        hist_weather_datetime_county["datetime"].dt.to_timestamp(), utc=True
    )

    # Merging (data_block_id) back after dropping it in the last step
    hist_weather_datetime_county = hist_weather_datetime_county.merge(
        hist_weather[["datetime", "data_block_id"]], how="left", on="datetime"
    )

    # Adding year column in train data
    data["year"] = data["datetime"].dt.year

    # Adding month column in train data
    data["month"] = data["datetime"].dt.month

    # Adding day column in train data
    data["day"] = data["datetime"].dt.day

    # Adding hour column in train data
    data["hour"] = data["datetime"].dt.hour

    # Adding dayofweek column in train data
    data["dayofweek"] = data["datetime"].dt.dayofweek

    # Adding dayofyear column in train data
    data["dayofyear"] = data["datetime"].dt.dayofyear

    # Adding hour column to electricity used to merge with the train data
    electricity["hour"] = electricity["datetime"].dt.hour

    # Merging train data with client data
    data = data.merge(
        client.drop(columns=["date"]),
        how="left",
        on=["data_block_id", "county", "is_business", "product_type"],
    )

    # Merging train data with gas data
    data = data.merge(
        gas[["data_block_id", "lowest_price_per_mwh", "highest_price_per_mwh"]],
        how="left",
        on="data_block_id",
    )

    # Merging train data with electricity data
    data = data.merge(
        electricity[["euros_per_mwh", "hour", "data_block_id"]],
        how="left",
        on=["hour", "data_block_id"],
    )

    # Merging train data with forecast_weather_datetime data
    data = data.merge(forecast_weather_datetime, how="left", on=["datetime"])

    # Merging train data with forecast_weather_datetime_county data
    data = data.merge(
        forecast_weather_datetime_county,
        how="left",
        on=["datetime", "county"],
        suffixes=("_fcast_mean", "_fcast_mean_by_county"),
    )

    # Creating hour columns in both historical_weather data | used to merge both data
    # with the train data
    hist_weather_datetime["hour"] = hist_weather_datetime["datetime"].dt.hour
    hist_weather_datetime_county["hour"] = hist_weather_datetime_county[
        "datetime"
    ].dt.hour

    # Dropping duplicates and (datetime) column
    hist_weather_datetime.drop_duplicates(inplace=True)
    hist_weather_datetime_county.drop_duplicates(inplace=True)
    hist_weather_datetime.drop("datetime", axis=1, inplace=True)
    hist_weather_datetime_county.drop("datetime", axis=1, inplace=True)

    # Merging hist_weather_datetime with train data
    data = data.merge(hist_weather_datetime, how="left", on=["data_block_id", "hour"])

    # Merging hist_weather_datetime_county with train data
    data = data.merge(
        hist_weather_datetime_county,
        how="left",
        on=["data_block_id", "county", "hour"],
        suffixes=("_hist_mean", "_hist_mean_by_county"),
    )

    # Filling nan values with hourly mean values for each column | Helps for the county
    #  missing value
    data = (
        data.groupby(["year", "day", "hour"], as_index=False)
        .apply(lambda x: x.ffill().bfill())
        .reset_index()
    )
    data = data.merge(counties, how="left", on="county").drop(columns=["county"])
    product_dict = {0: "Combined", 1: "Fixed", 2: "General service", 3: "Spot"}

    # Convert the dictionary to a list of tuples
    product_list = list(product_dict.items())

    # Create the DataFrame
    product = pd.DataFrame(product_list, columns=["product_type", "product_name"])

    data = data.merge(product, how="left", on="product_type").drop(
        columns=["product_type"]
    )
    data["is_business"] = data["is_business"].map({0: "Residential", 1: "Business"})
    data["is_consumption"] = data["is_consumption"].map(
        {0: "Production", 1: "Consumption"}
    )

    # Dropping unneeded data
    data.drop(
        ["row_id", "data_block_id", "year", "datetime"],
        axis=1,
        inplace=True,
    )

    data = data[
        list(
            chain(
                data.select_dtypes(include=["object"]).columns,
                data.select_dtypes(exclude=["object"]).columns,
            )
        )
    ]

    if path:
        data.to_csv(path, index=False)

    return data


# %%

if __name__ == "__main__":
    main()
