"""Preprocessing finalized in exploration.ipynb.

For a more ergonomic way to review preprocessing, please review
notebooks/exploration.ipynb.

Updated to fix the tolerance variable into a constant and remove the tolerance
parameter from functions that don't need it.

During development, a 5-minute tolerance window for the hourly data point was
settled on to ensure adequate periodicity in sample rate. We found that even in
our least-verbose infrastructure iterations, our HA/HB Smart Home server could
reliably produce data within a 5-minute window of the hour.

For reference, using the RTL-SDR straight into Home Assistant produced much
more resolute data, while the Homebridge > MQTT > Home Assistant pipeline
produced much more variable data, but still within a 5-minute window of the
hour. As a result, we find that even rough, less-direct iterations of this stack
can be reasonably expected to produce data within a 5-minute window of the
hour.

After tuning this ML model on months of data using a 5-minute tolerance window,
we realized that later allowing increased laxity in the tolerance window could change
the temporal mechanics of how/what data is presented to the model. This could
lead to performance drift or degradation while leaving the cause hidden in a
seemingly harmless config change. It would also hold subsequent
data to a less strict collection standard than the data used during model
development. Ultimately, changing this during model development would 
be changing the fundamental assumptions of the experiment, 
and would require reprocessing and re-evaluation of the model on data 
aligned with the new parameters. 

Because it would lead to a sort of rolling shutter effect in data collection, 
the 5-minute tolerance is therefore part of the provenance of this model and
the preprocessing contract under which it was trained and evaluated. For that
reason, the 5-minute forward-fill tolerance is treated as a preprocessing
invariant rather than a runtime configuration option.

It is still possible to change the tolerance window when developing a new
iteration of the model, but such a change should be accompanied by reprocessing
and re-evaluation under the new sampling policy. Accordingly, tolerance is not
exposed as a parameter of any function in this module.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

TEMPERATURE_COLUMN = "sensor.acurite_atlas_178_temperature"
DEW_POINT_COLUMN = "sensor.acurite_atlas_178_dew_point"
HUMIDITY_COLUMN = "sensor.acurite_atlas_178_humidity"
RAIN_TOTAL_COLUMN = "sensor.acurite_atlas_178_rain_total"
WIND_SPEED_COLUMN = "sensor.acurite_atlas_178_wind_speed"
WIND_DIRECTION_COLUMN = "sensor.acurite_atlas_178_wind_direction"
LIGHT_LEVEL_COLUMN = "sensor.acurite_atlas_178_light_level"
SOIL_MOISTURE_COLUMN = "sensor.fineoffset_wh51_0f8613_soil_moisture"
SOIL_COL = "sensor.fineoffset_wh51_0f8613_soil_moisture"
TEMP_COL = "sensor.acurite_atlas_178_temperature"
TARGET_COL = "target_soil_moisture_24h"
RAIN_TOTAL_COL = "sensor.acurite_atlas_178_rain_total"
HORIZON_HOURS = 24

HOURLY_TOLERANCE = "5min"
HOURLY_FREQ = "1h"

SOIL_COL = "sensor.fineoffset_wh51_0f8613_soil_moisture"
TEMP_COL = "sensor.acurite_atlas_178_temperature"
TARGET_COL = "target_soil_moisture_24h"

UNRELATED_WH51_SENSOR_IDS = (
    "0f85e7",
    "0f861a",
    "0fa7d7",
    "0fa9d0",
)

PHASE_1_COLUMNS_TO_DROP = (
    "automation.ginger_low_humidity",
    "sensor.dashboard_moon_phase",
    "sensor.acurite_atlas_a_178_atlas_lightning_distance_km",
    "sensor.acurite_atlas_a_178_atlas_lightning_distance_mi",
    "sensor.acurite_atlas_a_178_atlas_noise",
    "sensor.acurite_atlas_a_178_atlas_rssi",
    "sensor.acurite_atlas_a_178_atlas_snr",
    "sensor.acurite_atlas_a_178_atlas_humidity",
    "sensor.acurite_atlas_a_178_atlas_illuminance_lux",
    "sensor.acurite_atlas_a_178_atlas_lightning_strike_count",
    "sensor.acurite_atlas_a_178_atlas_rain_total_in",
    "sensor.acurite_atlas_a_178_atlas_rain_total_mm",
    "sensor.acurite_atlas_a_178_atlas_temperature_degc",
    "sensor.acurite_atlas_a_178_atlas_temperature_degf",
    "sensor.acurite_atlas_a_178_atlas_uv_index",
    "sensor.acurite_atlas_a_178_atlas_wind_avg_mph",
    "sensor.acurite_atlas_a_178_atlas_wind_direction_deg",
    "sensor.fineoffset_wh51_0f8613_model",
    "sensor.tfa_303151_16_humidity",
    "sensor.fineoffset_wh51_0f8613_battery_voltage",
    "sensor.fineoffset_wh51_0f8613_boost",
    "sensor.acurite_atlas_178_frequency",
    "sensor.fineoffset_wh51_0f8613_frequency_2",
    "sensor.acurite_atlas_178_sequence_num",
    "sensor.fineoffset_wh51_0f8613_frequency",
    "sensor.acurite_atlas_178_message_type",
    "sensor.acurite_atlas_178_raw_msg",
    "binary_sensor.none_atlas_battery_ok",
    "sensor.acurite_atlas_178_channel",
    "sensor.acurite_atlas_178_exception",
    "binary_sensor.atlas_condensation_risk",
    "sensor.acurite_atlas_178_model",
    "sensor.atlas_wind_direction",
)


PHASE_2_COLUMNS_TO_DROP = (
    "sensor.acurite_atlas_dew_point",
    "sensor.acurite_atlas_dew_point_simple_formula",
    "sensor.acurite_atlas_178_signal_rssi",
    "sensor.acurite_atlas_178_signal_snr",
    "sensor.acurite_atlas_178_noise_floor",
    "sensor.acurite_atlas_178_uv_index",
    "sensor.acurite_atlas_178_strike_count",
    "sensor.acurite_atlas_178_storm_distance",
    "sensor.fineoffset_wh51_0f8613_signal_rssi",
    "sensor.fineoffset_wh51_0f8613_signal_snr",
    "sensor.fineoffset_wh51_0f8613_noise_floor",
    "sensor.fineoffset_wh51_0f8613_ad_raw",
    "sensor.sun_next_dawn",
    "sensor.sun_next_dusk",
    "sensor.sun_next_midnight",
    "sensor.sun_next_noon",
    "sensor.sun_next_rising",
    "sensor.sun_next_setting",
    "sun.sun",
    "sensor.moon_phase",
)

def create_target(df):
    """Create the soil-moisture target at 24-hour target horizon."""
    df = df.copy()

    future_times = df.index + pd.Timedelta(hours=HORIZON_HOURS)
    df[TARGET_COL] = df[SOIL_COL].reindex(future_times).to_numpy()

    return df


def add_time_series_features(df):
    df = df.copy()

    for lag in (1, 3, 6):
        df[f"soil_delta_{lag}h"] = df[SOIL_COL].diff(lag)
        
    for window in (6, 12):
        df[f"soil_mean_{window}h"] = df[SOIL_COL].rolling(window).mean()

    return df



def read_and_pivot_dataset(dataset_path):
    """Read dataset.csv and pivot the Home Assistant entity states."""
    df = pd.read_csv(dataset_path)

    # Because of the format of our raw CSV, we'll need to pivot the DataFrame,
    # where columns are entity_id, values are state, and the index is last_changed.
    return df.pivot(index="last_changed", columns="entity_id", values="state")


def drop_integrity_and_battery_low_columns(df):
    """Drop integrity-check and low-battery columns outside the project scope."""
    # Drop all columns that end with '_integrity_check'.
    df = df.drop(df.loc[:, df.columns.str.endswith("_integrity_check")].columns, axis=1)

    # Next, we will drop those ending with _battery_low.
    return df.drop(df.loc[:, df.columns.str.endswith("_battery_low")].columns, axis=1)


def drop_unrelated_wh51_sensors(df):
    """Keep the 0f8613 sensor in the ginger plant and drop the other four."""
    for sensor_id in UNRELATED_WH51_SENSOR_IDS:
        df = df.drop(df.loc[:, df.columns.str.contains(sensor_id)].columns, axis=1)
    return df


def drop_phase_1_columns(df):
    """Apply the finalized first-pass column decisions."""
    return df.drop(columns=list(PHASE_1_COLUMNS_TO_DROP))


def drop_phase_2_columns(df):
    """Apply the finalized column-condensation decisions."""
    return df.drop(columns=list(PHASE_2_COLUMNS_TO_DROP))


def remove_empty_rows(df):
    """Remove rows with no remaining measurements."""
    measurement_cols = df.columns.drop("last_changed")

    return df.dropna(
        how="all",
        subset=measurement_cols,
    ).copy()


def convert_datatypes(df):
    """Convert the timestamp and numeric measurement columns."""
    df = df.copy()

    # UTC time stamp format, per our data
    df["last_changed"] = pd.to_datetime(
        df["last_changed"],
        errors="coerce",
        utc=True,
    )

    # numeric columns become numeric
    numeric_cols = df.columns.drop("last_changed")
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return df


def downsample_hourly(df, col):
    data = (
        df[["last_changed", col]]
        .dropna(subset=[col])
        .sort_values("last_changed")
        .drop_duplicates(subset="last_changed")
        .set_index("last_changed")[col]
    )

    hourly_index = pd.date_range(
        start=df["last_changed"].min().floor("h"),
        end=df["last_changed"].max().floor("h"),
        freq=HOURLY_FREQ,
        tz="UTC",
    )
    # forward fill makes sure that:
    # the data reindexed and returned stays within the time bounds (the given hour). 
    # method = nearest would allow something a minute or two after the hour bound;
    # ffill would ensure that reindexed data for that hour period stays strictly within the hour boundaries.
    return data.reindex(
        hourly_index,
        method="ffill",
        tolerance=pd.Timedelta(HOURLY_TOLERANCE),
    )


def make_hourly_dataframe(df):
    """Fit each measurement to the hourly timeseries index."""
    model_cols = [col for col in df.columns if col != "last_changed"]

    df_hourly = pd.DataFrame(index=pd.date_range(
        start=df["last_changed"].min().floor("h"),
        end=df["last_changed"].max().floor("h"),
        freq=HOURLY_FREQ,
        tz="UTC",
    ))

    for col in model_cols:
        df_hourly[col] = downsample_hourly(df, col)

    df_hourly.index.name = "last_changed"
    return df_hourly


def add_hourly_rainfall(df_hourly):
    """Convert cumulative rain total to hourly rain total."""
    df_hourly = df_hourly.copy()
    rain_col = "sensor.acurite_atlas_178_rain_total"

    df_hourly["rain_hourly"] = df_hourly[rain_col].diff()
    df_hourly.loc[df_hourly["rain_hourly"] < 0, "rain_hourly"] = 0
    df_hourly["rain_hourly"] = df_hourly["rain_hourly"].fillna(0)
    return df_hourly


def encode_wind_direction(df_hourly):
    """Convert wind degrees to sine and cosine features."""
    df_hourly = df_hourly.copy()
    wind_col = "sensor.acurite_atlas_178_wind_direction"

    radians = np.deg2rad(df_hourly[wind_col])
    df_hourly["wind_dir_sin"] = np.sin(radians)
    df_hourly["wind_dir_cos"] = np.cos(radians)
    df_hourly.drop(columns=[wind_col], inplace=True)
    return df_hourly


def preprocess_data(df):
    """Apply only the preprocessing decisions finalized in the notebook."""
    df = drop_integrity_and_battery_low_columns(df)
    df = drop_unrelated_wh51_sensors(df)
    df = drop_phase_1_columns(df)

    # The notebook resets the pivoted timestamp index before Phase 2.
    df = df.reset_index()

    df = drop_phase_2_columns(df)
    df = remove_empty_rows(df)
    df = convert_datatypes(df)
    df_hourly = make_hourly_dataframe(df)

    # Finally, we'll forward fill our data to smooth the gaps.
    return df_hourly.ffill()

def engineer_features(df_hourly):
    df_hourly = add_hourly_rainfall(df_hourly)
    return encode_wind_direction(df_hourly)


def load_and_preprocess_data(dataset_path):
    df = read_and_pivot_dataset(dataset_path)
    df = preprocess_data(df)
    return engineer_features(df)

def extend_dataframe(original_df, new_data):
    new_data = new_data.copy()

    if "timestamp" in new_data.columns:
        new_data["timestamp"] = pd.to_datetime(new_data["timestamp"], utc=True)
        new_data = new_data.set_index("timestamp")

    elif not isinstance(new_data.index, pd.DatetimeIndex):
        start = original_df.index[-1] + pd.Timedelta(HOURLY_FREQ)
        new_data.index = pd.date_range(
            start=start,
            periods=len(new_data),
            freq=HOURLY_FREQ,
        )

    combined = pd.concat([original_df, new_data])
    combined = combined.sort_index()
    combined = combined.ffill()

    return combined