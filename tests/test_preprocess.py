import numpy as np
import pandas as pd

from src import preprocess


RAIN_COL = "sensor.acurite_atlas_178_rain_total"
WIND_COL = "sensor.acurite_atlas_178_wind_direction"
TEMP_COL = "sensor.acurite_atlas_178_temperature"
SOIL_COL = "sensor.fineoffset_wh51_0f8613_soil_moisture"


def test_add_hourly_rainfall_from_cumulative_total():
    df = pd.DataFrame(
        {
            RAIN_COL: [0.10, 0.15, 0.13, 0.33],
        }
    )

    result = preprocess.add_hourly_rainfall(df)

    # First row has no prior observation.
    # Negative difference represents a counter reset and becomes zero.
    expected = [0.0, 0.05, 0.0, 0.20]

    assert np.allclose(result["rain_hourly"], expected)


def test_encode_wind_direction():
    df = pd.DataFrame(
        {
            WIND_COL: [0, 90, 180, 270],
        }
    )

    result = preprocess.encode_wind_direction(df)

    assert np.allclose(
        result["wind_dir_sin"],
        [0, 1, 0, -1],
        atol=1e-10,
    )

    assert np.allclose(
        result["wind_dir_cos"],
        [1, 0, -1, 0],
        atol=1e-10,
    )

    assert WIND_COL not in result.columns


def test_extend_dataframe_forward_fills_missing_values():
    index = pd.date_range(
        "2026-08-28 22:00",
        periods=2,
        freq="1h",
        tz="UTC",
    )

    original = pd.DataFrame(
        {
            TEMP_COL: [80.0, 81.0],
            SOIL_COL: [40.0, 40.0],
        },
        index=index,
    )

    new_data = pd.DataFrame(
        [
            {
                "timestamp": "2026-08-29T00:00:00Z",
                SOIL_COL: 35.0,
            }
        ]
    )

    result = preprocess.extend_dataframe(original, new_data)

    latest = result.iloc[-1]

    # Temperature was omitted, so the previous reading should persist.
    assert latest[TEMP_COL] == 81.0
    assert latest[SOIL_COL] == 35.0


def test_preprocessing_helpers_do_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            RAIN_COL: [0.10, 0.20],
        }
    )

    original = df.copy(deep=True)

    preprocess.add_hourly_rainfall(df)

    pd.testing.assert_frame_equal(df, original)