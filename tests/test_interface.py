import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from src.app import app, Observation, LLMExtraction, calculate_dew_point_f, observation_to_df
from src.preprocess import RAIN_TOTAL_COL

import numpy as np
import pandas as pd


def test_interface_parses_valid_observation_values():
    extraction = LLMExtraction(
        status="ok",
        values=Observation(
            temperature=80,
            humidity=40,
            rainfall=0.5,
            wind_direction=270,
        ),
        clarification=None,
    )

    assert extraction.status == "ok"
    assert extraction.values.temperature == 80
    assert extraction.values.humidity == 40
    assert extraction.values.rainfall == 0.5
    assert extraction.values.wind_direction == 270


def test_interface_rejects_invalid_sensor_values():
    with pytest.raises(ValidationError):
        Observation(
            humidity=140,
        )

def test_dew_point_from_temp_humidity():
    dew_point = calculate_dew_point_f(
        temperature_f=80,
        humidity=40,
    )

    assert np.isfinite(dew_point)
    assert dew_point < 80

def test_observation_to_df_converts_rainfall_to_cumulative_total():
    original_df = pd.DataFrame(
        {
            RAIN_TOTAL_COL: [0.10, 0.16],
        }
    )

    observation = Observation(
        temperature=82,
        rainfall=0.50,
    )

    result = observation_to_df(observation, original_df)

    assert result[RAIN_TOTAL_COL].iloc[0] == pytest.approx(0.66)
    assert "rainfall" not in result.columns


# Here we will test the interface.
# During acceptance testing, we did find some anomalies, such as Gemma erroring out when given temperature as "temperature is [0-9] F"
# I considered using Selenium for UI automation, but it was out of scope of the preferred stack, so we're going to test the entire /ask endpoint.
# First, we'll reset the dataset modifications to its clean state using the reset button's endpoint, call /ask, assert all is good, 
# Test that the temperature was parsed successfully, then finally reset the state again using the reset button's endpoint function, as we did in the beginning.
@pytest.mark.integration
def test_gemma_parses_fahrenheit_with_space():
    with TestClient(app) as client:
        response = client.post("/scenario/reset")
        assert response.status_code == 200

        # This previously caused trouble that led to us modifying the custom instructions.
        # int F is now parsed as degrees instead of erroring out.
        response = client.post(
            "/ask",
            json={"question": "What if the temperature is 80 F?"},
        )

        # assert that all returns 200 OK
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        response = client.get("/scenario")
        assert response.status_code == 200

        rows = response.json()["rows"]

        # assert tht the last row (using negative index slicing)'s temperature col is approx 80.0 F
        assert rows[-1]["temperature"] == pytest.approx(80.0)

        # clean up the live state
        # As we noted in the comments in the requirements.txt, this uses TestClient, which tests ASGI servers without needing a live instance.
        # Honestly, I'm not an expert on TestClient, but I believe when `with TestClient(app) as client` is called,
        # it spins up the ASGI application in-process (not a full server instantiation),
        # runs the tests on the application layer, then at the end of the test, the instance is sent to the garbage collector.
        # So the instance and therefore state is **probably** contained to the life of the test itself.
        # IF that's the case, we actualyl don't need to run our rest button state cleanup function at the beginning and possibly end.
        # But we will for best practices, as sometimes things persist in in unintended manner.
        client.post("/scenario/reset")