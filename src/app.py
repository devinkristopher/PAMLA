"""PAMLA's prediction API app.

This is called by the launcher at scripts/start_pamla.py, during the assignment of `api`, using subprocess.Popen(). 
> It is also callable directly with `uvicorn src.app:app --app-dir src --host`
> But without running `start_pamla.py`, it won't directly load `llama.cpp`, `Gemma`, and the local web servers, so the API endpoints will fail. :)

The entry point is `app`, a FastAPI instance.
`app`'s constructor is passed `lifespan`.
`lifespan` loads the config.yaml

GET /predict forecasts from the latest hour in the configured sensor CSV.
POST /ask interprets a question and explains that forecast using local Gemma.
Optional environment variables: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY.
Environment variables must be exported; this module does not load .env files.

As crazy as this may sound, our model will be predicting 3 distinct types of futures.
1. The Relative Future: relative to the model; the already-transpired readings logged by Home Assistant after the latest dataset timestamp.
- Our data collection period ended in Late August 2026, and it is currently Late September 2026. 
- That means we can obtain new data from the past month. While it is historical data to us, it is a predetermined future relative to the model's training period.
- This allows us to test the actual performance of the model on a known future, as if we were in the past and had to make a prediction without knowing the future.
2. The Absolute Future: our relative future; the actual future that is yet to transpire; that of which we cannot know the outcome.
- With this, our model can be given estimated future data (obtained from Apple WeatherKit API) to predict soil moisture in the future. 
- Your prediction horizon -- augmented by Apple's weather forecast -- is now greatly extended by the integration of absolute future data imputation.
- For example, if Apple provides a 12-hour forecast, we can impute our future data with that and predict 24 hours past their 12-hour forecast, giving us a 36-hour prediction horizon :D
- With that, our humble hyperlocal model -- merely trained to predict tokens of the next day -- now gives renewed worth to SaaS companies amidst the Saaspocalypse, 
- as it can now predict the future of the future, and the future of the future of the future, and so on.
3. The Immediate Future: Finally, our model will predict the immediate future, which is simply 24 hours from right now.
- While this may sound indistinguishable from the relative and absolute future, it is in fact quite different.
- The model's relative future relies on the formation of a contiguous dataset from the moment its training period ended. This removes sample bias, continuity bias, etc.;
- it resolves the question, "How would the model perform if it had another day, week, or month of data to learn from?"
- it is the model's ability to predict the future between the end of its training period and the present moment, where we can verify its performance against the predetermined future.
The model's prediction of the absolute future can include seaming a contiguous dataset from its training period to the present, 
but it could also rely on its pretrained weights and biases to predict the future without the introduction of new data. At any rate, it does require the model to predict the future without its or our knowledge of the actual outcome.
The immediate future is the model's "live" prediction of the next 24 hours based on the most recent, "live" data reading.
- Instead of measuring the model with known data or augmenting trend curves with external service predictions, the immediate future is the model's live, instantaneous prediction of the horizon, i.e., the initial intent of the model's design.
At any given moment, it would give you an immediate future prediction of the soil moisture.

The value is largely placed in how we contextualize the model's predictions.
- Relative: We evaluate the performance based on the actual future; at any point, we know the right and wrong answer. We could know a month out.
- Absolute: We evalute the prediction based on running probabilistic refinements. Predictive performance is evaluated based on what we know right now; 
- As Apple refines weather data and extended horizon predictions become closer, our ability to accurately evaluate the performance and our approximations improve as the current moment, science, and data approaches the given prediction.
- Immediate: Essentially the "next token" of our model's prediction; future prediction as well, so accuracy closes in as we approach the moment.
- The atomic function of active prediction, where the model generates live predictions based on complete information that exists right now, but no "future" data augmentations.
- Relative futures use past data, immediate futures use current data, and absolute futures use future data, showing that the tense is the derivative of the evaluation methodology.
"""

from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path
import io
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, AwareDatetime
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor
from typing import Literal
import math

from preprocess import DEW_POINT_COLUMN, HUMIDITY_COLUMN, RAIN_TOTAL_COL, TEMPERATURE_COLUMN, engineer_features, extend_dataframe, load_and_preprocess_data, preprocess_data, read_and_pivot_dataset
import preprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "configs" / "config.yaml"
MODELS_PATH = PROJECT_ROOT / "models"
DEPLOYMENT_JSON = MODELS_PATH / "deployment_metadata.json"
logger = logging.getLogger(__name__)

DISPLAY_COLUMNS = {
    "sensor.acurite_atlas_178_temperature": "temperature",
    "sensor.acurite_atlas_178_dew_point": "dew_point",
    "sensor.acurite_atlas_178_humidity": "humidity",
    "sensor.acurite_atlas_178_rain_total": "rain_total",
    "sensor.acurite_atlas_178_wind_speed": "wind_speed",
    "sensor.acurite_atlas_178_wind_direction": "wind_direction",
    "sensor.acurite_atlas_178_light_level": "light_level",
    "sensor.fineoffset_wh51_0f8613_soil_moisture": "soil_moisture",
}

def load_deployment_model(models_filepath):
    """Keep model-format handling separate from the prediction endpoint."""
    with (DEPLOYMENT_JSON).open() as file:
        metadata = json.load(file)
    model = XGBRegressor()
    model.load_model(str(models_filepath / metadata["model_file"]))
    return model, metadata

@asynccontextmanager
async def lifespan(app):
    with (CONFIG_YAML_PATH).open() as file:
        config = yaml.safe_load(file)
    model, metadata = load_deployment_model(MODELS_PATH)
    app.state.config = config
    app.state.model = model
    app.state.metadata = metadata

    app.state.df = read_and_pivot_dataset(
        PROJECT_ROOT / app.state.config["data"]["path"]
    )
    app.state.df = preprocess_data(app.state.df)

    yield

app = FastAPI(title="PAMLA", lifespan=lifespan)

def prepare_latest_features(df, metadata):
    """Prepare the latest hour of features for prediction, validating that all required features are present and finite."""
    df = preprocess.add_time_series_features(df)
    features = df.loc[:, metadata["features"]].tail(1)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(
            "Latest hour has missing or non-finite features. Supply at least "
            "12 hourly rows of sensor history with all required measurements."
        )
    return features

@app.get("/predict")
def predict():
    """Forecast 24-hour soil moisture using the configured dataset's latest hour."""
    config = app.state.config
    metadata = app.state.metadata
    try:
        df = load_and_preprocess_data(
            PROJECT_ROOT / config["data"]["path"]
        )


        print("\n=== PREPROCESSED DATAFRAME ===")
        print(df.columns.tolist())
        print("\nIndex:")
        print(df.index)
        print("\nLatest row:")
        print(df.tail(1))
        print("\n=== MODEL FEATURES ===")
        print(metadata["features"])

        features = prepare_latest_features(df, metadata)
    except FileNotFoundError as error:
        raise HTTPException(503, "Configured sensor dataset is unavailable.") from error
    except (KeyError, ValueError, pd.errors.EmptyDataError) as error:
        logger.exception("Sensor history could not be prepared")
        raise HTTPException(
            422,
            "Sensor history is incomplete or invalid. Check the CSV schema, "
            "required sensors, timestamps, and at least 12 hourly rows.",
        ) from error

    delta = float(app.state.model.predict(features)[0])
    current = float(features[metadata["current_soil_feature"]].iloc[0])
    predicted = current + delta
    if not np.isfinite(predicted):
        raise HTTPException(500, "Model returned a non-finite forecast.")
    timestamp = features.index[0]
    horizon = metadata["forecast_horizon_hours"]

    return {
        "model_name": metadata["model_name"],
        "data_as_of": timestamp.isoformat(),
        "forecast_for": (timestamp + pd.Timedelta(hours=horizon)).isoformat(),
        "forecast_horizon_hours": horizon,
        "current_soil_moisture": current,
        "predicted_change": delta,
        "predicted_soil_moisture": predicted,
        "persistence_baseline": current,
        "note": (
            "Forecast is relative to the dataset timestamp, not the current time. "
            "Preprocessing forward-fills missing readings as in training. "
            "Predictions are returned without clipping."
        ),
    }

class Observation(BaseModel):
    timestamp: AwareDatetime | None = Field(
        default=None,
        description="UTC timestamp, which matches the Home Assistant export. Absent values become previous_observation += 1 hour.",
    )
    temperature: float | None = Field(
        default=None,
        serialization_alias="sensor.acurite_atlas_178_temperature",
        description="Air temperature.",
    )

    dew_point: float | None = Field(
        default=None,
        serialization_alias="sensor.acurite_atlas_178_dew_point",
        description="Dew point.",
    )

    humidity: float | None = Field(
        default=None,
        ge=0,
        le=100,
        serialization_alias="sensor.acurite_atlas_178_humidity",
        description="Relative humidity as a percentage from 0 to 100.",
    )

    rainfall: float | None = Field(
        default=None,
        ge=0,
        description="Rainfall during this observation interval.",
    )

    wind_speed: float | None = Field(
        default=None,
        ge=0,
        serialization_alias="sensor.acurite_atlas_178_wind_speed",
        description="Wind speed.",
    )

    wind_direction: float | None = Field(
        default=None,
        ge=0,
        lt=360,
        serialization_alias="sensor.acurite_atlas_178_wind_direction",
        description="Wind direction in degrees.",
    )

    light_level: float | None = Field(
        default=None,
        ge=0,
        serialization_alias="sensor.acurite_atlas_178_light_level",
        description="Ambient light intensity measured in lux.",
    )

    soil_moisture: float | None = Field(
        default=None,
        serialization_alias="sensor.fineoffset_wh51_0f8613_soil_moisture",
        description="Soil-moisture reading.",
    )

class AskRequest(BaseModel):
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=1000
    )


class LLMExtraction(BaseModel):
    status: Literal["ok", "clarify"]
    values: Observation
    clarification: str | None = None


REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["forecast", "clarify", "unsupported"]},
        "plant": {"type": "string", "enum": ["ginger", "other", "unspecified"]},
        "horizon_hours": {"type": ["integer", "null"]},
    },
    "required": ["intent", "plant", "horizon_hours"],
    "additionalProperties": False,
}

def call_gemma(prompt):
    """Make one independent, bounded request to the local model server."""
    payload = {
        "model": os.environ.get("LLM_MODEL", "pamla-gemma"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 350,
        "stream": False,
    }
    # Request JSON in the prompt and validate it in Python. Avoid response_format:
    # this Gemma/llama.cpp combination fails when initializing its grammar sampler.
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except HTTPError as error:
        raise HTTPException(502, "Gemma rejected the request; check its server output.") from error
    except (URLError, TimeoutError, OSError) as error:
        raise HTTPException(503, "Gemma is unavailable or timed out. Check llama-server on port 8080.") from error
    except (ValueError, UnicodeError) as error:
        raise HTTPException(502, "Gemma returned an invalid response.") from error
    try:
        choice = result["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
            raise ValueError("Incomplete model response")
        return content.strip()
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise HTTPException(502, "Gemma returned an empty or incomplete response.") from error

def parse_gemma_json(content):
    """Accept plain JSON or a single Markdown JSON block; reject extra prose."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in ("```json", "```") or lines[-1].strip() != "```":
            raise ValueError("Expected a single JSON code block")
        text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)

@app.post("/ask")
def ask(request: AskRequest):

    question = request.question.strip()
    if not question:
        raise HTTPException(
            422,
            "Enter a question about ginger soil moisture."
        )

    # 1. Gemma extracts hypothetical changes
    prompt = f"""
    Interpret the user's request for PAMLA.

    Return JSON only in exactly this structure:

    {{
    "status": "ok" or "clarify",
    "values": {{}},
    "clarification": null or "a short clarification question"
    }}

    Use status="ok" when:
    - the requested sensor changes are clear, or
    - the user simply requests a forecast without specifying any sensor changes.

    Use status="clarify" when:
    - the user appears to request a sensor change but the value or meaning
    cannot be reliably determined.

    For a normal forecast with no sensor changes, return:
    {{
    "status": "ok",
    "values": {{}},
    "clarification": null
    }}

    Never invent, estimate, assume, or supply a sensor value merely because
    the user asks for a prediction.

    Only include sensor values explicitly stated or directly described by the user.
    Numbers unrelated to sensor measurements, such as money, counts, or identifiers,
    must not be interpreted as sensor values.

    Possible sensor fields:
    temperature
    dew_point
    humidity
    rainfall
    wind_speed
    wind_direction
    light_level
    soil_moisture

    Interpret common abbreviated units from context.

    Examples:
    - "80 F", "80°F", and "80 degrees F" mean temperature=80.
    - "70 percent humidity", "70% humidity", and "humidity 70" mean humidity=70.
    - Cardinal wind directions should be converted to degrees:
    north=0, east=90, south=180, west=270.
    - Return numeric values only; do not include unit strings.

    Normalize all extracted measurements to PAMLA's model units:
    - temperature: degrees Fahrenheit (°F)
    - dew_point: degrees Fahrenheit (°F)
    - humidity: percent from 0 to 100
    - rainfall: inches during the observation interval
    - wind_speed: miles per hour (mph)
    - wind_direction: degrees from 0 inclusive to 360 exclusive
    - light_level: lux from 0 to 120,000
    - soil_moisture: percent from 0 to 100

    Interpret qualitative light descriptions using these ranges:
    - Dark/Night: 0–500 lux
    - Low Light: 501–5,380 lux
    - Overcast/Shade: 5,381–21,520 lux
    - Daylight: 21,521–43,050 lux
    - Direct Sun: 43,051–120,000 lux

    When a qualitative light condition is explicitly stated, use the midpoint
    of its range. Do not infer light level solely from timestamp, time of day,
    weather, or other fields.

    Convert explicitly stated alternative units into PAMLA's units.
    Do not convert a value unless its unit is stated or unambiguous.

    Examples:

    User: "Predict ginger soil moisture 24 hours after the latest reading."
    Output:
    {{"status":"ok","values":{{}},"clarification":null}}

    User: "What if it is 80 F and humidity is 40%?"
    Output:
    {{"status":"ok","values":{{"temperature":80,"humidity":40}},"clarification":null}}

    User: "What if the temperature is higher?"
    Output:
    {{"status":"clarify","values":{{}},"clarification":"What temperature would you like me to use?"}}

    USER QUESTION:
    {json.dumps(question)}
    """

    try:
        parsed = parse_gemma_json(call_gemma(prompt))

        print("\n=== GEMMA EXTRACTION ===")
        print(parsed)

        extraction = LLMExtraction(**parsed)

    except (ValueError, TypeError) as error:
        raise HTTPException(
            502,
            "Gemma could not interpret the request reliably."
        ) from error
    
    if extraction.status == "clarify":
        return {
            "status": "needs_clarification",
            "answer": extraction.clarification,
            "forecast": None,
        }

    observation = extraction.values

    # 2. Load current session sensor history
    df = app.state.df.copy()

    changes = observation.model_dump(
        exclude_none=True,
        exclude={"timestamp"},
    )

    # 3. Append a hypothetical row when something changed
    if changes:
        new_df = observation_to_df(observation, df)
        df = extend_dataframe(df, new_df)

        # Recalculate dew point unless the user explicitly supplied one
        if "dew_point" not in changes:
            latest = df.index[-1]

            df.loc[latest, DEW_POINT_COLUMN] = calculate_dew_point_f(
                df.loc[latest, TEMPERATURE_COLUMN],
                df.loc[latest, HUMIDITY_COLUMN],
            )

        # Persist the completed hypothetical observation
        app.state.df = df

    # 4. Perform normal feature engineering
    final_df = engineer_features(df)

    features = prepare_latest_features(
        final_df,
        app.state.metadata,
    )

    metadata = app.state.metadata

    delta = float(app.state.model.predict(features)[0])

    current = float(
        features[metadata["current_soil_feature"]].iloc[0]
    )

    predicted = current + delta

    timestamp = features.index[0]
    horizon = metadata["forecast_horizon_hours"]

    forecast = {
        "model_name": metadata["model_name"],
        "data_as_of": timestamp.isoformat(),
        "forecast_for": (
            timestamp + pd.Timedelta(hours=horizon)
        ).isoformat(),
        "forecast_horizon_hours": horizon,
        "current_soil_moisture": current,
        "predicted_change": delta,
        "predicted_soil_moisture": predicted,
        "persistence_baseline": current,
    }


    parsed = parse_gemma_json(call_gemma(prompt))

    print("\n=== GEMMA EXTRACTION ===")
    print(parsed)

    extraction = LLMExtraction(**parsed)

    if extraction.status == "clarify":
        return {
            "status": "needs_clarification",
            "answer": extraction.clarification,
            "forecast": None,
        }

    observation = extraction.values

    print("\n=== OBSERVATION ===")
    print(observation)

    fallback = (
    f"Using the dataset as of {forecast['data_as_of']}, PAMLA forecasts soil "
    f"moisture of {forecast['predicted_soil_moisture']:.2f} at "
    f"{forecast['forecast_for']}, compared with "
    f"{forecast['current_soil_moisture']:.2f} at the dataset timestamp. "
    "This is a model estimate, not a live measurement or a watering recommendation."
    )

    try:
        explanation = call_gemma(
            "Explain this ginger soil-moisture forecast in at most three sentences. "
            "Use only the supplied facts; invent no causes, accuracy claims, thresholds, "
            "or watering advice. Include data_as_of and forecast_for; never call the "
            "data live or current today. State that this is an estimate. "
            "predicted_change is an additive change, not a relative percentage. "
            "The persistence_baseline simply repeats the last soil reading.\n"
            "FORECAST: " + json.dumps(forecast)
        )
        explanation_source = "gemma"
    except HTTPException:
        logger.warning(
            "Gemma explanation unavailable; returning the numerical forecast."
        )
        explanation = fallback
        explanation_source = "template"



    print("\n=== SESSION DATA ===")
    print(app.state.df.tail(3).T)

    return {
    "status": "ok",
    "answer": explanation,
    "explanation_source": explanation_source,
    "forecast": forecast,

}


def calculate_dew_point_f(temperature_f, humidity):
    temperature_c = (temperature_f - 32) * 5 / 9

    a = 17.625
    b = 243.04

    gamma = (
        math.log(humidity / 100)
        + (a * temperature_c) / (b + temperature_c)
    )

    dew_point_c = (b * gamma) / (a - gamma)

    return dew_point_c * 9 / 5 + 32

@app.get("/", response_class=FileResponse, include_in_schema=False)
def home():
    """Serve the local PAMLA interface."""
    return FileResponse(PROJECT_ROOT / "src" / "static" / "index.html")

@app.get("/scenario")
def get_scenario():
    display = (
        app.state.df[list(DISPLAY_COLUMNS.keys())]
        .tail(5)
        .rename(columns=DISPLAY_COLUMNS)
        .reset_index(names="timestamp")
    )

    display["timestamp"] = display["timestamp"].astype(str)

    return {
        "rows": display.to_dict(orient="records")
    }

@app.post("/scenario/reset")
def reset_scenario():
    df = read_and_pivot_dataset(
        PROJECT_ROOT / app.state.config["data"]["path"]
    )
    app.state.df = preprocess_data(df)

    return {"status": "ok"}

def observation_to_df(
    observation: Observation,
    original_df: pd.DataFrame,
) -> pd.DataFrame:
    data = observation.model_dump(
        by_alias=True,
        exclude_none=True,
    )

    rainfall = data.pop("rainfall", None)

    if rainfall is not None:
        previous_rain_total = original_df[RAIN_TOTAL_COL].iloc[-1]
        data[RAIN_TOTAL_COL] = previous_rain_total + rainfall

    return pd.DataFrame([data])