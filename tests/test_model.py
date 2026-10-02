from pathlib import Path

import numpy as np
import yaml

from src import preprocess
from src.app import load_deployment_model, prepare_latest_features
from src.train import create_target, split_data, evaluate_model


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "configs" / "config.yaml"

SOIL_COL = "sensor.fineoffset_wh51_0f8613_soil_moisture"


def load_test_objects():
    with CONFIG_PATH.open() as file:
        config = yaml.safe_load(file)

    model, metadata = load_deployment_model(MODELS_DIR)

    df = preprocess.load_and_preprocess_data(
        ROOT / config["data"]["path"]
    )

    return config, model, metadata, df


def test_deployed_model_prediction_shape_and_type():
    _, model, metadata, df = load_test_objects()

    features = prepare_latest_features(df, metadata)

    prediction = model.predict(features)

    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])
    assert isinstance(float(prediction[0]), float)


def test_deployed_model_meets_minimum_performance_threshold():
    config, model, metadata, df = load_test_objects()

    df = preprocess.add_time_series_features(df)
    df = create_target(df)
    df = df.dropna()

    _, X_test, _, y_test = split_data(
        df,
        train_size=config["split"]["train_size"],
        gap=config["split"]["gap_hours"],
    )

    X_test = X_test[metadata["features"]]

    predicted_delta = model.predict(X_test)

    absolute_prediction = (
        X_test[SOIL_COL].to_numpy()
        + predicted_delta
    )

    metrics = evaluate_model(
        y_test,
        absolute_prediction,
    )

    assert metrics["R2"] > 0.50