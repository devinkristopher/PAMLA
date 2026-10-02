import pandas as pd
import json

from preprocess import load_and_preprocess_data, add_time_series_features, create_target, SOIL_COL, TEMP_COL, TARGET_COL, HORIZON_HOURS, HOURLY_TOLERANCE, HOURLY_FREQ
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from itertools import product
from tensorflow import keras
import yaml
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.tensorflow
from hashlib import sha256
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import preprocess

def create_target(df):
    """Create the soil-moisture target at 24-hour target horizon."""
    df = df.copy()

    future_times = df.index + pd.Timedelta(hours=HORIZON_HOURS)
    df[TARGET_COL] = df[SOIL_COL].reindex(future_times).to_numpy()

    return df

def split_data(df, train_size=0.8, gap=24):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    split_idx = int(len(df) * train_size)
    train_end = split_idx - gap

    X_train = X.iloc[:train_end]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:train_end]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

def train_random_forest(X_train, y_train, params):
    model = RandomForestRegressor(
        **params
    )
    model.fit(X_train, y_train)

    return model

def train_xgboost(X_train, y_train, params):
    model = XGBRegressor(
        **params
    )
    model.fit(X_train, y_train)

    return model

def train_mlp(X_train, y_train, params):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        *[keras.layers.Dense(units, activation=params["activation"])
          for units in params["hidden_layers"]],
        keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.get(params["optimizer"]),
        loss=params["loss"],
        metrics=params["metrics"]
    )

    model.fit(
        X_train_scaled,
        y_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_split=params["validation_split"],
        shuffle=params["shuffle"],
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=params["patience"],
                restore_best_weights=params["restore_best_weights"],
                monitor=params["monitor"],
                min_delta=params["min_delta"],
                mode=params["mode"]
            )
        ],
        verbose=0
    )

    return model, scaler

def time_series_cv_xgboost_delta(X_train, y_train, params, n_splits=5, gap=24):
    """Evaluate delta-target XGBoost using chronological cross-validation."""
    tscv = TimeSeriesSplit(
    n_splits=n_splits,
    gap=gap
    )

    fold_metrics = []
    baseline_fold_metrics = []
    mlflow.log_params({"cv_splits": n_splits, "cv_gap_rows": gap})
    mlflow.log_params(params)

    for fold, (train_idx, val_idx) in enumerate(
        tscv.split(X_train),
        start=1
    ):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Train the model on change in soil moisture rather than
        # absolute future soil moisture.
        y_fold_train_delta = (
            y_fold_train - X_fold_train[SOIL_COL]
        )

        model = train_xgboost(
            X_fold_train,
            y_fold_train_delta,
            params
        )

        # Predict the future change in soil moisture.
        pred_delta = model.predict(X_fold_val)

        # Reconstruct absolute future soil moisture.
        pred_absolute = (
            X_fold_val[SOIL_COL].to_numpy()
            + pred_delta
        )

        # Persistence baseline for this validation fold.
        persistence_pred = X_fold_val[SOIL_COL].to_numpy()

        persistence_metrics = evaluate_model(
            y_fold_val,
            persistence_pred
        )

        # Evaluate in absolute soil-moisture space so these metrics
        # are comparable to the persistence baseline and final test.
        metrics = evaluate_model(
            y_fold_val,
            pred_absolute
        )

        fold_metrics.append(metrics)
        baseline_fold_metrics.append(persistence_metrics)
        print(f"\nFold {fold}")

        print(
            f"Train: {X_fold_train.index.min()} "
            f"→ {X_fold_train.index.max()}"
        )

        print(
            f"Validation: {X_fold_val.index.min()} "
            f"→ {X_fold_val.index.max()}"
        )

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nPersistence Baseline")

        for metric, value in persistence_metrics.items():
            print(f"{metric}: {value:.4f}")

        rmse_improvement = (
            persistence_metrics["RMSE"] - metrics["RMSE"]
        )

        print(
            f"XGBoost RMSE improvement vs persistence: "
            f"{rmse_improvement:.4f}"
        )

    print("\nMean CV Performance")

    mlflow.log_metrics({
        f"cv_{name.lower()}": float(np.mean([m[name] for m in fold_metrics]))
        for name in fold_metrics[0]
    })
    mlflow.log_metrics({
        f"cv_std_{name.lower()}": float(np.std([m[name] for m in fold_metrics]))
        for name in fold_metrics[0]
    })
    mlflow.log_metrics({
        f"cv_persistence_{name.lower()}": float(np.mean([m[name] for m in baseline_fold_metrics]))
        for name in baseline_fold_metrics[0]
    })
    # Save one representative trained model, not a model for every fold.
    # CV metrics above summarize all folds; this artifact is the last fold only.
    mlflow.log_param("model_training_scope", "last CV fold training partition")
    mlflow.xgboost.log_model(model, artifact_path="model")

    for metric in fold_metrics[0]:
        values = [
            fold[metric]
            for fold in fold_metrics
        ]

        print(
            f"{metric}: "
            f"{np.mean(values):.4f} "
            f"(± {np.std(values):.4f})"
        )

    return fold_metrics

def tune_xgboost_delta(
    X_train,
    y_train,
    base_params,
    tuning_params,
    run_params,
    experiment_id,
    n_splits=5,
    gap=24
):
    """Log each CV configuration and select its winner from MLflow."""
    combinations = product(
        tuning_params["n_estimators"],
        tuning_params["learning_rate"],
        tuning_params["max_depth"]
    )

    tuning_run_ids = []
    for n_estimators, learning_rate, max_depth in combinations:
        params = base_params.copy()
        params.update({
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth
        })
        with mlflow.start_run(
            run_name=f"xgb_tuning_n{n_estimators}_lr{learning_rate}_d{max_depth}"
        ) as run:
            tuning_run_ids.append(run.info.run_id)
            mlflow.log_params(run_params)
            time_series_cv_xgboost_delta(
                X_train, y_train, params, n_splits=n_splits, gap=gap
            )

    run_ids = ", ".join(f"'{run_id}'" for run_id in tuning_run_ids)
    if not tuning_run_ids:
        raise RuntimeError("The XGBoost tuning grid contains no configurations.")
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_id IN ({run_ids}) AND attributes.status = 'FINISHED'",
        order_by=["metrics.cv_rmse ASC", "attributes.start_time ASC"],
    )
    if runs.empty or "metrics.cv_rmse" not in runs.columns:
        raise RuntimeError("No completed tuning runs with CV RMSE were found.")
    runs = runs[np.isfinite(runs["metrics.cv_rmse"].astype(float))]
    if runs.empty:
        raise RuntimeError("No tuning runs have a finite CV RMSE.")

    print("\nMLflow XGBoost Tuning Results")
    print(runs[["run_id", "params.n_estimators", "params.learning_rate",
                "params.max_depth", "metrics.cv_mae", "metrics.cv_mse",
                "metrics.cv_rmse", "metrics.cv_r2"]].to_string(index=False))
    best = runs.iloc[0]
    best_params = base_params.copy()
    best_params.update({
        "n_estimators": int(best["params.n_estimators"]),
        "learning_rate": float(best["params.learning_rate"]),
        "max_depth": int(best["params.max_depth"]),
    })
    print("\n========== PROGRAMMATICALLY-IDENTIFIED BEST XGBOOST MODEL ==========")
    print("Run ID:", best["run_id"])
    print("Selection metric: CV RMSE")
    print(f"Best CV RMSE: {float(best['metrics.cv_rmse']):.4f}")
    print("Best XGBoost Parameters:", best_params)
    print("=================================================================")
    return best_params, runs, best["run_id"]

def save_deployment_model(model, model_name, features, horizon_hours, models_dir="models"):
    """Save the validated selection and its inference metadata at stable paths."""
    # Add portable exporters here when another model family wins validation.
    if model_name != "xgboost":
        raise ValueError(f"No portable deployment exporter for {model_name!r}.")

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "selected_model.json"
    model.save_model(str(model_path))

    metadata = {
        "model_name": model_name,
        "model_format": "xgboost_json",
        "model_file": model_path.name,
        "features": list(features),
        "forecast_horizon_hours": horizon_hours,
        "model_output": "future soil moisture minus current soil moisture",
        "current_soil_feature": SOIL_COL,
        "absolute_prediction": "current soil moisture + model prediction",
    }
    with (models_dir / "deployment_metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    print("Deployment model saved to:", model_path)

def load_config(config_path="configs/config.yaml"):
    """Load project configuration from YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# FOR MLFLOW: test_r2 is the metric to evaluate
def main():
    config = load_config()
    experiment = mlflow.set_experiment("predictive-ag-ml-assistant")
    df_model = load_and_preprocess_data(
        config["data"]["path"]
    )

    df_model = add_time_series_features(df_model)

    df_model = create_target(df_model)
    df_model = df_model.dropna()

    X_train, X_test, y_train, y_test = split_data(
        df_model,
        train_size=config["split"]["train_size"],
        gap=config["split"]["gap_hours"]
    )

    run_params = {
        "data_description": "Home Assistant sensor history; hourly ginger soil moisture forecasting",
        "data_path": config["data"]["path"],
        "data_sha256": sha256(Path(config["data"]["path"]).read_bytes()).hexdigest(),
        "features": list(X_train.columns),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_start": str(X_train.index.min()),
        "train_end": str(X_train.index.max()),
        "test_start": str(X_test.index.min()),
        "test_end": str(X_test.index.max()),
        "model_output": "future soil moisture minus current soil moisture",
        "absolute_prediction": f"current {SOIL_COL} + model prediction",
        "metric_space": "absolute soil moisture",
        "preprocessing": "hourly past-only alignment, forward fill, rainfall differences, wind sin/cos",
        "soil_delta_lags_rows": [1, 3, 6],
        "soil_rolling_windows_rows": [6, 12],
        "missing_model_rows": "dropped after feature and target construction",
        "forecast_horizon_hours": HORIZON_HOURS,
        "hourly_tolerance": HOURLY_TOLERANCE,
        "train_size": config["split"]["train_size"],
        "split_gap_rows": config["split"]["gap_hours"],
        "gap_units": "split and CV gaps count rows of the model-ready hourly data",
    }

    print("Train:", X_train.index.min(), "→", X_train.index.max())
    print("Test: ", X_test.index.min(), "→", X_test.index.max())

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("Train temp mean:", X_train[TEMP_COL].mean())
    print("Test temp mean:", X_test[TEMP_COL].mean())

    baseline_pred = X_test[SOIL_COL]
    baseline_metrics = evaluate_model(y_test, baseline_pred)
    print("\nPersistence Baseline")

    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nSoil Moisture Distribution")

    print(
        "Train:",
        X_train[SOIL_COL].min(),
        X_train[SOIL_COL].mean(),
        X_train[SOIL_COL].max()
    )

    print(
        "Test:",
        X_test[SOIL_COL].min(),
        X_test[SOIL_COL].mean(),
        X_test[SOIL_COL].max()
    )

    print("\nTarget Distribution")

    print(
        "Train:",
        y_train.min(),
        y_train.mean(),
        y_train.max()
    )

    print(
        "Test:",
        y_test.min(),
        y_test.mean(),
        y_test.max()
    )

    # predict 24 hour change in moisture
    y_train_delta = y_train - X_train[SOIL_COL]
    y_test_delta = y_test - X_test[SOIL_COL]
    print("\nXGBoost Delta - Time Series Cross Validation")

    with mlflow.start_run(run_name="xgboost_initial_cv"):
        mlflow.log_params(run_params)
        xgb_delta_cv_metrics = time_series_cv_xgboost_delta(
            X_train,
            y_train,
            params=config["models"]["xgboost"],
            n_splits=config["cross_validation"]["n_splits"],
            gap=config["cross_validation"]["gap_hours"]
        )

    print("\nXGBoost Hyperparameter Tuning")

    best_xgb_params, xgb_tuning_results, best_tuning_run_id = tune_xgboost_delta(
        X_train,
        y_train,
        base_params=config["models"]["xgboost"],
        tuning_params=config["tuning"]["xgboost"],
        run_params=run_params,
        experiment_id=experiment.experiment_id,
        n_splits=config["cross_validation"]["n_splits"],
        gap=config["cross_validation"]["gap_hours"]
    )

    print("\nDelta Target Distribution")
    print("Train:", y_train_delta.min(), y_train_delta.mean(), y_train_delta.max())
    print("Test: ", y_test_delta.min(), y_test_delta.mean(), y_test_delta.max())

    # Random Forest - predict change
    with mlflow.start_run(run_name="random_forest_delta"):
        mlflow.log_params(run_params)
        mlflow.log_metrics({
            f"test_persistence_{name.lower()}": float(value)
            for name, value in baseline_metrics.items()
        })
        rf_model = train_random_forest(
            X_train,
            y_train_delta,
            config["models"]["random_forest"],
        )

        rf_delta_pred = rf_model.predict(X_test)
        rf_pred = X_test[SOIL_COL].to_numpy() + rf_delta_pred

        rf_metrics = evaluate_model(y_test, rf_pred)

        print("\nRandom Forest - Delta Target")
        for metric, value in rf_metrics.items():
            print(f"{metric}: {value:.4f}")

        mlflow.log_params(config["models"]["random_forest"])
        mlflow.sklearn.log_model(rf_model, artifact_path="model")

        mlflow.log_metrics({
            "test_mae": rf_metrics["MAE"],
            "test_mse": rf_metrics["MSE"],
            "test_rmse": rf_metrics["RMSE"],
            "test_r2": rf_metrics["R2"],
        })

    # XGBoost - predict change, get feature importances
    with mlflow.start_run(run_name="xgboost_delta"):
        mlflow.log_params(run_params)
        mlflow.log_metrics({
            f"test_persistence_{name.lower()}": float(value)
            for name, value in baseline_metrics.items()
        })
        xgb_delta_model = train_xgboost(
            X_train,
            y_train_delta,
            best_xgb_params
        )

        # Display which features the trained XGBoost model relied on most.
        feature_importance = pd.Series(
            xgb_delta_model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        print("\nXGBoost Feature Importance")
        print(feature_importance)

        xgb_delta_pred = xgb_delta_model.predict(X_test)

        # Convert predicted change back into absolute soil moisture
        xgb_delta_absolute_pred = (
            X_test[SOIL_COL].to_numpy() + xgb_delta_pred
        )

        xgb_delta_metrics = evaluate_model(
            y_test,
            xgb_delta_absolute_pred
        )

        print("\nXGBoost - Delta Target")
        for metric, value in xgb_delta_metrics.items():
            print(f"{metric}: {value:.4f}")

        mlflow.log_params(best_xgb_params)
        mlflow.log_param("selected_from_run_id", best_tuning_run_id)
        mlflow.log_param("selection_metric", "cv_rmse")
        mlflow.log_metric("selected_cv_rmse", float(xgb_tuning_results.iloc[0]["metrics.cv_rmse"]))
        mlflow.xgboost.log_model(xgb_delta_model, artifact_path="model")
        mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")
        print("Final XGBoost model URI:", f"runs:/{mlflow.active_run().info.run_id}/model")

        mlflow.log_metrics({
            "test_mae": xgb_delta_metrics["MAE"],
            "test_mse": xgb_delta_metrics["MSE"],
            "test_rmse": xgb_delta_metrics["RMSE"],
            "test_r2": xgb_delta_metrics["R2"],
        })


    # MLP - predict change
    with mlflow.start_run(run_name="mlp_delta"):
        mlflow.log_params(run_params)
        mlflow.log_metrics({
            f"test_persistence_{name.lower()}": float(value)
            for name, value in baseline_metrics.items()
        })
        mlp_delta_model, mlp_delta_scaler = train_mlp(
            X_train,
            y_train_delta,
            config["models"]["mlp"]
        )

        X_test_delta_scaled = mlp_delta_scaler.transform(X_test)

        mlp_delta_pred = (
            mlp_delta_model
            .predict(X_test_delta_scaled, verbose=0)
            .flatten()
        )

        mlp_delta_absolute_pred = (
            X_test[SOIL_COL].to_numpy()
            + mlp_delta_pred
        )

        mlp_delta_metrics = evaluate_model(
            y_test,
            mlp_delta_absolute_pred
        )

        print("\nMLP - Delta Target")
        for metric, value in mlp_delta_metrics.items():
            print(f"{metric}: {value:.4f}")

        print(
            "MLP Delta predictions:",
            mlp_delta_absolute_pred.min(),
            mlp_delta_absolute_pred.mean(),
            mlp_delta_absolute_pred.max()
        )

        mlflow.log_params({name: str(value)
                           for name, value in config["models"]["mlp"].items()})
        mlflow.tensorflow.log_model(mlp_delta_model, artifact_path="model")
        mlflow.sklearn.log_model(mlp_delta_scaler, artifact_path="scaler")

        mlflow.log_metrics({
            "test_mae": mlp_delta_metrics["MAE"],
            "test_mse": mlp_delta_metrics["MSE"],
            "test_rmse": mlp_delta_metrics["RMSE"],
            "test_r2": mlp_delta_metrics["R2"],
        })

    # V1 keeps XGBoost
    selected_model = xgb_delta_model
    selected_model_name = "xgboost"
    save_deployment_model(
        selected_model,
        selected_model_name,
        features=X_train.columns,
        horizon_hours=config["forecast"]["horizon_hours"]
    )

if __name__ == "__main__":
    main()
