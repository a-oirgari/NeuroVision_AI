import mlflow
import mlflow.keras
from mlflow import MlflowClient

from src import config


def setup_mlflow():
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def _experiment_id():
    return mlflow.get_experiment_by_name(config.EXPERIMENT_NAME).experiment_id


def compare_runs():
    df = mlflow.search_runs(
        experiment_ids=[_experiment_id()],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.best_val_f1_macro DESC"],
    )
    cols = [
        "tags.mlflow.runName",
        "params.learning_rate",
        "params.trainable_params",
        "metrics.best_val_accuracy",
        "metrics.best_val_f1_macro",
        "metrics.best_val_loss",
        "metrics.duration_min",
    ]
    return df[[c for c in cols if c in df.columns]]


def get_run_model_uri(run_name):
    runs = mlflow.search_runs(
        experiment_ids=[_experiment_id()],
        filter_string=f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"Aucun run terminé nommé '{run_name}'.")
    return runs.iloc[0]["tags.model_uri"]


def promote_champion(metric="best_val_f1_macro"):
    runs = mlflow.search_runs(
        experiment_ids=[_experiment_id()],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError("Aucun run terminé.")
    best = runs.iloc[0]

    version = mlflow.register_model(best["tags.model_uri"], config.REGISTERED_MODEL)
    MlflowClient().set_registered_model_alias(
        config.REGISTERED_MODEL, config.CHAMPION_ALIAS, version.version)
    print(f"Champion : run '{best['tags.mlflow.runName']}' -> "
          f"{config.REGISTERED_MODEL} v{version.version} (alias '{config.CHAMPION_ALIAS}')")
    return best