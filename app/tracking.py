from pathlib import Path
from typing import Any

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DIRECTORY = PROJECT_ROOT / "artifacts" / "mlruns"


def configure_mlflow(experiment_name: str) -> None:
    """Configure local, repository-contained MLflow tracking."""
    TRACKING_DIRECTORY.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_DIRECTORY.resolve().as_uri())
    mlflow.set_experiment(experiment_name)


def log_retrieval_run(
    *,
    experiment_name: str,
    run_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
) -> str:
    """
    Record one retrieval experiment and return its MLflow run ID.

    Parameters describe the dataset and policy.
    Metrics contain only numeric evaluation results.
    """
    configure_mlflow(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                key: str(value)
                for key, value in parameters.items()
            }
        )

        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in metrics.items()
            }
        )

        return run.info.run_id