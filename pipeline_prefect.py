"""
pipeline_prefect.py
-------------------
Orchestration Prefect du pipeline de churn.

Reprend les fonctions de model_pipeline.py (déjà testées et modularisées)
et les transforme en tasks/flows Prefect.
"""

import argparse
import subprocess  # nosec B404
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

DATA_FILE = Path("Churn_Modelling.csv")
MODEL_FILE = Path("classifier.joblib")

# Les outils qualité/sécurité ne scannent que ces 3 fichiers
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_cmd(cmd):
    """Lance une commande, affiche la sortie, échoue si le code retour != 0."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
    if result.stdout:
        logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr or result.stdout)
        raise RuntimeError("Échec de la commande : " + " ".join(cmd))


# ---------------------------------------------------------------
# Tasks : suivi du code
# ---------------------------------------------------------------
@task(name="install-dependencies")
def install_dependencies_task():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="format-code")
def format_code_task():
    run_cmd([sys.executable, "-m", "black", *CODE_FILES])


@task(name="code-quality")
def code_quality_task():
    run_cmd([sys.executable, "-m", "flake8", "--max-line-length=100", *CODE_FILES])


@task(name="code-security")
def code_security_task():
    run_cmd([sys.executable, "-m", "bandit", "-q", *CODE_FILES])


@task(name="unit-tests")
def unit_tests_task():
    run_cmd([sys.executable, "-m", "pytest", "tests/", "-v"])


# ---------------------------------------------------------------
# Tasks : data / modèle
# ---------------------------------------------------------------
@task(name="prepare-data", retries=1, retry_delay_seconds=5)
def prepare_data_task(data_file: str = str(DATA_FILE)):
    logger = get_run_logger()
    x_train, x_test, y_train, y_test = prepare_data(data_file)
    logger.info(
        f"Données préparées : {len(x_train)} lignes train / {len(x_test)} lignes test."
    )
    return x_train, x_test, y_train, y_test


@task(name="train-model")
def train_model_task(x_train, y_train):
    logger = get_run_logger()
    model = train_model(x_train, y_train)
    logger.info("Modèle entraîné avec succès.")
    return model


@task(name="save-model")
def save_model_task(model, model_file: str = str(MODEL_FILE)):
    logger = get_run_logger()
    save_model(model, model_file)
    logger.info(f"Modèle sauvegardé : {model_file}")


@task(name="load-model")
def load_model_task(model_file: str = str(MODEL_FILE)):
    logger = get_run_logger()
    model = load_model(model_file)
    logger.info(f"Modèle chargé : {model_file}")
    return model


@task(name="evaluate-model")
def evaluate_model_task(model, x_test, y_test):
    logger = get_run_logger()
    metrics = evaluate_model(model, x_test, y_test)
    logger.info(f"Résultats de l'évaluation : {metrics}")
    return metrics


# ---------------------------------------------------------------
# Flows
# ---------------------------------------------------------------
@flow(name="code")
def code_flow():
    install_dependencies_task()
    format_code_task()
    code_quality_task()
    code_security_task()
    unit_tests_task()


@flow(name="train")
def train_flow():
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)


@flow(name="evaluate")
def evaluate_flow():
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = load_model_task()
    evaluate_model_task(model, x_test, y_test)


@flow(name="all")
def all_flow():
    install_dependencies_task()
    format_code_task()
    code_quality_task()
    code_security_task()
    unit_tests_task()
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)
    evaluate_model_task(model, x_test, y_test)


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "evaluate": evaluate_flow,
    "code": code_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=list(FLOWS), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
