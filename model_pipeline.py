"""
model_pipeline.py
------------------
Pipeline modulaire pour la prédiction du churn client.
Contient les fonctions réutilisables : prepare_data, train_model,
evaluate_model, save_model, load_model.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 1):
    """
    Charge le CSV, nettoie et prétraite les données, puis retourne
    les jeux d'entraînement et de test.

    Étapes reproduites depuis le notebook :
      - suppression des colonnes non prédictives (Surname, Geography,
        RowNumber, CustomerId)
      - encodage de la colonne Gender (LabelEncoder)
      - séparation features (X) / cible (y = Exited)
      - split train/test

    Args:
        filepath: chemin vers le fichier Churn_Modelling.csv
        test_size: proportion du jeu de test
        random_state: graine aléatoire pour la reproductibilité

    Returns:
        x_train, x_test, y_train, y_test
    """
    df = pd.read_csv(filepath)

    # Encodage de la variable catégorielle Gender
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Colonnes non pertinentes pour l'entraînement
    columns_to_drop = ["Surname", "Geography"]
    df = df.drop(columns_to_drop, axis=1)

    # Séparation features / cible
    x = df.drop(["Exited"], axis=1)
    y = df["Exited"]

    # Identifiants sans valeur prédictive
    x = x.drop(columns=["RowNumber", "CustomerId"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Entraîne un RandomForestClassifier sur les données d'entraînement.

    Args:
        x_train: features d'entraînement
        y_train: cible d'entraînement
        n_estimators: nombre d'arbres de la forêt
        random_state: graine aléatoire

    Returns:
        Le modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test) -> dict:
    """
    Évalue les performances du modèle sur le jeu de test.

    Args:
        model: modèle entraîné
        x_test: features de test
        y_test: cible de test

    Returns:
        Un dictionnaire contenant accuracy, matrice de confusion
        et rapport de classification.
    """
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy score: {accuracy * 100:.2f}%")
    print("Confusion matrix:")
    print(matrix)
    print("Classification report:")
    print(report)

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "classification_report": report,
    }


def save_model(model, filepath: str = "classifier.joblib") -> None:
    """
    Sauvegarde le modèle entraîné sur disque avec joblib.

    Args:
        model: modèle à sauvegarder
        filepath: chemin du fichier de sortie
    """
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé dans : {filepath}")


def load_model(filepath: str = "classifier.joblib"):
    """
    Charge un modèle précédemment sauvegardé.

    Args:
        filepath: chemin du fichier .joblib

    Returns:
        Le modèle chargé.
    """
    model = joblib.load(filepath)
    print(f"Modèle chargé depuis : {filepath}")
    return model
