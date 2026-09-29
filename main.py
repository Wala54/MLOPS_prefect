"""
main.py
-------
Point d'entrée pour exécuter le pipeline de churn via la ligne de commande.

Exemples d'utilisation :
    python main.py --mode train --data Churn_Modelling.csv --model classifier.joblib
    python main.py --mode evaluate --data Churn_Modelling.csv --model classifier.joblib
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de prédiction du churn client."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "train_evaluate"],
        default="train_evaluate",
        help="Étape à exécuter : train (entraîner + sauvegarder), "
        "evaluate (charger + évaluer), train_evaluate (les deux).",
    )
    parser.add_argument(
        "--data",
        default="Churn_Modelling.csv",
        help="Chemin vers le fichier CSV de données.",
    )
    parser.add_argument(
        "--model",
        default="classifier.joblib",
        help="Chemin du fichier modèle (.joblib) à sauvegarder ou charger.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Nombre d'arbres pour le RandomForestClassifier.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Étape 1 : préparation des données (toujours nécessaire pour avoir x_test/y_test)
    x_train, x_test, y_train, y_test = prepare_data(args.data)

    if args.mode == "train":
        model = train_model(x_train, y_train, n_estimators=args.n_estimators)
        save_model(model, args.model)

    elif args.mode == "evaluate":
        model = load_model(args.model)
        evaluate_model(model, x_test, y_test)

    elif args.mode == "train_evaluate":
        model = train_model(x_train, y_train, n_estimators=args.n_estimators)
        evaluate_model(model, x_test, y_test)
        save_model(model, args.model)


if __name__ == "__main__":
    main()
