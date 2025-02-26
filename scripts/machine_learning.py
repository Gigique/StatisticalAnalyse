import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration des chemins
DATA_PATH = "../clean_data/df_final.csv"
OUTPUT_RESULTS_PATH = "../outputs/analyse_Ml"


def create_output_directories():
    # Créer le dossier de sortie si il n'existe pas
    os.makedirs(OUTPUT_RESULTS_PATH, exist_ok=True)


def load_data():
    # Charger les données depuis le fichier CSV
    return pd.read_csv(DATA_PATH)


def definir_classe(taux):
    if taux < 80:
        return 'Faible'
    elif 80 <= taux < 90:
        return 'Moyen'
    elif 90 <= taux < 95:
        return 'Elevé'
    else:
        return 'Très élevé'


def preprocess_data(df):
    # Appliquer la fonction à la colonne 'taux_insertion'
    df['Classe du taux d’insertion'] = df['Taux d’insertion'].apply(definir_classe)

    selected_colonnes = [
        "Domaine_encoded", "Disciplines_encoded", "Discipline_encoded",
        "Secteur disciplinaire_encoded",
        "Part des emplois de niveau cadre ou profession intermédiaire",
        "Part des emplois stables", "Part des emplois à temps plein",
        "Salaire net mensuel médian des emplois à temps plein",
        "Part des femmes", "Part des diplômés boursiers dans la discipline",
        "Taux de chômage national"
    ]

    y = df["Classe du taux d’insertion"]
    X = df[selected_colonnes]

    return X, y


def train_test(X, y):
    # Séparation en deux : entraînement/validation et test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Séparation de l'ensemble entraînement en deux : entraînement et validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val, X, y


def validation_croisee(X, y):
    # Définir les meilleurs paramètres
    params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42
    }

    # Initialiser le modèle avec les meilleurs paramètres
    model = RandomForestClassifier(**params)

    # Validation croisée avec 5 folds
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    return model, cv_scores.mean(), cv_scores.std()


def random_forest(X, y):
    param_grid = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42
    }

    # Initialiser le modèle
    model = RandomForestClassifier(**param_grid)

    # Entraîner le modèle sur l'ensemble d'entraînement
    model.fit(X, y)

    train_predictions = model.predict(X)
    train_accuracy = accuracy_score(y, train_predictions)

    return model, train_accuracy


def evaluation_model(X, y, model):
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X)

    # Calcul de la précision
    accuracy = accuracy_score(y, y_pred)

    # Rapport de classification (précision, rappel, F1-score)
    classification_report_str = classification_report(y, y_pred)

    # Matrice de confusion
    confusion_matrix_str = str(confusion_matrix(y, y_pred))

    return accuracy, classification_report_str, confusion_matrix_str


def apprentissage(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    # Moyenne et écart-type des scores
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, label="Training score")
    plt.plot(train_sizes, test_scores_mean, label="Validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.show()


def gradient_boosting(X, y):
    model = GradientBoostingClassifier(random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    return model, cv_scores.mean(), cv_scores.std()


def save_results_to_file(file, train_accuracy, test_accuracy, cv_score_str, cv_sdt_str, cv_score_gd_str, cv_sdt_gd_str,
                         classification_report_str, confusion_matrix_str):
    with open(file, 'w', encoding='utf-8') as f:
        # Enregistrer les résultats de RandomForest
        f.write("RANDOM FOREST TEST\n")
        f.write("===================================\n\n")
        f.write(f"Accuracy moyenne sur les folds : {cv_score_str}\n")
        f.write(f"Écart-type de l'accuracy : {cv_sdt_str}\n\n")

        # Enregistrer les résultats du Gradient Boosting
        f.write("GRADIENT BOOSTING TEST\n")
        f.write("===================================\n\n")
        f.write(f"Accuracy moyenne sur les folds : {cv_score_gd_str}\n")
        f.write(f"Écart-type de l'accuracy : {cv_sdt_gd_str}\n\n")

        # Enregistrer l'accuracy d'entraînement et de test
        f.write("COMPARAISON ACCURACY SUR LE TRAIN ET LE TEST\n")
        f.write("===================================\n\n")
        f.write(f"Accuracy sur l'entraînement : {train_accuracy}\n")
        f.write(f"Accuracy sur le test : {test_accuracy}\n\n")

        # Enregistrer le classification report
        f.write("RAPPORT DE CLASSIFICATION\n")
        f.write("===================================\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report_str)
        f.write("\n")

        # Enregistrer la matrice de confusion
        f.write("MATRICE DE CONFUSION\n")
        f.write("===================================\n\n")
        f.write("Confusion Matrix:\n")
        f.write(confusion_matrix_str)
        f.write("\n")


def main():
    try:
        create_output_directories()

        # Chargement des données
        print("Chargement des données...")
        df = load_data()

        # Prétraitement des données
        print("Prétraitement des données...")
        X, y = preprocess_data(df)

        # Séparation des données en train, test et validation
        print("Séparation des données...")
        X_train, X_test, X_val, y_train, y_test, y_val, X, y = train_test(X, y)

        # Validation croisée pour évaluer le modèle RandomForest
        print("Validation croisée pour Random Forest...")
        model_rf, cv_score_rf, cv_sdt_rf = validation_croisee(X_train, y_train)

        # Validation croisée pour évaluer le modèle Gradient Boosting
        print("Validation croisée pour Gradient Boosting...")
        model_gb, cv_score_gb, cv_sdt_gb = gradient_boosting(X_train, y_train)

        # Entraîner le modèle RandomForestClassifier
        print("Entraînement du modèle Random Forest...")
        model_rf, train_accuracy = random_forest(X_val, y_val)

        # Courbe d'apprentissage
        print("Affichage de la courbe d'apprentissage...")
        apprentissage(model_rf, X_val, y_val)

        # Évaluation du modèle sur l'ensemble de test
        print("Évaluation du modèle...")
        test_accuracy, classification_report_str, confusion_matrix_str = evaluation_model(X_test, y_test, model_rf)

        # Sauvegarde des résultats
        print("Sauvegarde des résultats...")
        results_file = os.path.join(OUTPUT_RESULTS_PATH, "model_ML_results.txt")
        save_results_to_file(results_file, train_accuracy, test_accuracy,
                             str(cv_score_rf), str(cv_sdt_rf),
                             str(cv_score_gb), str(cv_sdt_gb),
                             classification_report_str, confusion_matrix_str)

        print("Analyse terminée. Consultez les résultats dans 'model_ML_results.txt'.")

    except Exception as e:
        print(f"Erreur : {e}")


if __name__ == "__main__":
    main()
