import pandas as pd
import statsmodels.api as sm
import numpy as np
import os

# Définition des chemins
DATA_PATH = "../clean_data/df_final.csv"
OUTPUT_RESULTS_PATH = "../outputs/reglog"  # Dossier pour sauvegarder tous les résultats


def create_output_directories():
    # Créer le dossier de sortie si il n'existe pas
    os.makedirs(OUTPUT_RESULTS_PATH, exist_ok=True)


def load_data():
    # Charger les données depuis le fichier CSV
    return pd.read_csv(DATA_PATH)


def preprocess_data(df):
    # Définir la variable cible et les variables explicatives
    y = df["Taux d’insertion"]
    X = df[
        ['Part des emplois stables', 'Part des emplois de niveau cadre ou profession intermédiaire', 'Part des femmes',
         'Part des diplômés boursiers dans la discipline', 'Domaine_encoded', 'Taux de chômage national']]

    # Application du seuil de 89 % pour la binarisation de y
    seuil = 89
    y_binaire = (y >= seuil).astype(int)  # Binarisation

    # Ajouter une constante à X pour l'intercept
    X_modified = sm.add_constant(X)

    return X_modified, y_binaire


def fit_logistic_regression(X_modified, y_binaire):
    # Créer et ajuster le modèle de régression logistique
    logit_model = sm.Logit(y_binaire, X_modified)
    result = logit_model.fit()
    return result


def calculate_odds_ratios(result):
    # Calcul des Odds Ratios (OR) en exponentiant les coefficients
    odds_ratios = result.params.apply(lambda x: round(np.exp(x), 3))
    return odds_ratios


def calculate_confidence_intervals(result):
    # Calcul des intervalles de confiance à 95% pour les coefficients
    conf_int = result.conf_int()
    # Exponentiation des intervalles pour obtenir les IC des OR
    conf_int_or = conf_int.apply(lambda x: round(np.exp(x), 3))
    return conf_int_or


def save_results(result, odds_ratios, conf_int_or):
    # Sauvegarder tous les résultats dans un seul fichier texte
    result_file = os.path.join(OUTPUT_RESULTS_PATH, 'regression_logistique_results.txt')

    with open(result_file, 'w', encoding='utf-8') as f:
        # Résumé du modèle
        f.write("Résumé du modèle de régression logistique\n")
        f.write("===============================\n\n")
        f.write(str(result.summary()))
        f.write("\n\n")

        # Odds Ratios
        f.write("Odds Ratios\n")
        f.write("===========\n\n")
        f.write(odds_ratios.to_string())
        f.write("\n\n")

        # Intervalles de confiance pour les Odds Ratios
        f.write("Intervalle de confiance à 95% pour les Odds Ratios\n")
        f.write("===================================================\n\n")
        f.write(conf_int_or.to_string())
        f.write("\n\n")


def main():
    try:
        create_output_directories()

        # Chargement des données
        print("Chargement des données...")
        df = load_data()

        # Prétraitement des données
        print("Prétraitement des données...")
        X_modified, y_binaire = preprocess_data(df)

        # Régression logistique
        print("Fitting du modèle de régression logistique...")
        result = fit_logistic_regression(X_modified, y_binaire)

        # Calcul des Odds Ratios et des intervalles de confiance
        print("Calcul des Odds Ratios et des intervalles de confiance...")
        odds_ratios = calculate_odds_ratios(result)
        conf_int_or = calculate_confidence_intervals(result)

        # Sauvegarde des résultats
        print("Sauvegarde des résultats...")
        save_results(result, odds_ratios, conf_int_or)

        print(
            "\nAnalyse terminée. Consultez les résultats dans le fichier 'regression_logistique_results.txt' dans le dossier 'outputs/reglog'.")

    except Exception as e:
        print(f"Erreur : {e}")


if __name__ == "__main__":
    main()






