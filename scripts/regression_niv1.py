import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, normaltest, skew, kurtosis, kstest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_goldfeldquandt
from statsmodels.stats.stattools import durbin_watson
import os

# Configuration des chemins
DATA_PATH = "../clean_data/df_final.csv"
OUTPUT_FIGURES_PATH = "../outputs/figures"
OUTPUT_RESULTS_PATH = "../outputs/results"

def convert_to_float(df, column):
    """Convertit une colonne en float en gérant les virgules"""
    if df[column].dtype == 'object':
        return df[column].str.replace(',', '.').astype(float)
    return df[column].astype(float)

def prepare_data(df):
    """Prépare les données pour la régression"""
    selected_features = [
    "Part des emplois stables",
    "Part des emplois de niveau cadre ou profession intermédiaire",
    "Part des femmes",
    "Part des diplômés boursiers dans la discipline",
    "Domaine_encoded",
    "Taux de chômage national"
    ]


    # Conversion des colonnes en float
    for col in selected_features:
        df[col] = convert_to_float(df, col)

    # Conversion de la variable cible
    df["Taux d’insertion"] = convert_to_float(df, "Taux d’insertion")

    # Préparation des variables X et y
    y = df["Taux d’insertion"]
    X = df[selected_features]

    # Vérification des types
    print("\nTypes des variables après conversion :")
    print(X.dtypes)
    print(f"Type de y : {y.dtype}")

    # Vérification des NaN
    print("\nVérification des valeurs manquantes :")
    print(X.isna().sum())
    print(f"NaN dans y : {y.isna().sum()}")

    # Ajout de la constante
    X = sm.add_constant(X)

    return X, y

def calculate_vif(X):
    """Calcule le VIF pour chaque variable"""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def create_output_dirs():
    """Crée les répertoires de sortie s'ils n'existent pas"""
    for path in [OUTPUT_FIGURES_PATH, OUTPUT_RESULTS_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"✓ Répertoire créé : {path}")
        else:
            print(f"✓ Répertoire existant : {path}")

def save_correlation_plots(df, X, y):
    """Crée et sauve les graphiques de corrélation"""
    try:
        # Matrice de corrélation complète
        plt.figure(figsize=(12, 10))
        correlation_matrix_selected = pd.concat([X, pd.Series(y, name="Taux d'insertion")], axis=1).corr()
        sns.heatmap(correlation_matrix_selected, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title("Matrice de corrélation")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIGURES_PATH, "correlation_matrix_selected.png"), dpi=300)
        plt.close()
        print("✓ Matrice de corrélation sauvegardée")
    except Exception as e:
        print(f"Erreur lors de la création de la matrice de corrélation : {str(e)}")

def analyze_model(X, y):
    """Analyse complète du modèle"""
    try:
        # Ajustement du modèle
        model = sm.OLS(y, X).fit()
        y_pred = model.predict(X)
        residuals = y - y_pred
        standardized_residuals = residuals / np.std(residuals)

        # Création des graphiques
        print("\nCréation des graphiques...")

        # 1. Distribution des résidus
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title("Distribution des résidus_niv1")
        plt.savefig(os.path.join(OUTPUT_FIGURES_PATH, "residuals_distribution_niv1.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print("✓ Distribution des résidus sauvegardée")

        # 2. Q-Q Plot
        plt.figure(figsize=(10, 6))
        sm.graphics.qqplot(residuals, line='45')
        plt.title("Q-Q Plot des résidus_niv1")
        plt.savefig(os.path.join(OUTPUT_FIGURES_PATH, "qq_plot_niv1.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print("✓ Q-Q Plot sauvegardé")

        # 3. Résidus vs Valeurs prédites
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Valeurs prédites")
        plt.ylabel("Résidus")
        plt.title("Résidus vs Valeurs prédites_niv1")
        plt.savefig(os.path.join(OUTPUT_FIGURES_PATH, "residuals_vs_predicted_niv1.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print("✓ Graphique des résidus sauvegardé")

        # Sauvegarde des résultats
        results_file = os.path.join(OUTPUT_RESULTS_PATH, "model_analysis_niv1.txt")
        with open(results_file, "w", encoding="utf-8") as f:
            f.write("RÉSULTATS DE L'ANALYSE DE RÉGRESSION\n")
            f.write("===================================\n\n")

            # Performance du modèle
            f.write("1. PERFORMANCE DU MODÈLE\n")
            f.write("-----------------------\n")
            f.write(f"R² : {model.rsquared:.4f}\n")
            f.write(f"R² ajusté : {model.rsquared_adj:.4f}\n")
            f.write(f"F-statistic : {model.fvalue:.4f} (p-value : {model.f_pvalue:.4f})\n\n")

            # Coefficients
            f.write("2. COEFFICIENTS DU MODÈLE\n")
            f.write("------------------------\n")
            f.write(model.summary().tables[1].as_text())
            f.write("\n\n")

            # Tests statistiques
            f.write("3. TESTS STATISTIQUES\n")
            f.write("--------------------\n")

            # Normalité
            shapiro_stat, shapiro_p = shapiro(residuals)
            f.write(f"Test de Shapiro-Wilk : p-value = {shapiro_p:.4f}\n")
            f.write(f"Skewness : {skew(residuals):.4f}\n")
            f.write(f"Kurtosis : {kurtosis(residuals):.4f}\n\n")

            # Homoscédasticité
            try:
                gq_result = het_goldfeldquandt(residuals, X)
                f.write(f"Test de Goldfeld-Quandt : F-stat = {gq_result[0]:.4f}, p-value = {gq_result[1]:.4f}\n")
            except Exception as e:
                f.write(f"Test de Goldfeld-Quandt : Erreur lors du calcul\n")

            # Autocorrélation
            dw_stat = durbin_watson(residuals)
            f.write(f"Statistique de Durbin-Watson : {dw_stat:.4f}\n\n")

            # Multicolinéarité
            f.write("4. ANALYSE DE LA MULTICOLINÉARITÉ (VIF)\n")
            f.write("------------------------------------\n")
            vif_data = calculate_vif(X)
            f.write(vif_data.to_string())
            f.write("\n\n")

            # Statistiques descriptives
            f.write("5. STATISTIQUES DESCRIPTIVES\n")
            f.write("---------------------------\n")
            f.write("\nVariables explicatives :\n")
            f.write(X.describe().to_string())
            f.write("\n\nVariable à expliquer :\n")
            f.write(pd.Series(y).describe().to_string())

        print(f"✓ Résultats sauvegardés : {results_file}")
        return model

    except Exception as e:
        print(f"Erreur lors de l'analyse du modèle : {str(e)}")
        raise

def main():
    try:
        # Création des répertoires
        create_output_dirs()

        # Chargement des données
        print("\nChargement des données...")
        df = pd.read_csv(DATA_PATH)
        print("✓ Données chargées avec succès")

        # Préparation des données
        print("\nPréparation des données...")
        X, y = prepare_data(df)
        print(f"✓ Données préparées (X: {X.shape}, y: {y.shape})")

        # Création des graphiques de corrélation
        print("\nCréation des graphiques de corrélation...")
        save_correlation_plots(df, X, y)
        print("✓ Graphiques de corrélation sauvegardés")

        # Analyse du modèle
        print("\nAnalyse du modèle...")
        model = analyze_model(X, y)
        print("✓ Analyse du modèle terminée")

        # Vérification finale
        print("\nVérification des fichiers générés :")
        for path in [OUTPUT_FIGURES_PATH, OUTPUT_RESULTS_PATH]:
            print(f"\nFichiers dans {path}:")
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"- {file} ({size:.1f} KB)")

    except Exception as e:
        print(f"\nErreur : {str(e)}")
        if 'df' in locals():
            print("\nColonnes disponibles :")
            print(df.columns.tolist())
        raise

if __name__ == "__main__":
    main()