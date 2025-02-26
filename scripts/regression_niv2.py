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

def create_output_dirs():
    """Crée les répertoires de sortie s'ils n'existent pas"""
    for path in [OUTPUT_FIGURES_PATH, OUTPUT_RESULTS_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"✓ Répertoire créé : {path}")
        else:
            print(f"✓ Répertoire existant : {path}")

def convert_to_float(df, column):
    """Convertit une colonne en float en gérant les virgules"""
    if df[column].dtype == 'object':
        return df[column].str.replace(',', '.').astype(float)
    return df[column].astype(float)

def prepare_data(df):
    """Prépare les données pour la régression avec traitement amélioré"""
    selected_features = [
    "Part des emplois stables",
    "Part des emplois de niveau cadre ou profession intermédiaire",
    "Part des femmes",
    "Part des diplômés boursiers dans la discipline",
    "Domaine_encoded",
    "Taux de chômage national"
    ]

    # 1. Conversion des colonnes en float
    for col in selected_features + ["Taux d’insertion"]:
        df[col] = convert_to_float(df, col)

    # Sauvegarde des statistiques avant transformation
    stats_before = df[selected_features + ["Taux d’insertion"]].describe()

    # 2. Traitement des outliers avec IQR plus strict
    df_clean = df.copy()
    outliers_summary = {}
    for col in selected_features + ["Taux d’insertion"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Compte des outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outliers_summary[col] = len(outliers)

        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)



    # 3. Standardisation robuste
    for col in selected_features:
        median = df_clean[col].median()
        mad = np.median(np.abs(df_clean[col] - median))
        df_clean[col] = (df_clean[col] - median) / (1.4826 * mad)

    # Sauvegarde des statistiques après transformation
    stats_after = df_clean[selected_features + ["Taux d’insertion"]].describe()

    # 4. Préparation finale
    y = df_clean["Taux d’insertion"]
    X = df_clean[selected_features]
    X = sm.add_constant(X)

    return X, y, stats_before, stats_after, outliers_summary

def calculate_vif(X):
    """Calcule le VIF pour chaque variable"""
    variable_names = ['const'] + [
    "Part des emplois stables",
    "Part des emplois de niveau cadre ou profession intermédiaire",
    "Part des femmes",
    "Part des diplômés boursiers dans la discipline",
    "Domaine_encoded",
    "Taux de chômage national"
    ]

    vif_data = pd.DataFrame()
    vif_data["Variable"] = variable_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return vif_data

def analyze_model(X, y):
    """Analyse améliorée du modèle"""
    model = sm.OLS(y, X).fit()

    # Calcul des résidus studentisés
    influence = model.get_influence()
    studentized_residuals = influence.resid_studentized_internal

    # Création des graphiques de diagnostic
    fig = plt.figure(figsize=(15, 15))

    # 1. Q-Q Plot avec intervalles de confiance
    ax1 = fig.add_subplot(221)
    sm.graphics.qqplot(studentized_residuals, line='45', fit=True, ax=ax1)
    ax1.set_title("Q-Q Plot des résidus studentisés_niv2")

    # 2. Distribution des résidus
    ax2 = fig.add_subplot(222)
    sns.histplot(studentized_residuals, kde=True, ax=ax2)
    ax2.set_title("Distribution des résidus studentisés_niv2")

    # 3. Résidus vs Valeurs prédites
    ax3 = fig.add_subplot(223)
    ax3.scatter(model.fittedvalues, studentized_residuals, alpha=0.5)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel("Valeurs prédites")
    ax3.set_ylabel("Résidus studentisés")
    ax3.set_title("Résidus vs Valeurs prédites_niv2")

    # 4. Scale-Location Plot
    ax4 = fig.add_subplot(224)
    sqrt_abs_resid = np.sqrt(np.abs(studentized_residuals))
    ax4.scatter(model.fittedvalues, sqrt_abs_resid, alpha=0.5)
    ax4.set_xlabel("Valeurs prédites")
    ax4.set_ylabel("√|Résidus studentisés|")
    ax4.set_title("Scale-Location Plot_niv2")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES_PATH, "diagnostic_plots_niv2.png"))
    plt.close()

    # Tests statistiques supplémentaires
    shapiro_stat, shapiro_p = shapiro(studentized_residuals)
    skewness = skew(studentized_residuals)
    kurt = kurtosis(studentized_residuals)

    return model, shapiro_p, skewness, kurt

def save_results(model, stats_before, stats_after, outliers_summary, shapiro_p, skewness, kurt):
    """Sauvegarde tous les résultats dans un fichier"""
    with open(os.path.join(OUTPUT_RESULTS_PATH, "regression_results_niv2.txt"), 'w', encoding='utf-8') as f:
        f.write("ANALYSE DE RÉGRESSION AMÉLIORÉE\n")
        f.write("==============================\n\n")

        f.write("1. STATISTIQUES DESCRIPTIVES\n")
        f.write("---------------------------\n")
        f.write("\nAvant transformation:\n")
        f.write(stats_before.to_string())
        f.write("\n\nAprès transformation:\n")
        f.write(stats_after.to_string())

        f.write("\n\n2. OUTLIERS DÉTECTÉS ET TRAITÉS\n")
        f.write("------------------------------\n")
        for col, count in outliers_summary.items():
            f.write(f"{col}: {count} outliers traités\n")

        f.write("\n3. RÉSULTATS DU MODÈLE\n")
        f.write("---------------------\n")
        f.write(model.summary().as_text())

        f.write("\n\n4. TESTS DE DIAGNOSTIC\n")
        f.write("---------------------\n")
        f.write(f"Test de Shapiro-Wilk : p-value = {shapiro_p:.4f}\n")
        f.write(f"Skewness : {skewness:.4f}\n")
        f.write(f"Kurtosis : {kurt:.4f}\n")

        f.write("\n5. FACTEURS D'INFLATION DE LA VARIANCE (VIF)\n")
        f.write("----------------------------------------\n")
        vif_data = calculate_vif(model.model.exog)
        f.write(vif_data.to_string())


def check_linearity(X, y) -> bool:
    # Ajuster le modèle (fit)
    model = sm.OLS(y, X)
    results = model.fit()

    # Calculer les valeurs prédites et les valeurs observées
    y_observed = model.endog  # Valeurs observées (Y)
    y_predicted = results.fittedvalues  # Valeurs prédites (Y_hat)

    # Tracer les valeurs observées contre les valeurs prédites
    fig = plt.figure(figsize=(15, 10))
    plt.scatter(y_observed, y_predicted)
    plt.plot([min(y_predicted), max(y_predicted)], [min(y_predicted), max(y_predicted)], color='b',
             linestyle='--')  # Tracer une droite en pointillée
    # Droite en pointillée : représente le cas ou Y observée = Y prédit
    plt.xlabel('Valeurs observées (Y)')
    plt.ylabel('Valeurs prédites (Y_chapeau)')
    plt.title('Graphique Y observées vs. Y prédites')
    plt.savefig(os.path.join(OUTPUT_FIGURES_PATH, "linearity_plot_niv2.png"))





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
        X, y, stats_before, stats_after, outliers_summary = prepare_data(df)
        print("✓ Données préparées")

        # Analyse du modèle
        print("\nAnalyse du modèle...")
        model, shapiro_p, skewness, kurt = analyze_model(X, y)
        print("✓ Analyse du modèle terminée")


        print("condition de linéarité...")
        check_linearity(X, y)

        # Sauvegarde des résultats
        print("\nSauvegarde des résultats...")
        save_results(model, stats_before, stats_after, outliers_summary,
                    shapiro_p, skewness, kurt)
        print("✓ Résultats sauvegardés")




        # Affichage des métriques principales
        print("\nMétriques principales :")
        print(f"R² ajusté : {model.rsquared_adj:.4f}")
        print(f"Test de Shapiro-Wilk (p-value) : {shapiro_p:.4f}")
        print(f"Nombre d'observations : {len(y)}")

    except Exception as e:
        print(f"\nErreur : {str(e)}")
        raise

if __name__ == "__main__":
    main()