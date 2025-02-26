import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Définition des chemins
DATA_PATH = "../clean_data/df_final.csv"
OUTPUT_FIGURES_PATH = "../outputs/figures"
OUTPUT_RESULTS_PATH = "../outputs/results"

def create_output_directories():
    os.makedirs(OUTPUT_FIGURES_PATH, exist_ok=True)
    os.makedirs(OUTPUT_RESULTS_PATH, exist_ok=True)

def analyze_descriptive_stats(df):
    return df.describe()


def plot_correlation_analysis(df):
    # Calcul des corrélations
    correlations = df.corr()["Taux d’insertion"].sort_values(ascending=False)

    plt.figure(figsize=(15, 12))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Matrice de corrélation globale')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES_PATH, 'correlation_matrix_global.png'))
    plt.close()

    return correlations

def plot_correlation_with_regression(df):
    # Calcul des corrélations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df.corr()["Taux d’insertion"].sort_values(ascending=False)

    n_cols = 2
    n_vars = len(numeric_cols) - 1
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.ravel()

    plot_idx = 0
    for col in numeric_cols:
        if col != "Taux d’insertion":
            sns.regplot(data=df,
                       x=col,
                       y="Taux d’insertion",
                       ax=axes[plot_idx],
                       scatter_kws={'alpha':0.5},
                       line_kws={'color': 'red'})

            corr_value = correlations[col]
            axes[plot_idx].set_title(f'{col}\nCorrélation: {corr_value:.3f}')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            plot_idx += 1

    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES_PATH, 'correlation_with_regression_globale.png'))
    plt.close()

def plot_distributions(df):
    # Distribution du taux d'insertion
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Taux d’insertion", kde=True)
    plt.title('Distribution du taux d\'insertion')
    plt.savefig(os.path.join(OUTPUT_FIGURES_PATH, 'taux_insertion_distribution.png'))
    plt.close()

    # Distribution des variables numériques avec boxplots
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 2
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    # Histogrammes avec KDE
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.ravel()

    for idx, col in enumerate(numeric_cols):
        # Histogramme avec KDE
        sns.histplot(data=df, x=col, kde=True, ax=axes[idx])
        axes[idx].set_title(f'Distribution de {col}')
        axes[idx].tick_params(axis='x', rotation=45)

    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES_PATH, 'distributions_histograms_globale.png'))
    plt.close()

    # Boxplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.ravel()

    for idx, col in enumerate(numeric_cols):
        # Boxplot
        sns.boxplot(data=df, y=col, ax=axes[idx])
        axes[idx].set_title(f'Boxplot de {col}')
        axes[idx].tick_params(axis='x', rotation=45)

    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES_PATH, 'distributions_boxplots_globale.png'))
    plt.close()

def save_results(desc_stats, correlations):
    with open(os.path.join(OUTPUT_RESULTS_PATH, 'analyse_statistique_globale.txt'), 'w', encoding='utf-8') as f:
        f.write("ANALYSE STATISTIQUE GLOBALE\n")
        f.write("==========================\n\n")

        f.write("1. STATISTIQUES DESCRIPTIVES\n")
        f.write("---------------------------\n")
        f.write(desc_stats.to_string())
        f.write("\n\n")

        f.write("2. CORRÉLATIONS AVEC LE TAUX D'INSERTION\n")
        f.write("-------------------------------------\n")
        f.write(correlations.to_string())

def main():
    try:
        create_output_directories()

        print("Chargement des données...")
        df = pd.read_csv(DATA_PATH)

        print("Analyse descriptive...")
        desc_stats = analyze_descriptive_stats(df)

        print("Analyse des corrélations...")
        correlations = plot_correlation_analysis(df)

        print("Création des graphiques de corrélation avec régression...")
        plot_correlation_with_regression(df)

        print("Création des visualisations des distributions...")
        plot_distributions(df)

        print("Sauvegarde des résultats...")
        save_results(desc_stats, correlations)

        print("\nAnalyse terminée. Consultez les résultats dans les dossiers outputs.")

    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()