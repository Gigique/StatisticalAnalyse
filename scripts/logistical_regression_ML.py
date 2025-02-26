import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt


# Fonction pour charger les données
def load_data(data_path):
    return pd.read_csv(data_path)


# Fonction pour préparer les données
def preprocess_data(df):
    y = df["Taux d’insertion"]
    X = df.drop(columns="Taux d’insertion")
    y_binaire = (y >= 89).astype(int)  # Binarisation du taux d'insertion avec un seuil arbitraire
    return X, y_binaire


# Fonction pour entraîner le modèle de régression logistique
def fit_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# Fonction pour tester les seuils et calculer les métriques
def diff_seuils(probabilites, y_test, seuils_proba):
    accuracies = []
    precisions = []

    for seuil in seuils_proba:
        predictions = (probabilites >= seuil).astype(int)
        accuracies.append(accuracy_score(y_test, predictions))
        precisions.append(precision_score(y_test, predictions))

    return accuracies, precisions


# Fonction pour calculer la courbe ROC et l'AUC
def calculate_roc_curve(y_test, probabilites):
    fpr, tpr, thresholds = roc_curve(y_test, probabilites)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


# Fonction pour déterminer le seuil optimal en maximisant la distance TPR - FPR
def find_optimal_threshold(fpr, tpr, thresholds):
    distance = tpr - fpr
    optimal_idx = np.argmax(distance)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    return optimal_threshold, optimal_tpr, optimal_fpr


# Fonction pour réentraîner le modèle avec le seuil optimal et afficher les résultats
def evaluate_with_optimal_threshold(probabilites, y_test, optimal_threshold):
    # Réévaluer avec le seuil optimal
    predictions_optimal = (probabilites >= optimal_threshold).astype(int)
    precision_optimal = precision_score(y_test, predictions_optimal)
    accuracy_optimal = accuracy_score(y_test, predictions_optimal)

    print(f"\nRésultats avec le seuil optimal de {optimal_threshold:.2f}:")
    print(f"Précision : {precision_optimal:.2f}")
    print(f"Exactitude : {accuracy_optimal:.2f}")

    return precision_optimal, accuracy_optimal


# Fonction pour sauvegarder les résultats dans un fichier texte
def save_ml_results(OUTPUT_RESULTS_PATH, precision, accuracy, seuils_proba, accuracies, precisions, optimal_threshold,
                    optimal_tpr, optimal_fpr, precision_optimal, accuracy_optimal):
    os.makedirs(OUTPUT_RESULTS_PATH, exist_ok=True)  # Créer le dossier si nécessaire

    # Sauvegarde des résultats dans un fichier texte
    results_file = os.path.join(OUTPUT_RESULTS_PATH, "model_ml_results.txt")
    with open(results_file, 'w', encoding='utf-8') as file:
        file.write("-" * 40 + "\n")
        file.write("Résultats du modèle entraîné avec un seuil de probabilité de 0.5\n")
        file.write(f"Précision au seuil de 0.5 : {precision:.2f}\n")
        file.write(f"Exactitude au seuil de 0.5 : {accuracy:.2f}\n")
        file.write("-" * 40 + "\n")
        file.write("Métriques pour différents seuils de probabilité :\n")
        for seuil, acc, prec in zip(seuils_proba, accuracies, precisions):
            file.write(f"Seuil: {seuil:.2f} | Accuracy: {acc:.2f} | Precision: {prec:.2f}\n")
        file.write("-" * 40 + "\n")
        file.write(f"Seuil optimal : {optimal_threshold:.2f}\n")
        file.write(f"TPR optimal : {optimal_tpr:.2f}\n")
        file.write(f"FPR optimal : {optimal_fpr:.2f}\n")
        file.write("-" * 40 + "\n")
        file.write("Résultats du modèle entraîné avec un seuil de probabilité de 0.56\n")
        file.write(f"Précision avec seuil optimal : {precision_optimal:.2f}\n")
        file.write(f"Exactitude avec seuil optimal : {accuracy_optimal:.2f}\n")


# Fonction pour sauvegarder la courbe ROC
def save_roc_curve(fpr, tpr, roc_auc, OUTPUT_RESULTS_PATH):
    os.makedirs(OUTPUT_RESULTS_PATH, exist_ok=True)  # Créer le dossier si nécessaire

    # Sauvegarde de la courbe ROC dans un fichier
    roc_curve_file = os.path.join(OUTPUT_RESULTS_PATH, "roc_curve.png")
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonale aléatoire : AUC de 0.5
    plt.xlabel('Taux de faux positifs (FPR)')
    plt.ylabel('Taux de vrais positifs (TPR)')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.savefig(roc_curve_file)
    plt.close()


# Fonction principale qui orchestre les appels
def main():
    try:
        # Paramètres du chemin de données et des résultats
        DATA_PATH = "../clean_data/df_final.csv"  #
        OUTPUT_RESULTS_PATH = "../outputs/reglog"

        # Charger et préparer les données
        df = load_data(DATA_PATH)
        X, y_binaire = preprocess_data(df)

        # Séparer les données en train et test
        X_train, X_test, y_train, y_test = train_test_split(X, y_binaire, test_size=0.3, random_state=42)

        # Entraîner le modèle de régression logistique
        model = fit_logistic_regression(X_train, y_train)

        # Prédictions sur le test set avec le seuil de probabilité à 0.5
        probabilites = model.predict_proba(X_test)[:, 1]
        predictions = (probabilites >= 0.5).astype(int)

        # Calcul des métriques de précision et exactitude au seuil de 0.5
        precision = precision_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)

        # Tester les différents seuils
        seuils_proba = np.arange(0.5, 1.05, 0.05)
        accuracies, precisions = diff_seuils(probabilites, y_test, seuils_proba)

        # Calcul de la courbe ROC et de l'AUC
        fpr, tpr, thresholds, roc_auc = calculate_roc_curve(y_test, probabilites)

        # Trouver le seuil optimal
        optimal_threshold, optimal_tpr, optimal_fpr = find_optimal_threshold(fpr, tpr, thresholds)

        # Réévaluer avec le seuil optimal et afficher les résultats
        precision_optimal, accuracy_optimal = evaluate_with_optimal_threshold(probabilites, y_test, optimal_threshold)

        # Sauvegarder les résultats dans un fichier texte
        save_ml_results(OUTPUT_RESULTS_PATH, precision, accuracy, seuils_proba, accuracies, precisions,
                        optimal_threshold,
                        optimal_tpr, optimal_fpr, precision_optimal, accuracy_optimal)

        # Sauvegarder la courbe ROC
        save_roc_curve(fpr, tpr, roc_auc, OUTPUT_RESULTS_PATH)

        print("\nLes résultats du modèle de machine learning ont été sauvegardés.")
        print("La courbe ROC a été sauvegardée.")

    except Exception as e:
        print(f"Erreur : {e}")


# Exécution du script
if __name__ == "__main__":
    main()
