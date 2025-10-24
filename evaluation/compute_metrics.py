import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances

# ===========================
# UTILS: data loader
# ===========================
import os

def leggi_nomi_csv(cartella):
    """
    Legge tutti i file .csv in una cartella e restituisce i nomi senza estensione.

    Args:
        cartella (str): percorso della cartella.

    Returns:
        list: lista dei nomi dei file (senza estensione .csv)
    """
    nomi = []
    for file in os.listdir(cartella):
        if file.lower().endswith('.csv'):
            print(file)
            nome_senza_estensione = os.path.splitext(file)[0]
            nomi.append(nome_senza_estensione)
    return nomi

def load_data(real_csv, synthetic_csv, class_column):
    df_real = pd.read_csv(real_csv)
    df_syn = pd.read_csv(synthetic_csv)

    feature_columns = [c for c in df_real.columns if c != class_column]

    missing_cols = set(feature_columns) - set(df_syn.columns)
    if missing_cols:
        raise ValueError(f"Mancano le colonne nel sintetico: {missing_cols}")

    extra_cols = set(df_syn.columns) - set(feature_columns) - {class_column}
    if extra_cols:
        print(f"⚠️ Attenzione: colonne extra nel sintetico (ignorate): {extra_cols}")

    X_real = df_real[feature_columns].values
    y_real = df_real[class_column].values

    X_syn = df_syn[feature_columns].values
    y_syn = df_syn[class_column].values

    return X_real, X_syn, y_real, y_syn, feature_columns

# ===========================
# Standardization needed
# ===========================

def silhouette_metric(X_real, X_syn, y_real, y_syn):
    X_all = np.vstack([X_real, X_syn])
    y_all = np.hstack([y_real, y_syn])
    X_all_std = StandardScaler().fit_transform(X_all)
    score = silhouette_score(X_all_std, y_all)
    return {"Silhouette": score}


def davies_bouldin_metric(X_real, X_syn, y_real, y_syn):
    X_all = np.vstack([X_real, X_syn])
    y_all = np.hstack([y_real, y_syn])
    X_all_std = StandardScaler().fit_transform(X_all)
    dbi = davies_bouldin_score(X_all_std, y_all)
    return {"Davies-Bouldin": dbi}


def intra_class_compactness(X_real, X_syn, y_real, y_syn):
    X_all = np.vstack([X_real, X_syn])
    y_all = np.hstack([y_real, y_syn])
    X_all_std = StandardScaler().fit_transform(X_all)

    compactness_results = {}
    for c in np.unique(y_all):
        Xc = X_all_std[y_all == c]
        mu_c = Xc.mean(axis=0)
        compactness = np.mean(np.linalg.norm(Xc - mu_c, axis=1)**2)
        compactness_results[f"Compactness_class_{c}"] = compactness
    return compactness_results


# ===========================
# no standardization
# ===========================

def median_distance_to_closest_record(real_csv, synthetic_csv, distance_metric="euclidean"):
    real_data = pd.read_csv(real_csv)
    synt_data = pd.read_csv(synthetic_csv)

    if list(real_data.columns) != list(synt_data.columns):
        raise ValueError("Le colonne dei due dataset non coincidono!")

    dist_mutual = pairwise_distances(synt_data, real_data, metric=distance_metric)
    min_dists_synt_to_real = np.min(dist_mutual, axis=1)

    dist_internal = pairwise_distances(real_data, real_data, metric=distance_metric)
    np.fill_diagonal(dist_internal, np.nan)
    min_dists_real_to_real = np.nanmin(dist_internal, axis=1)

    mut_nn = np.median(min_dists_synt_to_real)
    int_nn = np.median(min_dists_real_to_real)

    if int_nn == 0 and mut_nn == 0:
        dcr = 1.0
    elif int_nn == 0 and mut_nn != 0:
        dcr = 0.0
    else:
        dcr = mut_nn / int_nn

    return {"mDCR": dcr}

def evaluate_all_metrics(real_csv, synthetic_csv, class_column):
    X_real, X_syn, y_real, y_syn, _ = load_data(real_csv, synthetic_csv, class_column)

    results = {}
    # metriche standardizzate
    results.update(silhouette_metric(X_real, X_syn, y_real, y_syn))
    results.update(davies_bouldin_metric(X_real, X_syn, y_real, y_syn))
    results.update(intra_class_compactness(X_real, X_syn, y_real, y_syn))

    # metriche su dati grezzi
    results.update(median_distance_to_closest_record(real_csv, synthetic_csv))

    return results



#nomi = leggi_nomi_csv("C:/Users\gianp\Desktop\Codes\github\DataAugmentationRFD/imbalanced_datasets")
#print(nomi)

df_results = []
datasets = ['abalone9-18', 'cleveland-0_vs_4', 'dermatology-6', 'iris0','ecoli-0_vs_1', 'ecoli1',
            'kddcup-guess_passwd_vs_satan', 'Migraine_onevsrest_0', 'Migraine_onevsrest_1', 'Migraine_onevsrest_2',
            'Migraine_onevsrest_3', 'Migraine_onevsrest_4', 'Migraine_onevsrest_5', 'new-thyroid1', 'newthyroid2',
            'Obesity_onevsrest_0','Obesity_onevsrest_1', 'Obesity_onevsrest_2', 'Obesity_onevsrest_3',
            'Obesity_onevsrest_4', 'Obesity_onevsrest_5', 'Obesity_onevsrest_6',
            'page-blocks-1-3_vs_4','pima', 'transfusion','vowel0', 'yeast1', 'yeast3']


methodologies = ["casTGAN","SMOTE","SMOTECDNN","SYRFD_thr2","SYRFD_thr4","SYRFD_thr8"]

for method in methodologies:
    for ds in datasets:
        real_csv = f"C:/Users\gianp\Desktop\Codes\github\DataAugmentationRFD/imbalanced_datasets/{ds}.csv"
        if method not in ["casTGAN","SMOTE","SMOTECDNN"]:
            tmp = method.split("_thr")
            synthetic_csv = f"C:/Users\gianp\Desktop\Codes\github\DataAugmentationRFD/classification_results_{tmp[0]}_thr{tmp[1]}/new_tuples/{ds}_new_tuples_{tmp[1]}.csv"
        else:
            synthetic_csv = f"C:/Users\gianp\Desktop\Codes\github\DataAugmentationRFD/classification_results_{method}/new_tuples/{ds}_new_tuples_{method}.csv"

        class_column = "class"

        metrics_results = evaluate_all_metrics(real_csv, synthetic_csv, class_column)
        print(f"\n=== RISULTATI DELLE METRICHE PER {ds} CON {method} ===")
        for k, v in metrics_results.items():
            print(f"{k:25s}: {v:.4f}" if isinstance(v, (int, float, np.floating)) else f"{k}: {v}")

        row = {"dataset": ds, "metodo": method}
        row.update(metrics_results)
        df_results.append(row)

# Converte in DataFrame "wide"
df_risultati = pd.DataFrame(df_results)

# Ordina le colonne per leggibilità
col_order = ["dataset", "metodo"] + [c for c in df_risultati.columns if c not in ["dataset", "metodo"]]
df_risultati = df_risultati[col_order]

df_risultati = df_risultati.round(3)

# Salva in CSV
output_csv = f"risultati_metriche_wide.csv"
df_risultati.to_csv(output_csv, index=False, sep=";")
print(f"\n✅ File salvato in formato WIDE: {output_csv}")
print(df_risultati.head())

metriche=["Silhouette","Davies-Bouldin","Compactness_class_0","Compactness_class_1"]
# Calcola media e std
summary = df_risultati.groupby("metodo")[metriche].agg(["mean", "std"]).round(3)

# Combina media e std in una sola colonna con il simbolo ±
summary_combined = summary.copy()
for m in metriche:
    summary_combined[(m, "mean±std")] = summary[(m, "mean")].astype(str) + " ± " + summary[(m, "std")].astype(str)

# Tieni solo le colonne combinate
summary_final = summary_combined.loc[:, pd.IndexSlice[:, "mean±std"]]
summary_final.columns = [m for m, _ in summary_final.columns]

# Mostra la tabella finale
print(summary_final)

# Salva in CSV con separatore ;
summary_final.to_csv("statistiche_metriche_per_metodo_mean_std.csv", sep=";")
