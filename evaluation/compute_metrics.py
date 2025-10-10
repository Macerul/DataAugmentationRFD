import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances

# ===========================
# UTILS: caricamento dati
# ===========================

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
# METRICHE CHE RICHIEDONO STANDARDIZZAZIONE
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
# METRICHE SU DATI GREZZI (NON STANDARDIZZATI)
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


# ===========================
# FUNZIONE PRINCIPALE
# ===========================

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


# ===========================
# ESEMPIO DI UTILIZZO
# ===========================

if __name__ == "__main__":
    real_csv = "./Test/LiverCirrhosis_onevsrest_0.csv"
    synthetic_csv = "./Test/LiverCirrhosis_onevsrest_0.csv"
    class_column = "class"

    metrics_results = evaluate_all_metrics(real_csv, synthetic_csv, class_column)

    print("\n=== RISULTATI DELLE METRICHE ===")
    for k, v in metrics_results.items():
        print(f"{k:25s}: {v:.4f}" if isinstance(v, (int, float, np.floating)) else f"{k}: {v}")
