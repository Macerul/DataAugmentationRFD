import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from scipy.stats import entropy
import numpy as np


# UTILS: data loader
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


# Standardization needed
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

#  Helper per calcolo bin dinamico
def auto_n_bins(x):
    """
    Calcola automaticamente il numero di bin con la regola di Freedman–Diaconis.
    """
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return 10  # fallback per feature costante o quasi
    h = 2 * iqr / (len(x) ** (1 / 3))
    if h == 0:
        return 10
    n_bins = int(np.ceil((x.max() - x.min()) / h))
    # Limita il numero di bin per evitare casi estremi
    return max(5, min(n_bins, 200))

def auto_n_bins_1d(x, min_bins=5, max_bins=200):
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return 10
    h = 2 * iqr / (len(x) ** (1/3))
    if h == 0:
        return 10
    n_bins = int(np.ceil((x.max() - x.min()) / h)) if (x.max() - x.min()) > 0 else 10
    return max(min_bins, min(n_bins, max_bins))

# -------------------------
# helper: Freedman–Diaconis per 2D (calcola bins separati per ciascuna dimensione)
# ritorna ([xedges], [yedges])
# -------------------------
def auto_2d_bin_edges(x, y, min_bins=5, max_bins=200):
    nx = auto_n_bins_1d(x, min_bins=min_bins, max_bins=max_bins)
    ny = auto_n_bins_1d(y, min_bins=min_bins, max_bins=max_bins)
    xedges = np.linspace(np.min(x), np.max(x), nx + 1) if np.max(x) > np.min(x) else np.array([np.min(x), np.max(x)+1e-6])
    yedges = np.linspace(np.min(y), np.max(y), ny + 1) if np.max(y) > np.min(y) else np.array([np.min(y), np.max(y)+1e-6])
    return xedges, yedges

#  KL Divergence
def kl_divergence_metric(X_real, X_syn, y_real, y_syn):
    """
    Calcola la Kullback–Leibler Divergence media e per-feature
    usando la regola di Freedman–Diaconis per scegliere il numero di bin.
    """
    kl_results = {}
    for i in range(X_real.shape[1]):
        n_bins = auto_n_bins(X_real[:, i])

        # Costruisci istogrammi normalizzati
        p_hist, bin_edges = np.histogram(X_real[:, i], bins=n_bins, density=True)
        q_hist, _ = np.histogram(X_syn[:, i], bins=bin_edges, density=True)

        # Evita divisioni per zero
        p_hist = np.clip(p_hist, 1e-10, None)
        q_hist = np.clip(q_hist, 1e-10, None)

        kl = entropy(p_hist, q_hist)  # D_KL(P||Q)
        kl_results[f"KL_feature_{i}"] = kl

    kl_results["KL_mean"] = np.mean(list(kl_results.values()))
    return kl_results


#  Jensen–Shannon Divergence
def js_divergence_metric(X_real, X_syn, y_real, y_syn):
    """
    Calcola la Jensen–Shannon Divergence media e per-feature
    usando la regola di Freedman–Diaconis per i bin.
    """
    js_results = {}
    for i in range(X_real.shape[1]):
        n_bins = auto_n_bins(X_real[:, i])

        p_hist, bin_edges = np.histogram(X_real[:, i], bins=n_bins, density=True)
        q_hist, _ = np.histogram(X_syn[:, i], bins=bin_edges, density=True)

        p_hist = np.clip(p_hist, 1e-10, None)
        q_hist = np.clip(q_hist, 1e-10, None)

        m = 0.5 * (p_hist + q_hist)
        js = 0.5 * (entropy(p_hist, m) + entropy(q_hist, m))
        js_results[f"JS_feature_{i}"] = js

    js_results["JS_mean"] = np.mean(list(js_results.values()))
    return js_results

# Q-function:
def q_function_multi_attributes_similarity(X_real, X_syn, use_freedman=True, n_bins_fixed=20, max_pairs=1000, return_per_pair=False):
    """
    Calcola la Q-function (similarità multi-attributes) tra X_real e X_syn.

    Parametri:
        X_real, X_syn : array-like, shape (n_samples, n_features)
        use_freedman : bool, se True usa Freedman–Diaconis per bins 2D, altrimenti usa n_bins_fixed per dimensione
        n_bins_fixed : int, numero di bin se use_freedman=False
        max_pairs : int, numero massimo di coppie (i,j) da valutare; se il numero totale di coppie supera questo valore,
                    le coppie vengono campionate casualmente per ridurre il carico computazionale
        return_per_pair : bool, se True ritorna anche la lista di JS per coppia

    Restituisce:
        dict con chiavi:
            "Q_multi_attribute_similarity" : float (1 - JS_mean)
            "JS_multi_attribute_mean" : float (media delle JS fra coppie)
            "n_pairs_evaluated" : int
            (opzionale) "JS_per_pair" : list di float
    """
    Xr = np.asarray(X_real)
    Xs = np.asarray(X_syn)
    n_features = Xr.shape[1]
    pairs = [(i, j) for i in range(n_features) for j in range(i+1, n_features)]
    total_pairs = len(pairs)

    if total_pairs == 0:
        # meno di 2 features: non applicabile alle coppie
        return {"Q_multi_attribute_similarity": np.nan, "JS_multi_attribute_mean": np.nan, "n_pairs_evaluated": 0}

    # eventualmente campiona le coppie per limiti computazionali
    if total_pairs > max_pairs:
        rng = np.random.default_rng(42)
        pairs = rng.choice(pairs, size=max_pairs, replace=False).tolist()

    js_vals = []
    js_per_pair = []

    for (i, j) in pairs:
        xr = Xr[:, i]
        yr = Xr[:, j]
        xs = Xs[:, i]
        ys = Xs[:, j]

        # calcola bins 2D
        if use_freedman:
            xedges, yedges = auto_2d_bin_edges(xr, yr)
        else:
            xedges = np.linspace(np.min(xr), np.max(xr), n_bins_fixed + 1) if np.max(xr) > np.min(xr) else np.array([np.min(xr), np.max(xr)+1e-6])
            yedges = np.linspace(np.min(yr), np.max(yr), n_bins_fixed + 1) if np.max(yr) > np.min(yr) else np.array([np.min(yr), np.max(yr)+1e-6])

        # istogrammi 2D densità
        p_hist, _, _ = np.histogram2d(xr, yr, bins=[xedges, yedges], density=True)
        q_hist, _, _ = np.histogram2d(xs, ys, bins=[xedges, yedges], density=True)

        p = p_hist.flatten()
        q = q_hist.flatten()

        # evita zeri
        p = np.clip(p, 1e-12, None)
        q = np.clip(q, 1e-12, None)

        m = 0.5 * (p + q)
        js = 0.5 * (entropy(p, m) + entropy(q, m))
        js_vals.append(js)
        if return_per_pair:
            js_per_pair.append(((i, j), js))

    js_mean = float(np.mean(js_vals))
    q_similarity = 1.0 - js_mean

    out = {
        "Q_multi_attribute_similarity": q_similarity,
        "JS_multi_attribute_mean": js_mean,
        "n_pairs_evaluated": len(js_vals)
    }
    if return_per_pair:
        out["JS_per_pair"] = js_per_pair
    return out

# Divergenza sulla sola colonna classe (y_real, y_syn) - JS per la distribuzione delle etichette
def js_class_distribution(y_real, y_syn):
    y_real = np.asarray(y_real)
    y_syn = np.asarray(y_syn)
    # support dinamico (se label non sono 0/1)
    labels = np.union1d(np.unique(y_real), np.unique(y_syn))
    p = np.array([np.sum(y_real == lab) for lab in labels], dtype=float)
    q = np.array([np.sum(y_syn == lab) for lab in labels], dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    m = 0.5 * (p + q)
    js = 0.5 * (entropy(p, m) + entropy(q, m))
    return {"JS_class_distribution": float(js)}

# no standardization
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

    # metriche distribuzionali
    results.update(kl_divergence_metric(X_real, X_syn, y_real, y_syn))
    results.update(js_divergence_metric(X_real, X_syn, y_real, y_syn))

    # Q-function (multi-attribute)
    results.update(q_function_multi_attributes_similarity(X_real, X_syn, use_freedman=True, max_pairs=1000))

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


methodologies = ["casTGAN","SMOTE","SMOTECDNN","SYRFD_thr2","SYRFD_thr4","SYRFD_thr8","llama","deepseek"]

for method in methodologies:
    for ds in datasets:
        real_csv = f"C:/Users\gianp\Desktop\Codes\github\DataAugmentationRFD/imbalanced_datasets/{ds}.csv"
        if method not in ["casTGAN","SMOTE","SMOTECDNN","llama","deepseek"]:
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

'''
'''
# Salva in CSV
output_csv = f"risultati_metriche_wide_final.csv"
df_risultati.to_csv(output_csv, index=False, sep=";")
print(df_risultati.head())
'''
'''

metriche=["Silhouette","Davies-Bouldin","Compactness_class_1",
          "KL_mean","JS_mean","Q_multi_attribute_similarity"]

df_test_formatted = df_risultati[["dataset", "metodo"] + metriche].copy()

for m in metriche:
    df_test_formatted[m] = df_test_formatted[m].round(3).astype(str) + " ± 0.000"

# Salva il CSV pronto per SYRFD
df_test_formatted.to_csv("metriche_per_dataset_final.csv", index=False, sep=";")




metriche=["Silhouette","Davies-Bouldin","Compactness_class_1",
          "KL_mean","JS_mean","Q_multi_attribute_similarity"]
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
summary_final.to_csv("statistiche_metriche_per_metodo_mean_std_final.csv", sep=";")
