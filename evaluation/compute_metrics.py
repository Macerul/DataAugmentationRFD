"""
================================================================================
 VALUTAZIONE METRICHE - DATI SINTETICI vs REALI
================================================================================

 LEGENDA COMMENTI NEL CODICE
 ─────────────────────────────────────────────────────────────────────────────
 [ORIG]       Metrica presente nella versione originale del codice
 [V2]         Metrica aggiunta nella seconda versione (sessione precedente)
 [NEW-FID]    Nuova metrica di FIDELITY aggiunta in questa sessione
 [NEW-DIV]    Nuova metrica di DIVERSITY aggiunta in questa sessione
 ─────────────────────────────────────────────────────────────────────────────

 GUIDA INTERPRETAZIONE SCORE
 ─────────────────────────────────────────────────────────────────────────────
 FIDELITY (quanto i sintetici assomigliano ai reali)
 ┌─────────────────────────────────┬──────────────┬──────────┬───────────────────────────────────────┐
 │ Metrica                         │ Range        │ Ottimo   │ Note                                  │
 ├─────────────────────────────────┼──────────────┼──────────┼───────────────────────────────────────┤
 │ KL_mean             [ORIG]      │ [0, +∞)      │ → 0      │ 0 = dist. identiche; asimmetrica      │
 │ JS_mean             [ORIG]      │ [0, 0.693]   │ → 0      │ Simmetrica; 0.693 = max divergenza    │
 │ Q_multi_attr_sim    [ORIG]      │ [0, 1]       │ → 1      │ 1 - JS su coppie di feature           │
 │ KS_mean             [V2]        │ [0, 1]       │ → 0      │ 0 = distribuzioni identiche           │
 │ Chi2_mean           [V2]        │ [0, +∞)      │ → 0      │ Normalizzato per n; 0 = identiche     │
 │ WD_mean             [V2]        │ [0, +∞)      │ → 0      │ Earth Mover; scala dipende da dati    │
 │ pMSE                [V2]        │ [0, 0.25]    │ → 0      │ 0 = indistinguibili; 0.25 = max sep.  │
 │ MAD_mean            [NEW-FID]   │ [0, +∞)      │ → 0      │ |mu_real - mu_syn| standardizzata     │
 │ MMD                 [NEW-FID]   │ [0, +∞)      │ → 0      │ 0 = dist. identiche nello sp. kernel  │
 │ PCA_WD_mean         [NEW-FID]   │ [0, +∞)      │ → 0      │ WD sulle componenti principali        │
 │ Moment_mean_diff    [NEW-FID]   │ [0, +∞)      │ → 0      │ Diff. media assoluta del 1° momento   │
 │ Moment_var_diff     [NEW-FID]   │ [0, +∞)      │ → 0      │ Diff. media assoluta del 2° momento   │
 │ Moment_skew_diff    [NEW-FID]   │ [0, +∞)      │ → 0      │ Diff. media assoluta del 3° momento   │
 │ Moment_kurt_diff    [NEW-FID]   │ [0, +∞)      │ → 0      │ Diff. media assoluta del 4° momento   │
 ├─────────────────────────────────┼──────────────┼──────────┼───────────────────────────────────────┤
 │ DIVERSITY (quanto i sintetici coprono lo spazio dei reali)                                         │
 ├─────────────────────────────────┼──────────────┼──────────┼───────────────────────────────────────┤
 │ Silhouette          [ORIG]      │ [-1, 1]      │ → -1 *   │ *Basso = real/syn mescolati (buono)   │
 │ Davies-Bouldin      [ORIG]      │ [0, +∞)      │ ALTO **  │ **vedi nota §1 in basso               │
 │ Compactness_cls_k   [ORIG]      │ [0, +∞)      │ ≈ real   │ Confrontare con standalone X_real     │
 │ Precision           [V2]        │ [0, 1]       │ → 1      │ Fidelity lato sintetico               │
 │ Recall              [V2]        │ [0, 1]       │ → 1      │ Diversity: copertura dati reali       │
 │ Density             [V2]        │ [0, +∞)      │ ≈ 1      │ >1 = concentrati; <1 = troppo sparsi  │
 │ Coverage            [V2]        │ [0, 1]       │ → 1      │ Fraz. sfere reali coperte             │
 │ mDCR                [ORIG/V2]   │ [0, +∞)      │ ≈ 1      │ <1 troppo vicini; >1 troppo lontani   │
 │ FE_ratio_mean       [NEW-DIV]   │ [0, +∞)      │ ≈ 1      │ H(syn)/H(real); <1=collapse; >1=smoot │
 │ Coverage_div        [NEW-DIV]   │ [0, 1]       │ → 1      │ Fraz. regioni reali coperte dai sin.  │
 └─────────────────────────────────┴──────────────┴──────────┴───────────────────────────────────────┘

 §1 Davies-Bouldin (real vs syn):
    Calcolato su real+syn con label real=0, syn=1. Valore BASSO = i due gruppi
    sono separati (i sintetici sono distinguibili dai reali → scarsa fidelity).
    Valore ALTO = i due gruppi si sovrappongono (buona fidelity). Interpretazione
    opposta rispetto all'uso classico di clustering.
================================================================================
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import entropy, ks_2samp, wasserstein_distance, skew, kurtosis
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from sklearn.linear_model import LogisticRegression


# ==============================================================================
# [ORIG] UTILS: data loader
# ==============================================================================

def leggi_nomi_csv(cartella):
    """Legge tutti i file .csv in una cartella e restituisce i nomi senza estensione."""
    nomi = []
    for file in os.listdir(cartella):
        if file.lower().endswith('.csv'):
            print(file)
            nomi.append(os.path.splitext(file)[0])
    return nomi


def load_data(real_csv, synthetic_csv, class_column):
    df_real = pd.read_csv(real_csv)
    df_syn  = pd.read_csv(synthetic_csv)

    feature_columns = [c for c in df_real.columns if c != class_column]

    missing_cols = set(feature_columns) - set(df_syn.columns)
    if missing_cols:
        raise ValueError(f"Mancano le colonne nel sintetico: {missing_cols}")

    extra_cols = set(df_syn.columns) - set(feature_columns) - {class_column}
    if extra_cols:
        print(f"Attenzione: colonne extra nel sintetico (ignorate): {extra_cols}")

    X_real = df_real[feature_columns].values
    y_real = df_real[class_column].values
    X_syn  = df_syn[feature_columns].values
    y_syn  = df_syn[class_column].values

    return X_real, X_syn, y_real, y_syn, feature_columns


# ==============================================================================
# [ORIG] HELPER: calcolo bin dinamico (Freedman-Diaconis)
# ==============================================================================

def auto_n_bins(x):
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return 10
    h = 2 * iqr / (len(x) ** (1 / 3))
    if h == 0:
        return 10
    n_bins = int(np.ceil((x.max() - x.min()) / h))
    return max(5, min(n_bins, 200))


def auto_n_bins_1d(x, min_bins=5, max_bins=200):
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return 10
    h = 2 * iqr / (len(x) ** (1 / 3))
    if h == 0:
        return 10
    n_bins = int(np.ceil((x.max() - x.min()) / h)) if (x.max() - x.min()) > 0 else 10
    return max(min_bins, min(n_bins, max_bins))


def auto_2d_bin_edges(x, y, min_bins=5, max_bins=200):
    nx = auto_n_bins_1d(x, min_bins=min_bins, max_bins=max_bins)
    ny = auto_n_bins_1d(y, min_bins=min_bins, max_bins=max_bins)
    xedges = (np.linspace(np.min(x), np.max(x), nx + 1)
              if np.max(x) > np.min(x) else np.array([np.min(x), np.max(x) + 1e-6]))
    yedges = (np.linspace(np.min(y), np.max(y), ny + 1)
              if np.max(y) > np.min(y) else np.array([np.min(y), np.max(y) + 1e-6]))
    return xedges, yedges


# ==============================================================================
# ==============================================================================
#  BLOCCO 1 — METRICHE DI FIDELITY
# ==============================================================================
# ==============================================================================

# ------------------------------------------------------------------------------
# [ORIG] Silhouette (real vs syn come due label)
#   Range: [-1, 1]  |  Ottimo → -1  (le nuvole si sovrappongono = alta fidelity)
# ------------------------------------------------------------------------------
def silhouette_metric(X_real, X_syn, y_real, y_syn):
    X_all = np.vstack([X_real, X_syn])
    y_all = np.hstack([y_real, y_syn])
    X_all_std = StandardScaler().fit_transform(X_all)
    score = silhouette_score(X_all_std, y_all)
    return {"Silhouette": score}


# ------------------------------------------------------------------------------
# [ORIG] Davies-Bouldin Index (real vs syn come due "cluster")
#   Range: [0, +inf)  |  Ottimo → ALTO (vedi nota §1)
# ------------------------------------------------------------------------------
def davies_bouldin_metric(X_real, X_syn, y_real, y_syn):
    X_all = np.vstack([X_real, X_syn])
    y_all = np.hstack([y_real, y_syn])
    X_all_std = StandardScaler().fit_transform(X_all)
    dbi = davies_bouldin_score(X_all_std, y_all)
    return {"Davies-Bouldin": dbi}


# ------------------------------------------------------------------------------
# [ORIG] Intra-class compactness (sui label originali real+syn)
#   Range: [0, +inf)  |  Confrontare con compactness calcolata solo su X_real
# ------------------------------------------------------------------------------
def intra_class_compactness(X_real, X_syn, y_real, y_syn):
    X_all = np.vstack([X_real, X_syn])
    y_all = np.hstack([y_real, y_syn])
    X_all_std = StandardScaler().fit_transform(X_all)
    out = {}
    for c in np.unique(y_all):
        Xc  = X_all_std[y_all == c]
        mu  = Xc.mean(axis=0)
        out[f"Compactness_class_{c}"] = float(np.mean(np.linalg.norm(Xc - mu, axis=1) ** 2))
    return out


# ------------------------------------------------------------------------------
# [ORIG] KL Divergence (feature-wise, Freedman-Diaconis)
#   Range: [0, +inf)  |  Ottimo → 0
# ------------------------------------------------------------------------------
def kl_divergence_metric(X_real, X_syn, y_real=None, y_syn=None):
    out = {}
    for i in range(X_real.shape[1]):
        n_bins = auto_n_bins(X_real[:, i])
        p, edges = np.histogram(X_real[:, i], bins=n_bins, density=True)
        q, _     = np.histogram(X_syn[:, i],  bins=edges,  density=True)
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        out[f"KL_feature_{i}"] = float(entropy(p, q))
    out["KL_mean"] = float(np.mean(list(out.values())))
    return out


# ------------------------------------------------------------------------------
# [ORIG] Jensen-Shannon Divergence (feature-wise, Freedman-Diaconis)
#   Range: [0, 0.693]  |  Ottimo → 0
# ------------------------------------------------------------------------------
def js_divergence_metric(X_real, X_syn, y_real=None, y_syn=None):
    out = {}
    for i in range(X_real.shape[1]):
        n_bins = auto_n_bins(X_real[:, i])
        p, edges = np.histogram(X_real[:, i], bins=n_bins, density=True)
        q, _     = np.histogram(X_syn[:, i],  bins=edges,  density=True)
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        m  = 0.5 * (p + q)
        js = 0.5 * (entropy(p, m) + entropy(q, m))
        out[f"JS_feature_{i}"] = float(js)
    out["JS_mean"] = float(np.mean(list(out.values())))
    return out


# ------------------------------------------------------------------------------
# [ORIG] Q-function (multi-attribute similarity via 2D JS su coppie di feature)
#   Range: [0, 1]  |  Ottimo → 1
# ------------------------------------------------------------------------------
def q_function_multi_attributes_similarity(
    X_real, X_syn,
    use_freedman=True, n_bins_fixed=20, max_pairs=1000, return_per_pair=False
):
    Xr = np.asarray(X_real)
    Xs = np.asarray(X_syn)
    n_features = Xr.shape[1]
    pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]

    if len(pairs) == 0:
        return {"Q_multi_attribute_similarity": np.nan,
                "JS_multi_attribute_mean": np.nan, "n_pairs_evaluated": 0}

    if len(pairs) > max_pairs:
        rng   = np.random.default_rng(42)
        pairs = rng.choice(pairs, size=max_pairs, replace=False).tolist()

    js_vals = []
    js_per_pair = []
    for (i, j) in pairs:
        xr, yr = Xr[:, i], Xr[:, j]
        xs, ys = Xs[:, i], Xs[:, j]
        if use_freedman:
            xedges, yedges = auto_2d_bin_edges(xr, yr)
        else:
            xedges = (np.linspace(np.min(xr), np.max(xr), n_bins_fixed + 1)
                      if np.max(xr) > np.min(xr) else np.array([np.min(xr), np.max(xr) + 1e-6]))
            yedges = (np.linspace(np.min(yr), np.max(yr), n_bins_fixed + 1)
                      if np.max(yr) > np.min(yr) else np.array([np.min(yr), np.max(yr) + 1e-6]))

        p_hist, _, _ = np.histogram2d(xr, yr, bins=[xedges, yedges], density=True)
        q_hist, _, _ = np.histogram2d(xs, ys, bins=[xedges, yedges], density=True)
        p = np.clip(p_hist.flatten(), 1e-12, None)
        q = np.clip(q_hist.flatten(), 1e-12, None)
        m  = 0.5 * (p + q)
        js = 0.5 * (entropy(p, m) + entropy(q, m))
        js_vals.append(js)
        if return_per_pair:
            js_per_pair.append(((i, j), js))

    js_mean = float(np.mean(js_vals))
    out = {
        "Q_multi_attribute_similarity": 1.0 - js_mean,
        "JS_multi_attribute_mean":       js_mean,
        "n_pairs_evaluated":             len(js_vals),
    }
    if return_per_pair:
        out["JS_per_pair"] = js_per_pair
    return out


# ------------------------------------------------------------------------------
# [ORIG] JS sulla distribuzione delle classi
# ------------------------------------------------------------------------------
def js_class_distribution(y_real, y_syn):
    y_real = np.asarray(y_real)
    y_syn  = np.asarray(y_syn)
    labels = np.union1d(np.unique(y_real), np.unique(y_syn))
    p = np.array([np.sum(y_real == lab) for lab in labels], dtype=float)
    q = np.array([np.sum(y_syn  == lab) for lab in labels], dtype=float)
    p = np.clip(p / p.sum(), 1e-12, None)
    q = np.clip(q / q.sum(), 1e-12, None)
    m  = 0.5 * (p + q)
    js = 0.5 * (entropy(p, m) + entropy(q, m))
    return {"JS_class_distribution": float(js)}


# ------------------------------------------------------------------------------
# [ORIG/V2] median Distance to Closest Record (mDCR)
#   Range: [0, +inf)  |  Ottimo ≈ 1  (<1=troppo vicini; >1=troppo lontani)
# ------------------------------------------------------------------------------
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
    elif int_nn == 0:
        dcr = 0.0
    else:
        dcr = mut_nn / int_nn
    return {"mDCR": dcr}


# ------------------------------------------------------------------------------
# [V2] Kolmogorov-Smirnov test (feature-wise)
#   Range: [0, 1]  |  Ottimo → 0
# ------------------------------------------------------------------------------
def ks_test_metric(X_real, X_syn, y_real=None, y_syn=None):
    out = {}
    for i in range(X_real.shape[1]):
        stat, _ = ks_2samp(X_real[:, i], X_syn[:, i])
        out[f"KS_feature_{i}"] = float(stat)
    out["KS_mean"] = float(np.mean(list(out.values())))
    return out


# ------------------------------------------------------------------------------
# [V2] Chi-Square normalizzato (feature-wise)
#   Range: [0, +inf)  |  Ottimo → 0
# ------------------------------------------------------------------------------
def chi_square_metric(X_real, X_syn, y_real=None, y_syn=None):
    from scipy.stats import chi2_contingency
    out = {}
    for i in range(X_real.shape[1]):
        n_bins    = auto_n_bins(X_real[:, i])
        combined  = np.concatenate([X_real[:, i], X_syn[:, i]])
        bin_edges = np.linspace(combined.min(), combined.max(), n_bins + 1)
        p_cnt, _  = np.histogram(X_real[:, i], bins=bin_edges)
        q_cnt, _  = np.histogram(X_syn[:, i],  bins=bin_edges)
        p_cnt     = np.clip(p_cnt, 1, None)
        q_cnt     = np.clip(q_cnt, 1, None)
        chi2, _, _, _ = chi2_contingency(np.vstack([p_cnt, q_cnt]))
        n_total   = X_real.shape[0] + X_syn.shape[0]
        out[f"Chi2_feature_{i}"] = float(chi2 / n_total)
    out["Chi2_mean"] = float(np.mean(list(out.values())))
    return out


# ------------------------------------------------------------------------------
# [V2] Wasserstein Distance standardizzata (feature-wise)
#   Range: [0, +inf)  |  Ottimo → 0
# ------------------------------------------------------------------------------
def wasserstein_metric(X_real, X_syn, y_real=None, y_syn=None):
    scaler    = StandardScaler().fit(X_real)
    Xr_std    = scaler.transform(X_real)
    Xs_std    = scaler.transform(X_syn)
    out = {}
    for i in range(Xr_std.shape[1]):
        out[f"WD_feature_{i}"] = float(wasserstein_distance(Xr_std[:, i], Xs_std[:, i]))
    out["WD_mean"] = float(np.mean(list(out.values())))
    return out


# ------------------------------------------------------------------------------
# [V2] Propensity MSE
#   Range: [0, 0.25]  |  Ottimo → 0  (0 = indistinguibili)
# ------------------------------------------------------------------------------
def pmse_metric(X_real, X_syn, y_real=None, y_syn=None):
    n_real  = X_real.shape[0]
    n_syn   = X_syn.shape[0]
    X_all   = np.vstack([X_real, X_syn])
    y_prop  = np.hstack([np.zeros(n_real), np.ones(n_syn)])
    X_std   = StandardScaler().fit_transform(X_all)
    clf     = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_std, y_prop)
    proba   = clf.predict_proba(X_std)[:, 1]
    c       = n_syn / (n_real + n_syn)
    return {"pMSE": float(np.mean((proba - c) ** 2))}


# ------------------------------------------------------------------------------
# [V2] Precision / Recall / Density / Coverage (k-NN spheres)
#   Precision, Recall, Coverage: [0,1]  |  Ottimo → 1
#   Density: [0,+inf)  |  Ottimo ≈ 1
# ------------------------------------------------------------------------------
def _compute_nn_radii(X, k=1):
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=k + 1)
    return dists[:, -1], tree


def precision_recall_density_coverage(X_real, X_syn, y_real=None, y_syn=None, k=5):
    scaler = StandardScaler().fit(X_real)
    Xr     = scaler.transform(X_real)
    Xs     = scaler.transform(X_syn)

    radii_real, _ = _compute_nn_radii(Xr, k=k)
    radii_syn,  _ = _compute_nn_radii(Xs, k=k)
    n_real = Xr.shape[0]
    n_syn  = Xs.shape[0]

    precision_hits = sum(
        1 for i in range(n_syn)
        if np.any(np.linalg.norm(Xr - Xs[i], axis=1) <= radii_real)
    )
    recall_hits = sum(
        1 for j in range(n_real)
        if np.any(np.linalg.norm(Xs - Xr[j], axis=1) <= radii_syn)
    )
    density_sum = sum(
        np.sum(np.linalg.norm(Xr - Xs[i], axis=1) <= radii_real)
        for i in range(n_syn)
    )
    coverage_hits = sum(
        1 for j in range(n_real)
        if np.any(np.linalg.norm(Xs - Xr[j], axis=1) <= radii_real[j])
    )

    return {
        "Precision": float(precision_hits / n_syn),
        "Recall":    float(recall_hits    / n_real),
        "Density":   float(density_sum    / (k * n_syn)),
        "Coverage":  float(coverage_hits  / n_real),
    }


# ==============================================================================
# ==============================================================================
#  BLOCCO 2 — [NEW-FID] NUOVE METRICHE DI FIDELITY
# ==============================================================================
# ==============================================================================

# ------------------------------------------------------------------------------
# [NEW-FID] Mean Absolute Distance (MAD)
#
#   Differenza assoluta tra la media di ogni feature in X_real e X_syn,
#   su dati standardizzati con scaler fittato su X_real.
#
#   Range: [0, +inf)  |  Ottimo → 0
#   Regola pratica su dati standardizzati:
#     < 0.05  ottimo  |  0.05-0.15  accettabile  |  > 0.20  problematico
# ------------------------------------------------------------------------------
def mean_absolute_distance(X_real, X_syn, y_real=None, y_syn=None):
    """
    [NEW-FID] Mean Absolute Distance per feature (standardizzati su X_real).
    MAD_mean in [0, +inf)  |  Ottimo -> 0
    """
    scaler  = StandardScaler().fit(X_real)
    Xr_std  = scaler.transform(X_real)
    Xs_std  = scaler.transform(X_syn)
    out     = {}
    for i in range(Xr_std.shape[1]):
        mad = float(np.abs(np.mean(Xr_std[:, i]) - np.mean(Xs_std[:, i])))
        out[f"MAD_feature_{i}"] = mad
    out["MAD_mean"] = float(np.mean(list(out.values())))
    return out


# ------------------------------------------------------------------------------
# [NEW-FID] Maximum Mean Discrepancy (MMD) con kernel RBF Gaussiano
#
#   Distanza tra le distribuzioni reale e sintetica nello spazio del kernel.
#   Il bandwidth sigma viene stimato con la Median Heuristic.
#   MMD = 0 iff le due distribuzioni sono identiche.
#
#   Range: [0, +inf)  |  Ottimo → 0
#   Regola pratica su dati standardizzati:
#     < 0.01  ottimo  |  0.01-0.05  buono  |  0.05-0.1  accettabile  |  > 0.1  problematico
# ------------------------------------------------------------------------------
def maximum_mean_discrepancy(X_real, X_syn, y_real=None, y_syn=None, n_subsample=500):
    """
    [NEW-FID] Maximum Mean Discrepancy con kernel RBF (Gaussian).
    MMD in [0, +inf)  |  Ottimo -> 0
    """
    scaler = StandardScaler().fit(X_real)
    Xr     = scaler.transform(X_real)
    Xs     = scaler.transform(X_syn)

    rng = np.random.default_rng(42)
    if Xr.shape[0] > n_subsample:
        Xr = Xr[rng.choice(Xr.shape[0], n_subsample, replace=False)]
    if Xs.shape[0] > n_subsample:
        Xs = Xs[rng.choice(Xs.shape[0], n_subsample, replace=False)]

    # Median heuristic per bandwidth
    all_data = np.vstack([Xr, Xs])
    idx      = rng.choice(all_data.shape[0], min(200, all_data.shape[0]), replace=False)
    pw       = pairwise_distances(all_data[idx], metric="euclidean")
    sigma    = float(np.median(pw[pw > 0]))
    if sigma == 0:
        sigma = 1.0

    def rbf(A, B, s):
        diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
        return np.exp(-np.sum(diff ** 2, axis=-1) / (2 * s ** 2))

    K_rr = rbf(Xr, Xr, sigma)
    K_ss = rbf(Xs, Xs, sigma)
    K_rs = rbf(Xr, Xs, sigma)
    n, m = Xr.shape[0], Xs.shape[0]
    np.fill_diagonal(K_rr, 0)
    np.fill_diagonal(K_ss, 0)

    mmd2 = (K_rr.sum() / (n * (n - 1))
            + K_ss.sum() / (m * (m - 1))
            - 2 * K_rs.mean())
    return {"MMD": float(np.sqrt(max(mmd2, 0.0)))}


# ------------------------------------------------------------------------------
# [NEW-FID] PCA Comparison
#
#   Proietta real e syn nelle prime n_components componenti principali
#   (PCA fittata su X_real standardizzato) e calcola la Wasserstein Distance
#   su ciascuna componente. Cattura la fedeltà nella struttura globale di
#   varianza del dataset.
#
#   Range: PCA_WD_mean in [0, +inf)  |  Ottimo → 0
#   Regola pratica:
#     < 0.1   ottimo  |  0.1-0.3  accettabile  |  > 0.5  problematico
#   PCA_explained_var_cum: frazione di varianza spiegata dalle n_components
#     componenti; un valore > 0.80 garantisce che le componenti siano
#     rappresentative del dataset.
# ------------------------------------------------------------------------------
def pca_comparison(X_real, X_syn, y_real=None, y_syn=None, n_components=5):
    """
    [NEW-FID] WD sulle prime n_components componenti principali (PCA su X_real).
    PCA_WD_mean in [0, +inf)  |  Ottimo -> 0
    """
    scaler = StandardScaler().fit(X_real)
    Xr_std = scaler.transform(X_real)
    Xs_std = scaler.transform(X_syn)

    n_comp = min(n_components,
                 X_real.shape[1],
                 X_real.shape[0] - 1,
                 X_syn.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=42)
    pca.fit(Xr_std)

    Xr_pca = pca.transform(Xr_std)
    Xs_pca = pca.transform(Xs_std)

    out = {}
    for i in range(n_comp):
        out[f"PCA_WD_comp_{i}"] = float(wasserstein_distance(Xr_pca[:, i], Xs_pca[:, i]))

    out["PCA_WD_mean"]           = float(np.mean([out[f"PCA_WD_comp_{i}"] for i in range(n_comp)]))
    out["PCA_explained_var_cum"] = float(np.sum(pca.explained_variance_ratio_))
    return out


# ------------------------------------------------------------------------------
# [NEW-FID] Statistical Moments
#
#   Differenza assoluta tra media, varianza, skewness e kurtosi di X_real e
#   X_syn, calcolata per feature su dati standardizzati con scaler di X_real.
#
#   Range: tutti in [0, +inf)  |  Ottimo → 0 per tutte
#   Soglie indicative su dati standardizzati:
#     mean_diff  < 0.05  |  var_diff  < 0.10  |
#     skew_diff  < 0.20  |  kurt_diff < 0.50
# ------------------------------------------------------------------------------
def statistical_moments(X_real, X_syn, y_real=None, y_syn=None):
    """
    [NEW-FID] Differenza assoluta dei momenti statistici (mean, var, skew, kurt).
    Tutti in [0, +inf)  |  Ottimo -> 0
    """
    scaler = StandardScaler().fit(X_real)
    Xr     = scaler.transform(X_real)
    Xs     = scaler.transform(X_syn)

    mean_d, var_d, skew_d, kurt_d = [], [], [], []
    for i in range(Xr.shape[1]):
        mean_d.append(abs(float(np.mean(Xr[:, i]))  - float(np.mean(Xs[:, i]))))
        var_d.append( abs(float(np.var( Xr[:, i]))  - float(np.var( Xs[:, i]))))
        skew_d.append(abs(float(skew(   Xr[:, i]))  - float(skew(   Xs[:, i]))))
        kurt_d.append(abs(float(kurtosis(Xr[:, i])) - float(kurtosis(Xs[:, i]))))

    return {
        "Moment_mean_diff": float(np.mean(mean_d)),
        "Moment_var_diff":  float(np.mean(var_d)),
        "Moment_skew_diff": float(np.mean(skew_d)),
        "Moment_kurt_diff": float(np.mean(kurt_d)),
    }


# ==============================================================================
# ==============================================================================
#  BLOCCO 3 — [NEW-DIV] NUOVE METRICHE DI DIVERSITY
# ==============================================================================
# ==============================================================================

# ------------------------------------------------------------------------------
# [NEW-DIV] Feature Entropy (Shannon)
#
#   Calcola l'entropia di Shannon per ogni feature sia per X_real che per X_syn
#   usando gli stessi bin edges (basati su Freedman-Diaconis su X_real).
#   Il rapporto FE_ratio = H(syn) / H(real) indica:
#     ≈ 1  → ottimo (i sintetici hanno la stessa variabilita dei reali)
#     < 1  → mode collapse (i sintetici sono troppo concentrati)
#     > 1  → over-smoothing (i sintetici sono artificialmente piu uniformi)
#
#   Range: FE_ratio_mean in [0, +inf)  |  Ottimo ≈ 1
# ------------------------------------------------------------------------------
def feature_entropy(X_real, X_syn, y_real=None, y_syn=None):
    """
    [NEW-DIV] Feature Entropy (Shannon) per feature su X_real e X_syn.
    FE_ratio_mean in [0, +inf)  |  Ottimo ~ 1  |  < 1 = collapse  |  > 1 = over-smooth
    """
    out    = {}
    ratios = []

    for i in range(X_real.shape[1]):
        n_bins_r  = auto_n_bins(X_real[:, i])
        n_bins_s  = auto_n_bins(X_syn[:, i])
        combined  = np.concatenate([X_real[:, i], X_syn[:, i]])
        bin_edges = np.linspace(combined.min(), combined.max(),
                                max(n_bins_r, n_bins_s) + 1)

        p_r, _ = np.histogram(X_real[:, i], bins=bin_edges, density=True)
        p_s, _ = np.histogram(X_syn[:, i],  bins=bin_edges, density=True)
        p_r    = np.clip(p_r, 1e-10, None)
        p_s    = np.clip(p_s, 1e-10, None)

        h_real = float(entropy(p_r))
        h_syn  = float(entropy(p_s))

        out[f"FE_real_feature_{i}"]  = h_real
        out[f"FE_syn_feature_{i}"]   = h_syn
        ratio = (h_syn / h_real) if h_real > 0 else 0.0
        out[f"FE_ratio_feature_{i}"] = float(ratio)
        ratios.append(float(ratio))

    n = X_real.shape[1]
    out["FE_real_mean"]  = float(np.mean([out[f"FE_real_feature_{i}"]  for i in range(n)]))
    out["FE_syn_mean"]   = float(np.mean([out[f"FE_syn_feature_{i}"]   for i in range(n)]))
    out["FE_ratio_mean"] = float(np.mean(ratios))
    return out



# ------------------------------------------------------------------------------
# [NEW-DIV] Coverage Diversity (griglia ipercubi)
#
#   Divide lo spazio delle feature in una griglia di ipercubi (n_bins per
#   dimensione, con bordi fissati su X_real standardizzato). Calcola la
#   frazione di celle occupate dai reali che contengono anche almeno un
#   campione sintetico.
#
#   Range: [0, 1]  |  Ottimo → 1
#     > 0.8  eccellente  |  0.5-0.8  accettabile  |  < 0.5  mode collapse
#
#   NOTA: per d > 10 la griglia diventa sparsa (curse of dimensionality);
#   usare n_bins piccolo (default 5) e interpretare con cautela.
# ------------------------------------------------------------------------------
def coverage_diversity(X_real, X_syn, y_real=None, y_syn=None, n_bins=5):
    """
    [NEW-DIV] Coverage Diversity: frazione di celle reali coperte da X_syn.
    Coverage_div in [0, 1]  |  Ottimo -> 1
    """
    scaler = StandardScaler().fit(X_real)
    Xr     = scaler.transform(X_real)
    Xs     = scaler.transform(X_syn)
    n_feat = Xr.shape[1]

    edges = []
    for i in range(n_feat):
        lo = Xr[:, i].min() - 1e-6
        hi = Xr[:, i].max() + 1e-6
        edges.append(np.linspace(lo, hi, n_bins + 1))

    def assign_bins(X):
        cols = []
        for i in range(n_feat):
            idx = np.clip(np.digitize(X[:, i], edges[i]) - 1, 0, n_bins - 1)
            cols.append(idx)
        return set(map(tuple, np.column_stack(cols).tolist()))

    bins_real = assign_bins(Xr)
    bins_syn  = assign_bins(Xs)
    covered   = len(bins_real & bins_syn)
    total     = len(bins_real)
    return {"Coverage_div": float(covered / total) if total > 0 else 0.0}


# ==============================================================================
# ==============================================================================
#  FUNZIONE PRINCIPALE: evaluate_all_metrics
# ==============================================================================
# ==============================================================================

def evaluate_all_metrics(real_csv, synthetic_csv, class_column):
    X_real, X_syn, y_real, y_syn, _ = load_data(real_csv, synthetic_csv, class_column)

    results = {}

    # --------------------------------------------------------------------------
    # FIDELITY
    # --------------------------------------------------------------------------

    # [ORIG] Strutturali (standardizzati)
    results.update(silhouette_metric(X_real, X_syn, y_real, y_syn))
    results.update(davies_bouldin_metric(X_real, X_syn, y_real, y_syn))
    results.update(intra_class_compactness(X_real, X_syn, y_real, y_syn))

    # [ORIG] Divergenze distribuzionali
    results.update(kl_divergence_metric(X_real, X_syn))
    results.update(js_divergence_metric(X_real, X_syn))

    # [ORIG] Q-function multi-attributo
    results.update(q_function_multi_attributes_similarity(
        X_real, X_syn, use_freedman=True, max_pairs=1000))

    # [ORIG/V2] mDCR (richiede path CSV)
    results.update(median_distance_to_closest_record(real_csv, synthetic_csv))

    # [V2] KS, Chi2, Wasserstein, pMSE
    results.update(ks_test_metric(X_real, X_syn))
    results.update(chi_square_metric(X_real, X_syn))
    results.update(wasserstein_metric(X_real, X_syn))
    results.update(pmse_metric(X_real, X_syn))

    # [V2] Precision / Recall / Density / Coverage
    results.update(precision_recall_density_coverage(X_real, X_syn, k=5))

    # [NEW-FID] Mean Absolute Distance
    results.update(mean_absolute_distance(X_real, X_syn))

    # [NEW-FID] Maximum Mean Discrepancy (RBF kernel)
    results.update(maximum_mean_discrepancy(X_real, X_syn))

    # [NEW-FID] PCA Comparison (WD su componenti principali)
    results.update(pca_comparison(X_real, X_syn, n_components=5))

    # [NEW-FID] Statistical Moments (mean, var, skew, kurt)
    results.update(statistical_moments(X_real, X_syn))

    # --------------------------------------------------------------------------
    # DIVERSITY
    # --------------------------------------------------------------------------

    # [NEW-DIV] Feature Entropy (H_real, H_syn, ratio)
    results.update(feature_entropy(X_real, X_syn))



    # [NEW-DIV] Coverage Diversity (griglia ipercubi)
    results.update(coverage_diversity(X_real, X_syn, n_bins=5))

    return results


# ==============================================================================
# LOOP PRINCIPALE
# ==============================================================================

BASE_PATH = "C:/Users/mary_/PycharmProjects/DataAugmentationRFD"

datasets = [
    'abalone9-18', 'cleveland-0_vs_4', 'dermatology-6', 'iris0', 'ecoli-0_vs_1', 'ecoli1',
    'kddcup-guess_passwd_vs_satan', 'Migraine_onevsrest_0', 'Migraine_onevsrest_1',
    'Migraine_onevsrest_2', 'Migraine_onevsrest_3', 'Migraine_onevsrest_4',
    'Migraine_onevsrest_5', 'new-thyroid1', 'newthyroid2',
    'Obesity_onevsrest_0', 'Obesity_onevsrest_1', 'Obesity_onevsrest_2',
    'Obesity_onevsrest_3', 'Obesity_onevsrest_4', 'Obesity_onevsrest_5',
    'Obesity_onevsrest_6', 'page-blocks-1-3_vs_4', 'pima', 'transfusion',
    'vowel0', 'yeast1', 'yeast3',
]

# Metodi non-LLM: percorso singolo per dataset
METHODS_STANDARD = [
    "casTGAN", "ddpm", "SMOTE", "SMOTECDNN",
    "SYRFD_thr2", "SYRFD_thr4", "SYRFD_thr8",
    "GOGGLE", "tabdiff", "tvae",
    "llama", "deepseek", "ctabgan", "ctabganp"
]

# LLM con strategie multiple: ogni strategia ha la propria cartella
LLM_MODELS = [
    "devstral-small-2_24b-cloud",
    "gemma3_12b",
    "gemma4_31b_cloud",
    "gpt-oss_20b-cloud",
]

LLM_STRATEGIES = [
    "confidence",
    "decision_tree",
    "distribution_guidance",
    "ensemble",
    "hierarchical",
    "react",
    "schema_constraints",
    "self_consistency",
]

# ---------------------------------------------------------------------------
# Helper: restituisce il path del CSV sintetico dato metodo e dataset.
# Per i metodi SYRFD estrae la soglia dal nome (es. "SYRFD_thr4" -> thr=4).
# ---------------------------------------------------------------------------
def _get_synthetic_csv(base_path, method, ds, strategy=None):
    """
    Restituisce il percorso del file CSV sintetico.

    Parametri
    ----------
    base_path : str        - radice del progetto
    method    : str        - nome del metodo (es. "SMOTE", "SYRFD_thr2", "gemma3_12b")
    ds        : str        - nome del dataset
    strategy  : str | None - strategia LLM (es. "confidence"); None per metodi standard

    Per gli LLM vengono provati in ordine i seguenti pattern di nome file:
      1. {ds}_cot_{strategy}.csv   -- naming con prefisso _cot_
      2. {ds}_{strategy}.csv       -- naming diretto
    Viene restituito il primo path esistente su disco; se nessuno esiste
    viene restituito il primo candidato (il chiamante ricevera un errore
    leggibile che mostra il path atteso e facilita il debug).
    """
    if method.startswith("SYRFD_thr"):
        # Metodi SYRFD: classification_results_SYRFD_thr{N}/new_tuples/{ds}_new_tuples_{N}.csv
        thr = method.split("_thr")[1]
        return (
            f"{base_path}/classification_results_SYRFD_thr{thr}/"
            f"new_tuples/{ds}_new_tuples_{thr}.csv"
        )

    elif strategy is not None:
        # LLM con strategia: prova entrambe le convenzioni di denominazione
        folder = f"{base_path}/classification_results_{method}/{strategy}/new_tuples"

        candidates = [
            f"{folder}/{ds}_cot_{strategy}.csv",   # convenzione _cot_
            f"{folder}/{ds}_{strategy}.csv",        # convenzione diretta
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path

        # Nessun file trovato: avvisa e restituisce il primo candidato per
        # mostrare il path atteso nell errore a valle
        print(f"  [WARN] Nessun file trovato per {ds} / {method} / {strategy}. "
              f"Candidati provati: {candidates}")
        return candidates[0]

    else:
        # Tutti gli altri metodi standard (percorso piatto)
        return (
            f"{base_path}/classification_results_{method}/"
            f"new_tuples/{ds}_new_tuples_{method}.csv"
        )


df_results = []

# ---------------------------------------------------------------------------
# 1) Metodi standard (un solo percorso per dataset)
# ---------------------------------------------------------------------------
for method in METHODS_STANDARD:
    for ds in datasets:
        real_csv      = f"{BASE_PATH}/imbalanced_datasets/{ds}.csv"
        synthetic_csv = _get_synthetic_csv(BASE_PATH, method, ds)
        class_column  = "class"

        try:
            metrics_results = evaluate_all_metrics(real_csv, synthetic_csv, class_column)
        except Exception as e:
            print(f"[ERRORE] {ds} / {method}: {e}")
            continue

        print(f"\n=== RISULTATI PER {ds} CON {method} ===")
        for k, v in metrics_results.items():
            print(
                f"  {k:40s}: {v:.4f}"
                if isinstance(v, (int, float, np.floating))
                else f"  {k}: {v}"
            )

        # La colonna "metodo" contiene solo il nome del metodo (es. "SMOTE")
        row = {"dataset": ds, "metodo": method, "strategia": "—"}
        row.update(metrics_results)
        df_results.append(row)

# ---------------------------------------------------------------------------
# 2) LLM con strategie multiple (un percorso per combinazione model x strategy)
# ---------------------------------------------------------------------------
for model in LLM_MODELS:
    for strategy in LLM_STRATEGIES:
        for ds in datasets:
            real_csv      = f"{BASE_PATH}/imbalanced_datasets/{ds}.csv"
            synthetic_csv = _get_synthetic_csv(BASE_PATH, model, ds, strategy=strategy)
            class_column  = "class"

            try:
                metrics_results = evaluate_all_metrics(real_csv, synthetic_csv, class_column)
            except Exception as e:
                print(f"[ERRORE] {ds} / {model} / {strategy}: {e}")
                continue

            print(f"\n=== RISULTATI PER {ds} CON {model} [{strategy}] ===")
            for k, v in metrics_results.items():
                print(
                    f"  {k:40s}: {v:.4f}"
                    if isinstance(v, (int, float, np.floating))
                    else f"  {k}: {v}"
                )

            # "metodo" include sia il nome del modello che la strategia per
            # permettere raggruppamenti flessibili nel summary finale
            row = {
                "dataset":   ds,
                "metodo":    f"{model}__{strategy}",  # chiave univoca
                "llm_model": model,
                "strategia": strategy,
            }
            row.update(metrics_results)
            df_results.append(row)


# ==============================================================================
# EXPORT
# ==============================================================================

df_risultati = pd.DataFrame(df_results)

# Colonne identificative sempre in testa; il resto segue l'ordine di inserimento
id_cols   = ["dataset", "metodo", "strategia", "llm_model"]
id_cols   = [c for c in id_cols if c in df_risultati.columns]
other_cols = [c for c in df_risultati.columns if c not in id_cols]
df_risultati = df_risultati[id_cols + other_cols]

# Arrotonda solo le colonne numeriche
numeric_cols = df_risultati.select_dtypes(include="number").columns
df_risultati[numeric_cols] = df_risultati[numeric_cols].round(4)

df_risultati.to_csv("risultati_metriche_wide_final.csv", index=False, sep=";")
print(df_risultati.head())

# Metriche aggregate per summary (esclude colonne per-feature)
metriche_summary = [
    # FIDELITY ─────────────────────────────────────────────────────────────
    "Silhouette",                    # [ORIG]
    "Davies-Bouldin",                # [ORIG]
    "Compactness_class_1",           # [ORIG]
    "KL_mean",                       # [ORIG]
    "JS_mean",                       # [ORIG]
    "Q_multi_attribute_similarity",  # [ORIG]
    "KS_mean",                       # [V2]
    "Chi2_mean",                     # [V2]
    "WD_mean",                       # [V2]
    "pMSE",                          # [V2]
    "Precision",                     # [V2]
    "Recall",                        # [V2]
    "Density",                       # [V2]
    "Coverage",                      # [V2]
    "mDCR",                          # [ORIG/V2]
    "MAD_mean",                      # [NEW-FID]
    "MMD",                           # [NEW-FID]
    "PCA_WD_mean",                   # [NEW-FID]
    "PCA_explained_var_cum",         # [NEW-FID] (diagnostico)
    "Moment_mean_diff",              # [NEW-FID]
    "Moment_var_diff",               # [NEW-FID]
    "Moment_skew_diff",              # [NEW-FID]
    "Moment_kurt_diff",              # [NEW-FID]
    # DIVERSITY ────────────────────────────────────────────────────────────
    "FE_real_mean",                  # [NEW-DIV] (diagnostico)
    "FE_syn_mean",                   # [NEW-DIV] (diagnostico)
    "FE_ratio_mean",                 # [NEW-DIV]
    "Coverage_div",                  # [NEW-DIV]
]

available = [m for m in metriche_summary if m in df_risultati.columns]

# --------------------------------------------------------------------------
# Tabella per dataset (placeholder +/- 0.000 per compatibilita SYRFD)
# --------------------------------------------------------------------------
fmt_id_cols = [c for c in ["dataset", "metodo", "strategia", "llm_model"]
               if c in df_risultati.columns]
df_fmt = df_risultati[fmt_id_cols + available].copy()
for m in available:
    df_fmt[m] = df_fmt[m].round(3).astype(str) + " +/- 0.000"
df_fmt.to_csv("metriche_per_dataset_final.csv", index=False, sep=";")

# --------------------------------------------------------------------------
# Summary 1 — per "metodo" (raggruppa sia metodi standard che LLM+strategia)
# Ogni LLM+strategia appare come riga separata (chiave: "gemma3_12b__confidence" ecc.)
# --------------------------------------------------------------------------
summary_by_method = df_risultati.groupby("metodo")[available].agg(["mean", "std"]).round(4)
for m in available:
    summary_by_method[(m, "mean+/-std")] = (
        summary_by_method[(m, "mean")].astype(str)
        + " +/- "
        + summary_by_method[(m, "std")].astype(str)
    )
summary_by_method = summary_by_method.loc[:, pd.IndexSlice[:, "mean+/-std"]]
summary_by_method.columns = [m for m, _ in summary_by_method.columns]
summary_by_method.to_csv("statistiche_per_metodo_mean_std_final.csv", sep=";")
print("\n--- Summary per metodo ---")
print(summary_by_method)

# --------------------------------------------------------------------------
# Summary 2 — per modello LLM (media su tutte le strategie per ciascun LLM)
# Utile per confrontare i modelli LLM indipendentemente dalla strategia.
# --------------------------------------------------------------------------
df_llm = df_risultati[df_risultati["llm_model"].notna()].copy() \
    if "llm_model" in df_risultati.columns else pd.DataFrame()

if not df_llm.empty:
    summary_by_llm = df_llm.groupby("llm_model")[available].agg(["mean", "std"]).round(4)
    for m in available:
        summary_by_llm[(m, "mean+/-std")] = (
            summary_by_llm[(m, "mean")].astype(str)
            + " +/- "
            + summary_by_llm[(m, "std")].astype(str)
        )
    summary_by_llm = summary_by_llm.loc[:, pd.IndexSlice[:, "mean+/-std"]]
    summary_by_llm.columns = [m for m, _ in summary_by_llm.columns]
    summary_by_llm.to_csv("statistiche_per_llm_model_mean_std_final.csv", sep=";")
    print("\n--- Summary per modello LLM (media su tutte le strategie) ---")
    print(summary_by_llm)

# --------------------------------------------------------------------------
# Summary 3 — per strategia LLM (media su tutti i modelli LLM per strategia)
# Utile per capire quale strategia funziona meglio indipendentemente dal modello.
# --------------------------------------------------------------------------
if not df_llm.empty:
    summary_by_strategy = df_llm.groupby("strategia")[available].agg(["mean", "std"]).round(4)
    for m in available:
        summary_by_strategy[(m, "mean+/-std")] = (
            summary_by_strategy[(m, "mean")].astype(str)
            + " +/- "
            + summary_by_strategy[(m, "std")].astype(str)
        )
    summary_by_strategy = summary_by_strategy.loc[:, pd.IndexSlice[:, "mean+/-std"]]
    summary_by_strategy.columns = [m for m, _ in summary_by_strategy.columns]
    summary_by_strategy.to_csv("statistiche_per_strategia_llm_mean_std_final.csv", sep=";")
    print("\n--- Summary per strategia LLM (media su tutti i modelli) ---")
    print(summary_by_strategy)