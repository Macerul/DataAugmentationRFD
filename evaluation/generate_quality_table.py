# generate_quality_table.py
# --------------------------
# Legge statistiche_per_metodo_mean_std_final.csv e produce quality_table_full.tex
#
# Layout per ICDM (due colonne, una pagina intera):
#   - Tabella TRASPOSTA: colonne = metodi, righe = metriche
#   - Intestazioni di colonna ruotate a 90 gradi con rotatebox
#   - Solo la media (no std) per risparmiare spazio
#   - scalebox aggressivo + tabcolsep minimo
#   - table* per occupare entrambe le colonne
#
# Richiede nel preambolo LaTeX:
#   usepackage{booktabs, xcolor, graphicx, rotating, array, makecell, colortbl}
#   newcommand{system}{SyRFD}

import csv, os, sys
import pandas as pd

INPUT_CSV  = "statistiche_per_metodo_mean_std_final.csv"
OUTPUT_TEX = "quality_table_full.tex"
OVERFLOW   = 1e6

# Radice del progetto — usata solo per il calcolo dei duplicati
BASE_PATH  = r"C:/Users/mary_/PycharmProjects/DataAugmentationRFD"

DATASETS = [
    "abalone9-18", "cleveland-0_vs_4", "dermatology-6", "iris0",
    "ecoli-0_vs_1", "ecoli1", "kddcup-guess_passwd_vs_satan",
    "Migraine_onevsrest_0", "Migraine_onevsrest_1", "Migraine_onevsrest_2",
    "Migraine_onevsrest_3", "Migraine_onevsrest_4", "Migraine_onevsrest_5",
    "new-thyroid1", "newthyroid2",
    "Obesity_onevsrest_0", "Obesity_onevsrest_1", "Obesity_onevsrest_2",
    "Obesity_onevsrest_3", "Obesity_onevsrest_4", "Obesity_onevsrest_5",
    "Obesity_onevsrest_6", "page-blocks-1-3_vs_4", "pima", "transfusion",
    "vowel0", "yeast1", "yeast3",
]

LLM_MODELS_KEYS = [
    "devstral-small-2_24b-cloud",
    "gemma3_12b",
    "gemma4_31b_cloud",
    "gpt-oss_20b-cloud",
]
STRATEGIES_KEYS = [
    "confidence", "decision_tree", "distribution_guidance", "ensemble",
    "hierarchical", "react", "schema_constraints", "self_consistency",
]


# ─────────────────────────────────────────────────────────────────────────────
# Path resolver  (same logic as metrics_evaluation.py)
# ─────────────────────────────────────────────────────────────────────────────

def _get_synthetic_csv(base_path, method, ds, strategy=None):
    if method.startswith("SYRFD_thr"):
        thr = method.split("_thr")[1]
        return (
            f"{base_path}/classification_results_SYRFD_thr{thr}/"
            f"new_tuples/{ds}_new_tuples_{thr}.csv"
        )
    elif strategy is not None:
        folder = f"{base_path}/classification_results_{method}/{strategy}/new_tuples"
        candidates = [
            f"{folder}/{ds}_cot_{strategy}.csv",
            f"{folder}/{ds}_{strategy}.csv",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        print(f"  [WARN] Nessun file trovato per {ds}/{method}/{strategy}. "
              f"Candidati: {candidates}")
        return candidates[0]
    else:
        return (
            f"{base_path}/classification_results_{method}/"
            f"new_tuples/{ds}_new_tuples_{method}.csv"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate counter
# ─────────────────────────────────────────────────────────────────────────────

def count_duplicates_in_csv(path):
    """
    Legge il CSV sintetico e restituisce il numero di righe duplicate
    (n_righe - n_righe_uniche). Ritorna None se il file non esiste.
    """
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        return int(len(df) - len(df.drop_duplicates()))
    except Exception as e:
        print(f"  [ERRORE duplicati] {path}: {e}")
        return None


def compute_duplicates_mean(base_path, method, strategy=None):
    """
    Media del numero di duplicati su tutti i dataset, per un dato metodo
    (e strategia opzionale). Restituisce None se nessun file trovato.
    """
    counts = []
    for ds in DATASETS:
        path = _get_synthetic_csv(base_path, method, ds, strategy=strategy)
        n = count_duplicates_in_csv(path)
        if n is not None:
            counts.append(n)
    if not counts:
        return None
    return sum(counts) / len(counts)

# ─────────────────────────────────────────────────────────────────────────────
# Metriche: (csv_col, short_label, direction, group_label)
# direction: 'low' → bold=min | 'high' → bold=max | 'approx1' → bold=closest to 1
# ─────────────────────────────────────────────────────────────────────────────
METRICS = [
    # Fidelity — distributional
    ("Silhouette",                   r"$\mathcal{S}$",       "low",     "Fid."),
    ("Davies-Bouldin",               r"$DBI$",               "high",    "Fid."),
    ("KL_mean",                      r"$D_{KL}$",            "low",     "Fid."),
    ("JS_mean",                      r"$D_{JS}$",            "low",     "Fid."),
    ("Q_multi_attribute_similarity", r"$\mathcal{Q}$",       "high",    "Fid."),
    ("KS_mean",                      r"$KS$",                "low",     "Fid."),
    ("WD_mean",                      r"$WD$",                "low",     "Fid."),
    ("pMSE",                         r"$pMSE$",              "low",     "Fid."),
    ("MMD",                          r"$MMD$",               "low",     "Fid."),
    ("MAD_mean",                     r"$MAD$",               "low",     "Fid."),
    ("Moment_mean_diff",             r"$\Delta\mu$",         "low",     "Fid."),
    ("Moment_skew_diff",             r"$\Delta sk$",         "low",     "Fid."),
    # Neighbourhood
    ("Precision",                    r"$Prec.$",             "high",    "Neigh."),
    ("Recall",                       r"$Rec.$",              "high",    "Neigh."),
    ("Density",                      r"$Dens.$",             "high",    "Neigh."),
    ("Coverage",                     r"$Cov.$",              "high",    "Neigh."),
    ("mDCR",                         r"$mDCR$",              "approx1", "Neigh."),
    # Diversity
    ("Coverage_div",                 r"$Cov_{div}$",         "high",    "Div."),
]

# ─────────────────────────────────────────────────────────────────────────────
# Method order & display names (columns of the transposed table)
# ─────────────────────────────────────────────────────────────────────────────
STRATEGY_SHORT = {
    "confidence":            "conf",
    "decision_tree":         "dtree",
    "distribution_guidance": "distr",
    "ensemble":              "ens",
    "hierarchical":          "hier",
    "react":                 "react",
    "schema_constraints":    "schema",
    "self_consistency":      "self",
}

LLM_SHORT = {
    "devstral-small-2_24b-cloud": "Dev",
    "gemma3_12b":                 "G3",
    "gemma4_31b_cloud":           "G4",
    "gpt-oss_20b-cloud":          "GPT",
}

def build_method_order(all_csv_keys):
    """Returns list of (csv_key, col_header_tex, group)."""
    order = []

    standard = [
        ("SMOTE",      r"SMOTE",       "base"),
        ("SMOTECDNN",  r"SMOTE-C",     "base"),
        ("casTGAN",    r"casTGAN",     "base"),
        ("ctabgan",    r"CTAB",        "deep"),
        ("ctabganp",   r"CTAB+",       "deep"),
        ("ddpm",       r"DDPM",        "deep"),
        ("tabdiff",    r"TabDiff",     "deep"),
        ("tvae",       r"TVAE",        "deep"),
        ("GOGGLE",     r"GOGGLE",      "deep"),
        ("deepseek",   r"DeepSeek",    "llm0"),
        ("llama",      r"LLaMA",       "llm0"),
    ]
    for key, disp, grp in standard:
        if key in all_csv_keys:
            order.append((key, disp, grp))

    llm_models = [
        "devstral-small-2_24b-cloud",
        "gemma3_12b",
        "gemma4_31b_cloud",
        "gpt-oss_20b-cloud",
    ]
    strategies = [
        "confidence", "decision_tree", "distribution_guidance", "ensemble",
        "hierarchical", "react", "schema_constraints", "self_consistency",
    ]
    for model in llm_models:
        mshort = LLM_SHORT.get(model, model[:3])
        for strat in strategies:
            key = f"{model}__{strat}"
            if key in all_csv_keys:
                sshort = STRATEGY_SHORT.get(strat, strat[:4])
                # Two-line rotated header: model / strategy
                disp = rf"{mshort}/{sshort}"
                order.append((key, disp, f"llm_{model}"))

    for key, disp in [
        ("SYRFD_thr2", r"\system $\phi$2"),
        ("SYRFD_thr4", r"\system $\phi$4"),
        ("SYRFD_thr8", r"\system $\phi$8"),
    ]:
        if key in all_csv_keys:
            order.append((key, disp, "syrfd"))

    return order


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse(val):
    try:
        parts = str(val).split("+/-")
        m, s = float(parts[0].strip()), float(parts[1].strip())
        if abs(m) > OVERFLOW or abs(s) > OVERFLOW:
            return None, None
        return m, s
    except Exception:
        return None, None


def best_value(rows, col, direction, all_keys):
    vals = [parse(rows[k].get(col, ""))[0] for k in all_keys if k in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if direction == "low":    return min(vals)
    if direction == "high":   return max(vals)
    return min(vals, key=lambda x: abs(x - 1))


def fmtmean(x):
    """Compact mean-only formatting."""
    if x is None:
        return "--"
    if abs(x) >= 100: return f"{x:.0f}"
    if abs(x) >= 10:  return f"{x:.1f}"
    if abs(x) >= 1:   return f"{x:.2f}"
    return f"{x:.2f}"


def fmt_cell(m, is_best):
    if m is None:
        return r"\textcolor{gray}{--}"
    val = fmtmean(m)
    if is_best:
        return r"\textbf{" + val + r"}"
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Transposed table builder
# ─────────────────────────────────────────────────────────────────────────────

def make_transposed_table(rows, method_order, metrics_list, dup_means, label, caption):
    """
    Rows = metrics + 1 duplicate row, Columns = methods.
    Column headers rotated 90 degrees.
    Bold = best value per row (per metric, direction-aware).
    dup_means: dict  csv_key -> mean_duplicates (float or None)
    """
    all_keys  = [k for k, _, _ in method_order]
    n_methods = len(all_keys)

    # Pre-compute best per metric (excluding the dup row, lower=better)
    best = {col: best_value(rows, col, direction, all_keys)
            for col, _, direction, _ in metrics_list}

    # Best for duplicates: lower is better
    dup_vals = [dup_means.get(k) for k in all_keys if dup_means.get(k) is not None]
    best_dup = min(dup_vals) if dup_vals else None

    # ── Column spec ──────────────────────────────────────────────────────────
    col_spec = r"l" + r"r" * n_methods

    # ── Group spans for header line 1 ────────────────────────────────────────
    prev_grp  = None
    grp_spans = []
    cur_start = 2
    for idx, (key, disp, grp) in enumerate(method_order):
        col = idx + 2
        if grp != prev_grp:
            if prev_grp is not None:
                grp_spans.append((prev_grp, cur_start, col - 1))
            cur_start = col
            prev_grp  = grp
    if prev_grp is not None:
        grp_spans.append((prev_grp, cur_start, len(method_order) + 1))

    GRP_LABEL = {
        "base":  r"\textit{Classic}",
        "deep":  r"\textit{Deep gen.}",
        "llm0":  r"\textit{LLM}",
        "syrfd": r"\textbf{\system}",
    }
    for model in LLM_MODELS_KEYS:
        short = LLM_SHORT.get(model, model[:3])
        GRP_LABEL[f"llm_{model}"] = rf"\textit{{{short}}}"

    grp_cells  = [r"\multicolumn{1}{l}{}"]
    cmidrules  = []
    for grp, s, e in grp_spans:
        span = e - s + 1
        lbl  = GRP_LABEL.get(grp, grp)
        grp_cells.append(rf"\multicolumn{{{span}}}{{c}}{{{lbl}}}")
        cmidrules.append(rf"\cmidrule(lr){{{s}-{e}}}")
    hdr_grp = " & ".join(grp_cells) + r" \\"

    # ── Header line 2: rotated method names ──────────────────────────────────
    meth_cells = [r"\textbf{Metric}"]
    for key, disp, grp in method_order:
        meth_cells.append(r"\rotatebox{90}{\footnotesize " + disp + r"}")
    hdr_meth = " & ".join(meth_cells) + r" \\"

    # ── Metric-group separators (row index where group changes) ───────────────
    grp_change_before = set()
    prev = None
    for ridx, (col, lbl, direction, grp_lbl) in enumerate(metrics_list):
        if prev is not None and grp_lbl != prev:
            grp_change_before.add(ridx)
        prev = grp_lbl

    # ── Assemble ──────────────────────────────────────────────────────────────
    L = []
    L.append(r"% ================================================================")
    L.append(r"% quality_table_full.tex  — transposed, single-page ICDM format")
    L.append(r"% Preamble: \usepackage{booktabs,graphicx,rotating,array,makecell,colortbl}")
    L.append(r"%           \newcommand{\system}{SyRFD}")
    L.append(r"% Bold = best value per row (direction-aware, mean only).")
    L.append(r"% ================================================================")
    L.append(r"")
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\setlength\tabcolsep{1.8pt}")
    L.append(r"\renewcommand{\arraystretch}{0.82}")
    L.append(r"\scalebox{0.56}{")
    L.append(r"\begin{tabular}{" + col_spec + "}")
    L.append(r"\toprule")
    L.append(hdr_grp)
    L.append(" ".join(cmidrules))
    L.append(hdr_meth)
    L.append(r"\midrule")

    # ── Metric rows ───────────────────────────────────────────────────────────
    for ridx, (col, lbl, direction, grp_lbl) in enumerate(metrics_list):
        if ridx in grp_change_before:
            L.append(r"\midrule")

        arrow = {"low": r"$\downarrow$", "high": r"$\uparrow$",
                 "approx1": r"$\approx\!1$"}[direction]
        row_cells = [lbl + r"\," + arrow]

        for key, disp, grp in method_order:
            m, _ = parse(rows[key].get(col, "")) if key in rows else (None, None)
            is_best = (best[col] is not None and m is not None
                       and abs(m - best[col]) < 1e-9)
            row_cells.append(fmt_cell(m, is_best))

        L.append(" & ".join(row_cells) + r" \\")

    # ── Duplicates row (last, separated by midrule) ───────────────────────────
    L.append(r"\midrule")
    dup_cells = [r"$\#Dup.$\,$\downarrow$"]
    for key, disp, grp in method_order:
        d = dup_means.get(key)
        if d is None:
            dup_cells.append(r"\textcolor{gray}{--}")
        else:
            is_best = (best_dup is not None and abs(d - best_dup) < 1e-9)
            val = f"{d:.1f}" if d < 10 else (f"{d:.0f}" if d < 1000 else f"{d/1000:.1f}k")
            dup_cells.append(r"\textbf{" + val + r"}" if is_best else val)
    L.append(" & ".join(dup_cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"}")
    L.append(r"\caption{" + caption + r"}")
    L.append(r"\label{" + label + r"}")
    L.append(r"\end{table*}")

    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(INPUT_CSV):
        print(f"[ERRORE] File non trovato: {INPUT_CSV}")
        sys.exit(1)

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows   = {r["metodo"]: r for r in reader}

    all_csv_keys = set(rows.keys())
    method_order = build_method_order(all_csv_keys)
    all_keys     = [k for k, _, _ in method_order]

    # ── Exclude metrics with overflow ─────────────────────────────────────────
    bad_cols = set()
    for col, _, _, _ in METRICS:
        for k in all_keys:
            if k not in rows: continue
            raw = rows[k].get(col, "")
            if raw not in ("", "nan", "None"):
                m, _ = parse(raw)
                if m is None:
                    bad_cols.add(col); break

    if bad_cols:
        print(f"[INFO] Metriche escluse per overflow: {bad_cols}")

    clean_metrics = [(col, lbl, d, g) for col, lbl, d, g in METRICS if col not in bad_cols]

    # ── Compute mean duplicates per method ────────────────────────────────────
    # Maps: csv_key -> mean number of duplicates across all datasets
    # For standard methods: strategy=None; for LLM keys ("model__strat"): split.
    print("Calcolo duplicati per ogni metodo...")
    dup_means = {}
    for key, disp, grp in method_order:
        if "__" in key:
            # LLM + strategy: key = "model__strategy"
            model, strategy = key.split("__", 1)
            mean_dup = compute_duplicates_mean(BASE_PATH, model, strategy=strategy)
        else:
            mean_dup = compute_duplicates_mean(BASE_PATH, key, strategy=None)

        dup_means[key] = mean_dup
        status = f"{mean_dup:.1f}" if mean_dup is not None else "N/A (file non trovati)"
        print(f"  {key:<45s}: {status}")

    # ── Build and write table ─────────────────────────────────────────────────
    table_tex = make_transposed_table(
        rows, method_order, clean_metrics, dup_means,
        label="tab:quality_full",
        caption=(
            r"Quality metrics for all methods and LLM strategies "
            r"(mean across datasets). "
            r"\textbf{Bold}: best per metric. "
            r"$\downarrow$ lower is better; $\uparrow$ higher is better; "
            r"$\approx\!1$ closer to 1 is better. "
            r"$\#Dup.$ = mean number of duplicate rows in the synthetic data."
        ),
    )

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(table_tex)

    print(f"\nTabella LaTeX salvata in: {OUTPUT_TEX}")
    print(f"  Metodi  : {len(method_order)}")
    print(f"  Metriche: {len(clean_metrics)} + 1 (duplicati)")




# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP — duplicati per dataset x metodo (tutti i metodi, nessuna media)
# ─────────────────────────────────────────────────────────────────────────────

DATASET_SHORT = {
    "abalone9-18":                  "D1",
    "cleveland-0_vs_4":             "D2",
    "dermatology-6":                "D3",
    "iris0":                        "D4",
    "ecoli-0_vs_1":                 "D5",
    "ecoli1":                       "D6",
    "kddcup-guess_passwd_vs_satan": "D7",
    "Migraine_onevsrest_0":         "D8",
    "Migraine_onevsrest_1":         "D9",
    "Migraine_onevsrest_2":         "D10",
    "Migraine_onevsrest_3":         "D11",
    "Migraine_onevsrest_4":         "D12",
    "Migraine_onevsrest_5":         "D13",
    "new-thyroid1":                 "D14",
    "newthyroid2":                  "D15",
    "Obesity_onevsrest_0":          "D16",
    "Obesity_onevsrest_1":          "D17",
    "Obesity_onevsrest_2":          "D18",
    "Obesity_onevsrest_3":          "D19",
    "Obesity_onevsrest_4":          "D20",
    "Obesity_onevsrest_5":          "D21",
    "Obesity_onevsrest_6":          "D22",
    "page-blocks-1-3_vs_4":        "D23",
    "pima":                         "D24",
    "transfusion":                  "D25",
    "vowel0":                       "D26",
    "yeast1":                       "D27",
    "yeast3":                       "D28",
}


def build_heatmap_columns():
    """
    Returns an ordered list of (csv_key, display_label, group) for the heatmap.
    Columns: Classic baselines, Deep generative, legacy LLM,
             LLM x strategy (4 models x 8 strategies), SYRFD.
    """
    cols = []
    standard = [
        ("SMOTE",     "SMOTE",    "Classic"),
        ("SMOTECDNN", "SMOTE-C",  "Classic"),
        ("casTGAN",   "casTGAN",  "Classic"),
        ("ctabgan",   "CTAB",     "Deep"),
        ("ctabganp",  "CTAB+",    "Deep"),
        ("ddpm",      "DDPM",     "Deep"),
        ("tabdiff",   "TabDiff",  "Deep"),
        ("tvae",      "TVAE",     "Deep"),
        ("GOGGLE",    "GOGGLE",   "Deep"),
        ("deepseek",  "DeepSeek", "LLM"),
        ("llama",     "LLaMA",    "LLM"),
    ]
    for key, lbl, grp in standard:
        cols.append((key, lbl, grp))

    model_short = {
        "devstral-small-2_24b-cloud": "Dev",
        "gemma3_12b":                 "G3",
        "gemma4_31b_cloud":           "G4",
        "gpt-oss_20b-cloud":          "GPT",
    }
    strat_short = {
        "confidence":            "conf",
        "decision_tree":         "dtree",
        "distribution_guidance": "distr",
        "ensemble":              "ens",
        "hierarchical":          "hier",
        "react":                 "react",
        "schema_constraints":    "schema",
        "self_consistency":      "self",
    }
    for model in LLM_MODELS_KEYS:
        ms = model_short.get(model, model[:3])
        for strat in STRATEGIES_KEYS:
            key = f"{model}__{strat}"
            ss  = strat_short.get(strat, strat[:4])
            cols.append((key, f"{ms}\n{ss}", ms))

    cols.append(("SYRFD_thr2", "SyRFD\nphi=2", "SyRFD"))
    cols.append(("SYRFD_thr4", "SyRFD\nphi=4", "SyRFD"))
    cols.append(("SYRFD_thr8", "SyRFD\nphi=8", "SyRFD"))

    return cols


def build_duplicates_matrix(base_path):
    """
    DataFrame (n_datasets x n_methods) with raw duplicate counts.
    NaN where the synthetic file was not found.
    """
    import numpy as np
    heatmap_cols = build_heatmap_columns()
    ds_labels    = [DATASET_SHORT[d] for d in DATASETS]
    col_keys     = [k for k, _, _ in heatmap_cols]

    matrix = pd.DataFrame(index=ds_labels, columns=col_keys, dtype=float)

    for ds_full, ds_short in zip(DATASETS, ds_labels):
        for key, lbl, grp in heatmap_cols:
            if "__" in key:
                model, strategy = key.split("__", 1)
                path = _get_synthetic_csv(base_path, model, ds_full, strategy=strategy)
            else:
                path = _get_synthetic_csv(base_path, key, ds_full)
            n = count_duplicates_in_csv(path)
            matrix.loc[ds_short, key] = float(n) if n is not None else float("nan")

    return matrix, heatmap_cols


def plot_duplicates_heatmap(base_path, output_path="duplicates_heatmap.pdf"):
    """
    Heatmap: rows = datasets (D1-D28), columns = all methods/strategies.
    Cell colour = log(1+n_duplicates). Raw count annotated inside each cell.
    Vertical white lines separate method groups.
    Group names shown on a secondary x-axis at the top.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    print("\nCostruzione matrice duplicati per heatmap...")
    matrix, heatmap_cols = build_duplicates_matrix(base_path)

    n_ds   = len(DATASETS)
    n_cols = len(heatmap_cols)

    col_labels = [lbl for _, lbl, _ in heatmap_cols]
    row_labels = list(matrix.index)

    data     = matrix.values.astype(float)
    data_log = np.where(np.isnan(data), np.nan, np.log1p(data))

    vmin = 0.0
    vmax = float(np.nanmax(data_log)) if not np.all(np.isnan(data_log)) else 1.0

    # Figure sizing: columns drive width, datasets drive height
    col_w  = 0.27
    row_h  = 0.30
    fig_w  = max(18, n_cols * col_w + 3.0)
    fig_h  = max(6,  n_ds   * row_h + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#ebebeb")

    im = ax.imshow(data_log, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    # Cell annotations
    threshold = (vmin + vmax) * 0.6
    for r in range(n_ds):
        for c in range(n_cols):
            val = data[r, c]
            if np.isnan(val):
                continue
            raw_val = int(val)
            txt     = str(raw_val) if raw_val < 1000 else f"{raw_val/1000:.1f}k"
            color   = "white" if data_log[r, c] > threshold else "#2a2a2a"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=4.2, color=color)

    # X / Y ticks
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=5.5,
                       ha="center", multialignment="center")
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0, pad=3)

    # Vertical group separators
    grp_info  = []
    prev_grp  = None
    grp_start = 0
    for ci, (key, lbl, grp) in enumerate(heatmap_cols):
        if grp != prev_grp:
            if prev_grp is not None:
                grp_info.append((grp_start, ci - 1, prev_grp))
                ax.axvline(x=ci - 0.5, color="white", linewidth=2.0)
            grp_start = ci
            prev_grp  = grp
    grp_info.append((grp_start, n_cols - 1, prev_grp))

    # Group labels on secondary x-axis (top)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([(s + e) / 2 for s, e, _ in grp_info])
    ax2.set_xticklabels([g for _, _, g in grp_info],
                        fontsize=7, fontweight="bold")
    ax2.tick_params(axis="x", length=0)

    # Colourbar with original-scale ticks
    cbar = fig.colorbar(im, ax=ax, fraction=0.010, pad=0.008, aspect=45)
    cbar.set_label("log(1 + #duplicates)", fontsize=8)
    tick_log = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(tick_log)
    cbar.set_ticklabels([str(int(round(np.expm1(v)))) for v in tick_log],
                        fontsize=6)

    ax.set_xlabel("Method / Strategy", fontsize=9, labelpad=8)
    ax.set_ylabel("Dataset", fontsize=9, labelpad=6)
    ax.set_title("Duplicate rows in synthetic data (per dataset x method/strategy)",
                 fontsize=10, pad=16)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Heatmap salvata in: {output_path}")

if __name__ == "__main__":
    main()
    plot_duplicates_heatmap(BASE_PATH, output_path="duplicates_heatmap.pdf")