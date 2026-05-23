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

INPUT_CSV  = "statistiche_per_metodo_mean_std_final.csv"
OUTPUT_TEX = "quality_table_full.tex"
OVERFLOW   = 1e6

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

def make_transposed_table(rows, method_order, metrics_list, label, caption):
    """
    Rows = metrics, Columns = methods.
    Column headers rotated 90°.
    Bold = best value per row (= per metric).
    """
    all_keys = [k for k, _, _ in method_order]
    n_methods = len(all_keys)

    # Pre-compute best per metric
    best = {col: best_value(rows, col, direction, all_keys)
            for col, _, direction, _ in metrics_list}

    # ── Column spec ──────────────────────────────────────────────────────────
    # First col = metric label (left-aligned), rest = one tiny col per method
    col_spec = r"l" + r"r" * n_methods

    # ── Rotated column headers ────────────────────────────────────────────────
    # Group separators via \cmidrule
    prev_grp = None
    grp_spans = []   # list of (grp, start_col_1based, end_col_1based)
    cur_start = 2    # col 1 = metric label; methods start at col 2
    for idx, (key, disp, grp) in enumerate(method_order):
        col = idx + 2
        if grp != prev_grp:
            if prev_grp is not None:
                grp_spans.append((prev_grp, cur_start, col - 1))
            cur_start = col
            prev_grp = grp
    if prev_grp is not None:
        grp_spans.append((prev_grp, cur_start, len(method_order) + 1))

    # Group display names
    GRP_LABEL = {
        "base":  r"\textit{Classic}",
        "deep":  r"\textit{Deep gen.}",
        "llm0":  r"\textit{LLM}",
        "syrfd": r"\textbf{\system}",
    }
    for model in ["devstral-small-2_24b-cloud","gemma3_12b","gemma4_31b_cloud","gpt-oss_20b-cloud"]:
        short = LLM_SHORT.get(model, model[:3])
        GRP_LABEL[f"llm_{model}"] = rf"\textit{{{short}}}"

    # Build header lines
    # Line 1: group spans (multicolumn)
    grp_cells = [r"\multicolumn{1}{l}{}"]  # empty corner
    cmidrules = []
    for grp, s, e in grp_spans:
        span = e - s + 1
        lbl  = GRP_LABEL.get(grp, grp)
        grp_cells.append(rf"\multicolumn{{{span}}}{{c}}{{{lbl}}}")
        cmidrules.append(rf"\cmidrule(lr){{{s}-{e}}}")
    hdr_grp = " & ".join(grp_cells) + r" \\"

    # Line 2: individual method names, rotated
    meth_cells = [r"\textbf{Metric}"]
    for key, disp, grp in method_order:
        meth_cells.append(
            r"\rotatebox{90}{\footnotesize " + disp + r"}"
        )
    hdr_meth = " & ".join(meth_cells) + r" \\"

    # ── Metric group separators ───────────────────────────────────────────────
    # collect row indices where group changes
    grp_change_before = set()
    prev = None
    for ridx, (col, lbl, direction, grp_lbl) in enumerate(metrics_list):
        if prev is not None and grp_lbl != prev:
            grp_change_before.add(ridx)
        prev = grp_lbl

    # ── Assemble table ────────────────────────────────────────────────────────
    L = []
    L.append(r"% ================================================================")
    L.append(r"% quality_table_full.tex  —  transposed, single-page ICDM format")
    L.append(r"% Preamble: \usepackage{booktabs,graphicx,rotating,array,makecell,colortbl}")
    L.append(r"%           \newcommand{\system}{SyRFD}")
    L.append(r"% Bold = best value per row (direction-aware, mean only).")
    L.append(r"% ================================================================")
    L.append(r"")
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\setlength\tabcolsep{1.8pt}")
    L.append(r"\renewcommand{\arraystretch}{0.82}")
    L.append(r"\scalebox{0.56}{")   # tune this if still too wide/tall
    L.append(r"\begin{tabular}{" + col_spec + "}")
    L.append(r"\toprule")
    L.append(hdr_grp)
    L.append(" ".join(cmidrules))
    L.append(hdr_meth)
    L.append(r"\midrule")

    for ridx, (col, lbl, direction, grp_lbl) in enumerate(metrics_list):
        if ridx in grp_change_before:
            L.append(r"\midrule")

        # direction arrow appended to label
        arrow = {"low": r"$\downarrow$", "high": r"$\uparrow$",
                 "approx1": r"$\approx\!1$"}[direction]
        row_cells = [lbl + r"\," + arrow]

        for key, disp, grp in method_order:
            m, _ = parse(rows[key].get(col, "")) if key in rows else (None, None)
            is_best = (best[col] is not None and m is not None
                       and abs(m - best[col]) < 1e-9)
            row_cells.append(fmt_cell(m, is_best))

        L.append(" & ".join(row_cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"}")   # scalebox
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

    # Exclude metrics with overflow in any method
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

    table_tex = make_transposed_table(
        rows, method_order, clean_metrics,
        label="tab:quality_full",
        caption=(
            r"Quality metrics for all methods and LLM strategies "
            r"(mean across datasets). "
            r"\textbf{Bold}: best per metric. "
            r"$\downarrow$ lower is better; $\uparrow$ higher is better; "
            r"$\approx\!1$ closer to 1 is better."
        ),
    )

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(table_tex)

    print(f"Tabella LaTeX salvata in: {OUTPUT_TEX}")
    print(f"  Metodi  : {len(method_order)}")
    print(f"  Metriche: {len(clean_metrics)}")


if __name__ == "__main__":
    main()
