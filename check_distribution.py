"""
augmentation_distribution_test.py
===================================
Analizza se il processo di data augmentation altera la distribuzione
statistica delle singole feature di un dataset.

Per ogni feature vengono prodotti:
  1. Test statistici di confronto (originale vs aumentato)
  2. Identificazione della distribuzione teorica più adatta (fitting su originale)
  3. Raccomandazione del generatore random Python/NumPy più appropriato

Test statistici (numeriche):
  - Kolmogorov-Smirnov (KS)     : confronto non parametrico CDF
  - Anderson-Darling k-sample   : più sensibile alle code
  - Mann-Whitney U               : confronto stochastic ordering
  - Levene                       : uguaglianza delle varianze
  - Shapiro-Wilk (su ciascun campione separatamente)

Test statistici (categoriche/binarie):
  - Chi-quadro di Pearson        : confronto frequenze relative

Uso:
  python augmentation_distribution_test.py \\
      --original  original.csv \\
      --augmented augmented.csv \\
      [--alpha 0.05] \\
      [--output  report.csv] \\
      [--plots]
"""

import argparse
import re
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ks_2samp,
    mannwhitneyu,
    levene,
    anderson_ksamp,
    chi2_contingency,
    shapiro,
    kstest,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# COLORI ANSI
# ════════════════════════════════════════════════════════════════════════════

BOLD   = "\033[1m"
RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
MAGENTA= "\033[95m"
WHITE  = "\033[97m"
DIM    = "\033[2m"

def cprint(text, color=RESET, end="\n"):
    print(f"{color}{text}{RESET}", end=end)

def hline(char="─", n=72, color=CYAN):
    cprint(char * n, color)

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "   "

# ════════════════════════════════════════════════════════════════════════════
# TEST DI CONFRONTO vs TEST DIAGNOSTICI
# ════════════════════════════════════════════════════════════════════════════
# I test di CONFRONTO misurano se le due distribuzioni (orig vs aug) sono
# diverse → contribuiscono al verdetto "Alterata?".
# I test DIAGNOSTICI (Shapiro-Wilk) misurano una proprietà del singolo
# campione (normalità) e NON devono influenzare il verdetto di alterazione.

COMPARISON_TESTS = {"KS", "AD", "MannWhitney", "Levene", "Chi2"}
DIAGNOSTIC_TESTS = {"Shapiro_orig", "Shapiro_aug"}

# ════════════════════════════════════════════════════════════════════════════
# RILEVAMENTO TIPO FEATURE
# ════════════════════════════════════════════════════════════════════════════

def detect_feature_type(series: pd.Series, max_categories: int = 20) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        return "binary" if series.nunique(dropna=True) == 2 else "categorical"
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        if n_unique == 2:
            return "binary"
        if n_unique <= max_categories:
            return "categorical"
        return "numeric"
    return "categorical"

# ════════════════════════════════════════════════════════════════════════════
# FITTING DISTRIBUZIONE TEORICA
# ════════════════════════════════════════════════════════════════════════════

# Catalogo: (nome_scipy, etichetta leggibile, generatore_random, note_parametri)
DISTRIBUTION_CATALOG = [
    ("norm",        "Normale (Gaussiana)",       "numpy.random.normal(loc, scale)",              "loc=μ, scale=σ"),
    ("lognorm",     "Log-Normale",               "numpy.random.lognormal(mean, sigma)",           "mean=log(μ), sigma=log(σ)"),
    ("expon",       "Esponenziale",              "numpy.random.exponential(scale)",               "scale=1/λ"),
    ("gamma",       "Gamma",                     "numpy.random.gamma(shape, scale)",              "shape=α, scale=β"),
    ("beta",        "Beta",                      "numpy.random.beta(a, b)",                       "a=α, b=β  (valori in [0,1])"),
    ("uniform",     "Uniforme Continua",         "numpy.random.uniform(low, high)",               "low=a, high=b"),
    ("triang",      "Triangolare",               "numpy.random.triangular(left, mode, right)",    "c=(mode-left)/(right-left)"),
    ("chi2",        "Chi-quadro",                "numpy.random.chisquare(df)",                    "df=gradi di libertà"),
    ("t",           "t di Student",              "numpy.random.standard_t(df)",                   "df=gradi di libertà"),
    ("f",           "F di Fisher",               "numpy.random.f(dfn, dfd)",                      "dfn, dfd=gradi libertà"),
    ("weibull_min", "Weibull",                   "numpy.random.weibull(a)",                       "a=shape (c in scipy)"),
    ("pareto",      "Pareto",                    "numpy.random.pareto(a)",                        "a=shape"),
    ("laplace",     "Laplace (Double Exp.)",     "numpy.random.laplace(loc, scale)",              "loc=μ, scale=b"),
    ("logistic",    "Logistica",                 "numpy.random.logistic(loc, scale)",             "loc=μ, scale=s"),
    ("cauchy",      "Cauchy",                    "random: scipy.stats.cauchy.rvs(loc, scale)",    "loc=x0, scale=γ  (nessuna media)"),
    ("poisson",     "Poisson (discreta)",        "numpy.random.poisson(lam)",                     "lam=λ (su dati interi non-neg.)"),
    ("nbinom",      "Neg. Binomiale (discreta)", "numpy.random.negative_binomial(n, p)",          "n, p"),
]

# Distribuzione discreta → trattamento speciale
DISCRETE_DISTS = {"poisson", "nbinom"}

# Mappa rapida per lookup
DIST_MAP = {d[0]: d for d in DISTRIBUTION_CATALOG}


def fit_distribution(data: np.ndarray, top_n: int = 3) -> list[dict]:
    """
    Prova a fittare ogni distribuzione del catalogo sui dati (continui).
    Restituisce i top_n candidati ordinati per p-value KS decrescente.
    """
    results = []
    data = data[np.isfinite(data)]
    if len(data) < 10:
        return []

    for dist_name, label, generator, param_note in DISTRIBUTION_CATALOG:
        if dist_name in DISCRETE_DISTS:
            continue  # gestite separatamente
        try:
            dist_obj = getattr(stats, dist_name)
            params   = dist_obj.fit(data)
            ks_stat, ks_p = kstest(data, dist_name, args=params)
            results.append({
                "dist_name":  dist_name,
                "label":      label,
                "params":     params,
                "ks_stat":    ks_stat,
                "ks_p":       ks_p,
                "generator":  generator,
                "param_note": param_note,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["ks_p"], reverse=True)
    return results[:top_n]


def fit_discrete(data: np.ndarray) -> dict | None:
    """
    Tenta fitting Poisson e Negative Binomiale per dati interi non negativi.
    """
    data = data[np.isfinite(data)].astype(int)
    if data.min() < 0 or len(data) < 10:
        return None

    best = None
    best_p = -1.0

    # Poisson: λ = media
    lam = data.mean()
    if lam > 0:
        expected_p = stats.poisson.pmf(np.arange(data.max() + 1), lam)
        observed_c = np.bincount(data, minlength=data.max() + 1)
        # chi2 GoF
        mask = expected_p * len(data) >= 5
        if mask.sum() >= 2:
            chi2_val, p_val = stats.chisquare(
                observed_c[mask], f_exp=expected_p[mask] * len(data)
            )
            if p_val > best_p:
                best_p = p_val
                best = {
                    "dist_name": "poisson",
                    "label":     "Poisson",
                    "params":    (lam,),
                    "ks_p":      p_val,
                    "generator": f"numpy.random.poisson(lam={lam:.4f})",
                    "param_note": f"lam={lam:.4f}",
                }

    return best


def format_params(dist_name: str, params: tuple) -> str:
    """Formatta i parametri di scipy in modo leggibile."""
    dist_obj = getattr(stats, dist_name, None)
    if dist_obj is None:
        return str(params)

    shapes = dist_obj.shapes.split(",") if dist_obj.shapes else []
    shapes = [s.strip() for s in shapes]
    # scipy: shapes..., loc, scale
    named = {}
    for i, s in enumerate(shapes):
        if i < len(params):
            named[s] = params[i]
    if len(params) >= len(shapes) + 1:
        named["loc"]   = params[-2]
    if len(params) >= len(shapes) + 2:
        named["scale"] = params[-1]

    return ", ".join(f"{k}={v:.4f}" for k, v in named.items())


def recommend_generator(dist_name: str, params: tuple) -> str:
    """Restituisce la chiamata Python pronta all'uso per generare campioni."""
    try:
        p = params
        if dist_name == "norm":
            return (f"numpy.random.normal(loc={p[-2]:.4f}, scale={p[-1]:.4f})\n"
                    f"   # oppure: random.gauss({p[-2]:.4f}, {p[-1]:.4f})")
        if dist_name == "lognorm":
            # scipy lognorm: params = (s, loc, scale); mean_log=log(scale), sigma_log=s
            sigma = p[0]; mu_log = np.log(p[-1]) if p[-1] > 0 else 0
            return (f"numpy.random.lognormal(mean={mu_log:.4f}, sigma={sigma:.4f})\n"
                    f"   # (mean e sigma sono nella scala logaritmica)")
        if dist_name == "expon":
            scale = p[-1]
            return (f"numpy.random.exponential(scale={scale:.4f})\n"
                    f"   # λ = {1/scale:.4f}  →  random.expovariate({1/scale:.4f})")
        if dist_name == "gamma":
            a = p[0]; scale = p[-1]
            return f"numpy.random.gamma(shape={a:.4f}, scale={scale:.4f})"
        if dist_name == "beta":
            a, b = p[0], p[1]; loc = p[-2]; scale_b = p[-1]
            note = ""
            if abs(loc) > 1e-6 or abs(scale_b - 1) > 1e-3:
                note = f"\n   # Nota: i dati non sono in [0,1]; rescala con *{scale_b:.4f}+{loc:.4f}"
            return f"numpy.random.beta(a={a:.4f}, b={b:.4f}){note}"
        if dist_name == "uniform":
            low = p[-2]; high = p[-2] + p[-1]
            return (f"numpy.random.uniform(low={low:.4f}, high={high:.4f})\n"
                    f"   # oppure: random.uniform({low:.4f}, {high:.4f})")
        if dist_name == "triang":
            c = p[0]; loc = p[-2]; scale_t = p[-1]
            left = loc; right = loc + scale_t; mode = loc + c * scale_t
            return f"numpy.random.triangular(left={left:.4f}, mode={mode:.4f}, right={right:.4f})"
        if dist_name == "chi2":
            df = p[0]
            return f"numpy.random.chisquare(df={df:.2f})"
        if dist_name == "t":
            df = p[0]
            return f"numpy.random.standard_t(df={df:.2f})  # poi scalare/traslare manualmente"
        if dist_name == "weibull_min":
            c = p[0]
            return f"numpy.random.weibull(a={c:.4f})  # shape=c={c:.4f}"
        if dist_name == "pareto":
            b = p[0]
            return f"numpy.random.pareto(a={b:.4f})"
        if dist_name == "laplace":
            return f"numpy.random.laplace(loc={p[-2]:.4f}, scale={p[-1]:.4f})"
        if dist_name == "logistic":
            return f"numpy.random.logistic(loc={p[-2]:.4f}, scale={p[-1]:.4f})"
        if dist_name == "poisson":
            return f"numpy.random.poisson(lam={p[0]:.4f})"
        # fallback generico
        return f"scipy.stats.{dist_name}.rvs({format_params(dist_name, params)}, size=N)"
    except Exception:
        return f"scipy.stats.{dist_name}.rvs(*params, size=N)"


# ════════════════════════════════════════════════════════════════════════════
# TEST STATISTICI — NUMERICHE
# ════════════════════════════════════════════════════════════════════════════

KS_DESCRIPTION = (
    "Confronta le CDF empiriche dei due campioni. "
    "H₀: le distribuzioni sono identiche."
)
AD_DESCRIPTION = (
    "Versione k-sample del test Anderson-Darling; più potente di KS sulle code. "
    "H₀: i campioni provengono dalla stessa distribuzione."
)
MW_DESCRIPTION = (
    "Test non parametrico sul rango. "
    "H₀: P(X > Y) = P(Y > X)  (stochastic ordering)."
)
LEV_DESCRIPTION = (
    "Testa l'omoscedasticità tramite le deviazioni dalla mediana. "
    "H₀: le varianze dei due gruppi sono uguali."
)
SW_DESCRIPTION = (
    "Testa la normalità del singolo campione (applicato a ciascuno separatamente). "
    "H₀: il campione proviene da una distribuzione normale."
)


def test_numeric(orig: np.ndarray, aug: np.ndarray, alpha: float) -> dict:
    results = {}

    # ── KS ──────────────────────────────────────────────────────────────────
    ks_stat, ks_p = ks_2samp(orig, aug)
    results["KS"] = {
        "label":       "Kolmogorov-Smirnov",
        "description": KS_DESCRIPTION,
        "H0":          "Le due distribuzioni sono identiche",
        "statistic":   ks_stat,
        "stat_label":  "D",
        "p_value":     ks_p,
        "reject_H0":   ks_p < alpha,
    }

    # ── Anderson-Darling k-sample ─────────────────────────────────────────
    try:
        ad_res  = anderson_ksamp([orig, aug])
        ad_stat = ad_res.statistic
        ad_p    = ad_res.significance_level
        if ad_p > 1:
            ad_p /= 100.0
        results["AD"] = {
            "label":       "Anderson-Darling k-sample",
            "description": AD_DESCRIPTION,
            "H0":          "I campioni provengono dalla stessa distribuzione",
            "statistic":   ad_stat,
            "stat_label":  "AD",
            "p_value":     ad_p,
            "reject_H0":   ad_p < alpha,
        }
    except Exception as e:
        results["AD"] = {
            "label": "Anderson-Darling k-sample",
            "description": AD_DESCRIPTION,
            "H0": "—", "statistic": np.nan, "stat_label": "AD",
            "p_value": np.nan, "reject_H0": False, "error": str(e),
        }

    # ── Mann-Whitney U ───────────────────────────────────────────────────
    mw_stat, mw_p = mannwhitneyu(orig, aug, alternative="two-sided")
    results["MannWhitney"] = {
        "label":       "Mann-Whitney U",
        "description": MW_DESCRIPTION,
        "H0":          "Le due distribuzioni hanno lo stesso stochastic ordering",
        "statistic":   mw_stat,
        "stat_label":  "U",
        "p_value":     mw_p,
        "reject_H0":   mw_p < alpha,
    }

    # ── Levene ────────────────────────────────────────────────────────────
    lev_stat, lev_p = levene(orig, aug)
    results["Levene"] = {
        "label":       "Levene (varianze)",
        "description": LEV_DESCRIPTION,
        "H0":          "Le varianze dei due campioni sono uguali",
        "statistic":   lev_stat,
        "stat_label":  "W",
        "p_value":     lev_p,
        "reject_H0":   lev_p < alpha,
    }

    # ── Shapiro-Wilk su ciascun campione (max 5 000 osservazioni) ─────────
    for tag, arr in [("orig", orig), ("aug", aug)]:
        sample = arr[:5000] if len(arr) > 5000 else arr
        try:
            sw_stat, sw_p = shapiro(sample)
        except Exception:
            sw_stat, sw_p = np.nan, np.nan
        results[f"Shapiro_{tag}"] = {
            "label":       f"Shapiro-Wilk ({tag})",
            "description": SW_DESCRIPTION,
            "H0":          "Il campione è normalmente distribuito",
            "statistic":   sw_stat,
            "stat_label":  "W",
            "p_value":     sw_p,
            "reject_H0":   sw_p < alpha if not np.isnan(sw_p) else False,
        }

    return results


# ════════════════════════════════════════════════════════════════════════════
# TEST STATISTICI — CATEGORICHE
# ════════════════════════════════════════════════════════════════════════════

CHI2_DESCRIPTION = (
    "Confronta le distribuzioni di frequenza relativa delle categorie. "
    "H₀: la distribuzione delle categorie è invariata."
)


def test_categorical(orig: pd.Series, aug: pd.Series, alpha: float) -> dict:
    results = {}
    all_cats   = set(orig.dropna().unique()) | set(aug.dropna().unique())
    orig_cnts  = orig.value_counts().reindex(all_cats, fill_value=0)
    aug_cnts   = aug.value_counts().reindex(all_cats, fill_value=0)
    contingency = pd.DataFrame({"original": orig_cnts, "augmented": aug_cnts}).T
    try:
        chi2, p, dof, expected = chi2_contingency(contingency)
        results["Chi2"] = {
            "label":       "Chi-quadro di Pearson",
            "description": CHI2_DESCRIPTION,
            "H0":          "Le distribuzioni di categoria sono identiche",
            "statistic":   chi2,
            "stat_label":  "χ²",
            "dof":         dof,
            "p_value":     p,
            "reject_H0":   p < alpha,
        }
    except Exception as e:
        results["Chi2"] = {
            "label": "Chi-quadro di Pearson",
            "description": CHI2_DESCRIPTION,
            "H0": "—", "statistic": np.nan, "stat_label": "χ²",
            "p_value": np.nan, "reject_H0": False, "error": str(e),
        }
    return results


# ════════════════════════════════════════════════════════════════════════════
# STATISTICHE DESCRITTIVE
# ════════════════════════════════════════════════════════════════════════════

def descriptive_stats(orig: pd.Series, aug: pd.Series) -> dict:
    num = pd.api.types.is_numeric_dtype
    return {
        "orig_n":       len(orig.dropna()),
        "aug_n":        len(aug.dropna()),
        "orig_mean":    orig.mean()     if num(orig) else np.nan,
        "aug_mean":     aug.mean()      if num(aug)  else np.nan,
        "orig_std":     orig.std()      if num(orig) else np.nan,
        "aug_std":      aug.std()       if num(aug)  else np.nan,
        "orig_median":  orig.median()   if num(orig) else np.nan,
        "aug_median":   aug.median()    if num(aug)  else np.nan,
        "orig_skew":    orig.skew()     if num(orig) else np.nan,
        "aug_skew":     aug.skew()      if num(aug)  else np.nan,
        "orig_kurt":    orig.kurtosis() if num(orig) else np.nan,
        "aug_kurt":     aug.kurtosis()  if num(aug)  else np.nan,
        "orig_min":     orig.min()      if num(orig) else np.nan,
        "aug_min":      aug.min()       if num(aug)  else np.nan,
        "orig_max":     orig.max()      if num(orig) else np.nan,
        "aug_max":      aug.max()       if num(aug)  else np.nan,
    }


# ════════════════════════════════════════════════════════════════════════════
# STAMPA REPORT PER FEATURE
# ════════════════════════════════════════════════════════════════════════════

def categorical_generator_info(series: pd.Series, ftype: str) -> dict:
    """
    Calcola la distribuzione empirica di una feature categorica/binaria
    e restituisce le informazioni per il generatore random consigliato.
    """
    counts = series.dropna().value_counts()
    total  = counts.sum()
    categories = counts.index.tolist()
    probs      = (counts / total).tolist()

    # Rappresentazione compatta delle probabilità
    cats_repr  = "[" + ", ".join(repr(c) for c in categories) + "]"
    probs_repr = "[" + ", ".join(f"{p:.4f}" for p in probs) + "]"

    if ftype == "binary":
        c0, c1 = categories[0], categories[1]
        p0, p1 = probs[0], probs[1]
        dist_label = f"Bernoulli  (p={p1:.4f} per '{c1}')"
        gen_lines = [
            f"numpy.random.choice({cats_repr}, p={probs_repr})",
            f"   # oppure, se codificata 0/1:",
            f"   numpy.random.binomial(n=1, p={p1:.4f})",
            f"   # oppure: random.choices({cats_repr}, weights={probs_repr})[0]",
        ]
    else:
        dist_label = f"Categorica  ({len(categories)} classi)"
        gen_lines = [
            f"numpy.random.choice(",
            f"    {cats_repr},",
            f"    p={probs_repr}",
            f")",
            f"   # oppure: random.choices({cats_repr}, weights={probs_repr})[0]",
        ]

    return {
        "dist_label":  dist_label,
        "categories":  categories,
        "probs":       probs,
        "cats_repr":   cats_repr,
        "probs_repr":  probs_repr,
        "gen_lines":   gen_lines,
    }


def print_feature_report(feature: str, ftype: str, desc: dict,
                         results: dict, fit_results: list,
                         alpha: float, orig_series: pd.Series = None):
    W = 72
    hline("═", W)
    cprint(f"  FEATURE: {feature}  [{ftype.upper()}]", BOLD + WHITE)
    hline("═", W)

    # ── Statistiche descrittive ─────────────────────────────────────────────
    cprint("  ▸ Statistiche descrittive", CYAN + BOLD)
    cprint(f"  {'Metrica':<16}  {'Originale':>14}  {'Aumentato':>14}  {'Δ':>12}", DIM)
    hline("·", W, DIM)

    def _row(label, k_o, k_a):
        v_o = desc[k_o]; v_a = desc[k_a]
        if np.isnan(v_o) or np.isnan(v_a):
            return
        delta = v_a - v_o
        delta_str = f"{delta:+.4f}"
        color = YELLOW if abs(delta) / (abs(v_o) + 1e-10) > 0.05 else RESET
        print(f"  {label:<16}  {v_o:>14.4f}  {v_a:>14.4f}  "
              f"{color}{delta_str:>12}{RESET}")

    if ftype == "numeric":
        _row("N",       "orig_n",      "aug_n")
        _row("Media",   "orig_mean",   "aug_mean")
        _row("Dev.Std", "orig_std",    "aug_std")
        _row("Mediana", "orig_median", "aug_median")
        _row("Skewness","orig_skew",   "aug_skew")
        _row("Kurtosis","orig_kurt",   "aug_kurt")
        _row("Min",     "orig_min",    "aug_min")
        _row("Max",     "orig_max",    "aug_max")
    else:
        print(f"  N orig  : {desc['orig_n']}")
        print(f"  N aug   : {desc['aug_n']}")

    # ── Test statistici ─────────────────────────────────────────────────────
    print()
    cprint("  ▸ Test statistici di confronto  (orig vs aug)", CYAN + BOLD)
    cprint("    Contribuiscono al verdetto 'Alterata?'", DIM)

    # Stampa prima i test di confronto, poi quelli diagnostici con separatore
    comparison_items  = [(k, v) for k, v in results.items() if k in COMPARISON_TESTS]
    diagnostic_items  = [(k, v) for k, v in results.items() if k in DIAGNOSTIC_TESTS]
    ordered_items = comparison_items + diagnostic_items

    printed_diag_header = False
    for test_key, res in ordered_items:
        if test_key in DIAGNOSTIC_TESTS and not printed_diag_header:
            printed_diag_header = True
            print()
            cprint("  ▸ Test diagnostici  (singolo campione — NON influenzano 'Alterata?')", CYAN + BOLD)
            cprint("    Shapiro-Wilk testa la normalità di ciascun campione separatamente,", DIM)
            cprint("    non il confronto tra i due. Un rifiuto indica non-normalità, non variazione.", DIM)

        label = res.get("label", test_key)
        descr = res.get("description", "")
        h0    = res.get("H0", "")
        stat  = res.get("statistic", np.nan)
        slbl  = res.get("stat_label", "stat")
        pv    = res.get("p_value", np.nan)
        rej   = res.get("reject_H0", False)
        dof   = res.get("dof", None)
        err   = res.get("error", None)

        hline("─", W, DIM)
        cprint(f"  [{label}]", BOLD)
        for line in textwrap.wrap(descr, width=68):
            cprint(f"    {line}", DIM)
        cprint(f"    H₀  : {h0}", DIM)

        if err:
            cprint(f"    ⚠  Errore: {err}", YELLOW)
            continue

        if not np.isnan(stat):
            dof_str = f"  df={dof}" if dof is not None else ""
            print(f"    {slbl} = {stat:.6f}{dof_str}")
        if not np.isnan(pv):
            sig = stars(pv)
            if rej:
                cprint(f"    p = {pv:.6f}  {sig}  →  H₀ RIFIUTATA  ✗  (p < α={alpha})", RED + BOLD)
            else:
                cprint(f"    p = {pv:.6f}  {sig}  →  H₀ non rifiutata  ✓  (p ≥ α={alpha})", GREEN)

        interp = _interpret_test(test_key, pv, rej, alpha, desc)
        if interp:
            cprint(f"    ↳ {interp}", YELLOW if rej else DIM)

    # ── Fitting distribuzione ────────────────────────────────────────────────
    print()
    cprint("  ▸ Distribuzione teorica più adatta (fitting sul dataset originale)", CYAN + BOLD)
    hline("─", W, DIM)

    if not fit_results:
        # Feature categorica/binaria: generatore basato sulla distribuzione empirica
        if orig_series is not None and ftype in ("binary", "categorical"):
            cat_info = categorical_generator_info(orig_series, ftype)
            cprint(f"    Distribuzione : {cat_info['dist_label']}", MAGENTA + BOLD)
            print()
            # Tabella frequenze
            cprint(f"    {'Categoria':<20}  {'Freq. rel.':>10}  {'Freq. %':>8}", DIM)
            hline("·", 50, DIM)
            for cat, prob in zip(cat_info["categories"], cat_info["probs"]):
                bar = "█" * int(prob * 30)
                print(f"    {str(cat):<20}  {prob:>10.4f}  {prob*100:>7.2f}%  {bar}")
            print()
            cprint("  ▸ Generatore random consigliato", CYAN + BOLD)
            hline("─", W, DIM)
            for line in cat_info["gen_lines"]:
                cprint(f"    {line}", MAGENTA if not line.strip().startswith("#") else DIM)
        else:
            cprint("    (non disponibile per questo tipo di feature)", DIM)
    else:
        for rank, fr in enumerate(fit_results, 1):
            is_best = (rank == 1)
            tag_color = MAGENTA + BOLD if is_best else DIM
            cprint(f"    #{rank}  {fr['label']}", tag_color)
            print(f"         Parametri stimati : {format_params(fr['dist_name'], fr['params'])}")
            print(f"         KS goodness-of-fit: D={fr['ks_stat']:.5f},  p={fr['ks_p']:.5f}  "
                  f"({'non rifiutata ✓' if fr['ks_p'] >= alpha else 'rifiutata ✗'})")
            if is_best:
                cprint(f"         Nota parametri    : {fr['param_note']}", DIM)

        # ── Generatore random raccomandato ───────────────────────────────────
        best = fit_results[0]
        print()
        cprint("  ▸ Generatore random consigliato", CYAN + BOLD)
        hline("─", W, DIM)
        gen_code = recommend_generator(best["dist_name"], best["params"])
        for line in gen_code.split("\n"):
            cprint(f"    {line}", MAGENTA if not line.strip().startswith("#") else DIM)
        # fallback scipy generico
        param_str = ", ".join(f"{v:.4f}" for v in best["params"])
        cprint(f"\n    Alternativa scipy:", DIM)
        cprint(f"    scipy.stats.{best['dist_name']}.rvs({param_str}, size=N)", DIM)

    print()


def _interpret_test(test_key: str, pv: float, rej: bool,
                    alpha: float, desc: dict) -> str:
    """Aggiunge una riga di interpretazione contestuale."""
    if np.isnan(pv):
        return ""
    if test_key == "KS":
        if rej:
            return "La forma globale della distribuzione è cambiata dopo l'augmentation."
        return "La forma globale non mostra variazioni significative."
    if test_key == "AD":
        if rej:
            return "Le code della distribuzione risultano alterate (test più potente di KS sulle code)."
        return "Nessuna differenza rilevante nelle code."
    if test_key == "MannWhitney":
        om, am = desc.get("orig_median"), desc.get("aug_median")
        if rej and not (np.isnan(om) or np.isnan(am)):
            shift = am - om
            return f"Shift mediana: {om:.4f} → {am:.4f}  (Δ={shift:+.4f})."
        if rej:
            return "L'ordinamento stocastico è cambiato (possibile shift della mediana)."
        return "Nessuna variazione significativa nella posizione centrale."
    if test_key == "Levene":
        os, as_ = desc.get("orig_std"), desc.get("aug_std")
        if rej and not (np.isnan(os) or np.isnan(as_)):
            return f"Varianza alterata: std {os:.4f} → {as_:.4f}."
        if rej:
            return "La dispersione (varianza) è cambiata dopo l'augmentation."
        return "Nessuna variazione significativa nella dispersione."
    if test_key.startswith("Shapiro"):
        tag = "originale" if "orig" in test_key else "aumentato"
        if rej:
            return f"Il campione {tag} NON segue una distribuzione normale."
        return f"Il campione {tag} è compatibile con la normalità."
    if test_key == "Chi2":
        if rej:
            return "Le proporzioni delle categorie sono cambiate dopo l'augmentation."
        return "Le proporzioni categoriche sono rimaste stabili."
    return ""


# ════════════════════════════════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════════════════════════════════

def plot_feature(feature: str, orig: pd.Series, aug: pd.Series,
                 ftype: str, results: dict, fit_results: list,
                 alpha: float, out_dir: Path):
    if not MATPLOTLIB_AVAILABLE:
        return

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"Feature: {feature}  [{ftype}]", fontsize=13, fontweight="bold")
    gs = GridSpec(1, 4, figure=fig, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

    orig_c = orig.dropna()
    aug_c  = aug.dropna()

    if ftype == "numeric":
        bins = min(50, max(10, int(np.sqrt(len(orig_c)))))
        lo   = min(orig_c.min(), aug_c.min())
        hi   = max(orig_c.max(), aug_c.max())
        edges = np.linspace(lo, hi, bins + 1)

        ax1.hist(orig_c, bins=edges, alpha=0.55, density=True, label="original",  color="#4C72B0")
        ax1.hist(aug_c,  bins=edges, alpha=0.55, density=True, label="augmented", color="#DD8452")

        # Sovrapponi la PDF teorica migliore
        if fit_results:
            best = fit_results[0]
            x_plot = np.linspace(lo, hi, 300)
            try:
                dist_obj = getattr(stats, best["dist_name"])
                pdf_vals = dist_obj.pdf(x_plot, *best["params"])
                ax1.plot(x_plot, pdf_vals, "k--", lw=1.5,
                         label=f"fit: {best['label'][:12]}")
            except Exception:
                pass
        ax1.set_title("Distribuzione (densità)")
        ax1.legend(fontsize=7)

        # CDF empirica
        for data, label, color in [(orig_c, "original", "#4C72B0"),
                                   (aug_c,  "augmented", "#DD8452")]:
            sd  = np.sort(data)
            cdf = np.arange(1, len(sd) + 1) / len(sd)
            ax2.plot(sd, cdf, label=label, color=color)
        ax2.set_title("CDF empirica")
        ax2.legend(fontsize=7)

        # Box plot
        bp = ax3.boxplot([orig_c, aug_c], labels=["orig", "aug"],
                         patch_artist=True, widths=0.4,
                         medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor("#4C72B0"); bp["boxes"][0].set_alpha(0.55)
        bp["boxes"][1].set_facecolor("#DD8452"); bp["boxes"][1].set_alpha(0.55)
        ax3.set_title("Box plot")

    else:
        cats = sorted(set(orig_c.unique()) | set(aug_c.unique()), key=str)
        x = np.arange(len(cats)); w = 0.35
        of = orig_c.value_counts(normalize=True).reindex(cats, fill_value=0)
        af = aug_c.value_counts(normalize=True).reindex(cats,  fill_value=0)
        ax1.bar(x - w/2, of.values, w, label="original",  color="#4C72B0", alpha=0.8)
        ax1.bar(x + w/2, af.values, w, label="augmented", color="#DD8452", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(c) for c in cats], rotation=30, ha="right", fontsize=7)
        ax1.set_title("Frequenza relativa")
        ax1.legend(fontsize=7)
        for ax in (ax2, ax3): ax.axis("off")

    # Pannello test
    ax4.axis("off")
    lines = [f"α = {alpha}"]
    for test_key, res in results.items():
        pv  = res.get("p_value", np.nan)
        rej = res.get("reject_H0", False)
        lbl = res.get("label", test_key)[:18]
        slbl= res.get("stat_label", "S")
        st  = res.get("statistic", np.nan)
        if np.isnan(pv): continue
        verdict = "✗ RIFIUTATA" if rej else "✓ ok"
        lines.append(f"{lbl}\n  {slbl}={st:.4f}  p={pv:.4f}  {verdict}")
    if fit_results:
        best = fit_results[0]
        lines.append(f"\nBest fit: {best['label']}\n  p_KS={best['ks_p']:.4f}")
    ax4.text(0.03, 0.97, "\n".join(lines),
             transform=ax4.transAxes, fontsize=7.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7", alpha=0.9))
    ax4.set_title("Riepilogo test & fit")

    fname = out_dir / f"feat_{feature.replace('/', '_').replace(' ', '_')}.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# BUILD CSV REPORT
# ════════════════════════════════════════════════════════════════════════════

def build_report_row(feature: str, ftype: str, desc: dict,
                     results: dict, fit_results: list,
                     orig_name: str = "", aug_name: str = "",
                     orig_series: pd.Series = None) -> dict:
    row = {"dataset_originale": orig_name, "dataset_aumentato": aug_name,
           "feature": feature, "type": ftype}
    row.update({k: v for k, v in desc.items()})
    for test_key, res in results.items():
        row[f"{test_key}_stat"]   = res.get("statistic", np.nan)
        row[f"{test_key}_pvalue"] = res.get("p_value",   np.nan)
        row[f"{test_key}_reject"] = res.get("reject_H0", False)
    if fit_results:
        best = fit_results[0]
        row["best_dist"]      = best["label"]
        row["best_dist_ks_p"] = best["ks_p"]
        row["best_generator"] = recommend_generator(best["dist_name"], best["params"]).split("\n")[0]
    elif ftype in ("binary", "categorical") and orig_series is not None:
        cat_info = categorical_generator_info(orig_series, ftype)
        row["best_dist"]      = "Categorica empirica" if ftype == "categorical" else "Bernoulli empirica"
        row["best_dist_ks_p"] = np.nan
        # Generatore con valori reali delle probabilità
        row["best_generator"] = (
            f"numpy.random.choice({cat_info['cats_repr']}, p={cat_info['probs_repr']})"
        )
    return row


# ════════════════════════════════════════════════════════════════════════════
# RIEPILOGO FINALE
# ════════════════════════════════════════════════════════════════════════════

def print_summary(report_rows: list, alpha: float, all_test_keys: list):
    W = 110
    # Solo i test di confronto determinano "Alterata?"
    verdict_keys = [t for t in all_test_keys if t in COMPARISON_TESTS]

    print()
    hline("═", W)
    cprint("  RIEPILOGO FINALE", BOLD + WHITE)
    hline("═", W)

    n_total   = len(report_rows)
    n_altered = sum(
        1 for r in report_rows
        if any(r.get(f"{t}_reject", False) for t in verdict_keys)
    )

    cprint(f"  Feature analizzate         : {n_total}")
    color = RED if n_altered > 0 else GREEN
    cprint(f"  Feature con dist. alterata : {n_altered}", color)
    cprint(f"  Feature stabili            : {n_total - n_altered}", GREEN)
    cprint(f"  (Verdetto basato su: {', '.join(sorted(verdict_keys))})", DIM)

    # Tabella per feature
    print()
    cprint(f"  {'Feature':<22}  {'Tipo':<12}  {'Best fit':<22}  {'Generatore consigliato':<38}  {'Alterata?'}", BOLD)
    hline("─", W, DIM)
    for r in report_rows:
        feat  = r["feature"][:22]
        ftype = r["type"][:12]
        bfit  = str(r.get("best_dist", "—"))[:22]
        gen   = str(r.get("best_generator", "—"))[:38]
        any_rej = any(r.get(f"{t}_reject", False) for t in verdict_keys)
        color = RED if any_rej else GREEN
        verdict = "✗ SÌ" if any_rej else "✓ no"
        cprint(f"  {feat:<22}  {ftype:<12}  {bfit:<22}  {gen:<38}  {verdict}", color)

    # Dettaglio feature alterate
    altered = [r for r in report_rows
               if any(r.get(f"{t}_reject", False) for t in verdict_keys)]
    if altered:
        print()
        cprint("  Feature con variazione significativa:", RED + BOLD)
        for r in altered:
            tests_failed = [t for t in verdict_keys if r.get(f"{t}_reject", False)]
            cprint(f"    • {r['feature']}  →  test falliti: {', '.join(tests_failed)}", RED)
    else:
        print()
        cprint("  ✓ Nessuna feature mostra variazione di distribuzione.", GREEN + BOLD)

    # Riepilogo Shapiro (diagnostico, separato)
    shapiro_nonnorm = [
        r for r in report_rows
        if r.get("Shapiro_orig_reject", False) or r.get("Shapiro_aug_reject", False)
    ]
    if shapiro_nonnorm:
        print()
        cprint("  ℹ  Feature non-normali (Shapiro-Wilk, solo diagnostico):", YELLOW + BOLD)
        cprint("     Queste feature non sono gaussiane → utile per scegliere il generatore.", DIM)
        for r in shapiro_nonnorm:
            tags = []
            if r.get("Shapiro_orig_reject"): tags.append("orig")
            if r.get("Shapiro_aug_reject"):  tags.append("aug")
            cprint(f"    • {r['feature']}  ({', '.join(tags)})", YELLOW)

    hline("═", W)
    print()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Test statistici sulla distribuzione delle feature "
                    "prima e dopo data augmentation."
    )
    p.add_argument("--original",  required=True,  help="CSV del dataset originale")
    p.add_argument("--augmented", required=True,  help="CSV del dataset aumentato")
    p.add_argument("--alpha",     type=float, default=0.05,
                   help="Livello α (default: 0.05)")
    p.add_argument("--output",    default=None,
                   help="Percorso CSV di output per il report")
    p.add_argument("--plots",     action="store_true",
                   help="Genera PNG per ogni feature")
    p.add_argument("--plot-dir",  default="augmentation_plots",
                   help="Cartella plot (default: augmentation_plots)")
    p.add_argument("--sep",       default=",",
                   help="Separatore CSV (default: ',')")
    p.add_argument("--features",  nargs="*", default=None,
                   help="Lista feature da testare (default: tutte le comuni)")
    p.add_argument("--top-fits",  type=int, default=3,
                   help="Numero di distribuzioni candidate nel ranking (default: 3)")
    return p.parse_args()


def make_output_paths(original: str, augmented: str) -> tuple[Path, Path, Path]:
    """
    Restituisce (out_dir, csv_path, log_path) dove:
      - out_dir  = distribution_analysis/
      - csv_path = distribution_analysis/report_<dataset>_<algo>_<timestamp>.csv
      - log_path = distribution_analysis/<dataset>_<algo>_<timestamp>.log
    <dataset> = nome base del file originale (senza estensione)
    <algo>    = nome della directory padre del file aumentato
    """
    from datetime import datetime
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset  = Path(original).stem                       # es. "dermatology-6"
    aug_path = Path(augmented)
    # parent directory del file aumentato → nome algoritmo
    # se il file è nella cwd, usiamo "unknown_algo"
    algo = aug_path.parent.name if aug_path.parent.name not in (".", "") else "unknown_algo"

    out_dir  = Path("distribution_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    stem     = f"{dataset}__{algo}__{ts}"
    csv_path = out_dir / f"report_{stem}.csv"
    log_path = out_dir / f"{stem}.log"
    return out_dir, csv_path, log_path


class TeeLogger:
    """Scrive contemporaneamente su stdout e su un file di log (testo puro, senza codici ANSI)."""

    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, log_path: Path):
        self._log   = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, text: str):
        self._stdout.write(text)
        self._log.write(self.ANSI_RE.sub("", text))

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        self._log.close()

    # consente l'uso con isatty() da parte di librerie esterne
    def isatty(self):
        return False


def main():
    args = parse_args()

    # ── Cartella output e percorsi automatici ────────────────────────────────
    out_dir, auto_csv, log_path = make_output_paths(args.original, args.augmented)

    # Se --output non è specificato usiamo il nome automatico
    csv_path = Path(args.output) if args.output else auto_csv

    # Avvia il tee: da qui in poi tutto ciò che va su stdout finisce anche nel log
    tee = TeeLogger(log_path)
    sys.stdout = tee

    W = 72
    hline("═", W)
    cprint("  AUGMENTATION DISTRIBUTION TEST", BOLD + WHITE)
    cprint("  Confronto distribuzione feature: originale vs aumentato", DIM)
    hline("═", W)

    try:
        df_orig = pd.read_csv(args.original, sep=args.sep)
        df_aug  = pd.read_csv(args.augmented, sep=args.sep)
    except Exception as e:
        cprint(f"Errore nella lettura dei file: {e}", RED)
        sys.stdout = tee._stdout
        tee.close()
        sys.exit(1)

    cprint(f"\n  Dataset originale : {df_orig.shape[0]:,} righe × {df_orig.shape[1]} colonne  ({args.original})")
    cprint(f"  Dataset aumentato : {df_aug.shape[0]:,} righe × {df_aug.shape[1]} colonne  ({args.augmented})")
    cprint(f"  Livello α         : {args.alpha}")
    cprint(f"  Top candidati fit : {args.top_fits}")
    cprint(f"  Log               : {log_path.resolve()}", DIM)
    cprint(f"  Report CSV        : {csv_path.resolve()}", DIM)

    if df_aug.shape[0] <= df_orig.shape[0]:
        cprint("\n  ⚠  Il dataset aumentato non ha più righe dell'originale!", YELLOW)

    # ── Feature selezionate ──────────────────────────────────────────────────
    common = list(set(df_orig.columns) & set(df_aug.columns))
    if args.features:
        selected = [f for f in args.features if f in common]
        missing  = [f for f in args.features if f not in common]
        if missing:
            cprint(f"\n  ⚠  Feature mancanti: {missing}", YELLOW)
    else:
        selected = common
    selected = sorted(selected)
    cprint(f"\n  Feature analizzate: {len(selected)}\n")

    # ── Cartella plot ────────────────────────────────────────────────────────
    if args.plots:
        if not MATPLOTLIB_AVAILABLE:
            cprint("  ⚠  matplotlib non trovato – --plots ignorato.", YELLOW)
            args.plots = False
        else:
            plot_dir = Path(args.plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            cprint(f"  Plot salvati in: {plot_dir.resolve()}\n")

    # ── Analisi ──────────────────────────────────────────────────────────────
    report_rows   = []
    all_test_keys = set()

    for feature in selected:
        orig_s = df_orig[feature]
        aug_s  = df_aug[feature]
        ftype  = detect_feature_type(orig_s)
        desc   = descriptive_stats(orig_s, aug_s)

        if ftype == "numeric":
            orig_arr = orig_s.dropna().values
            aug_arr  = aug_s.dropna().values
            results  = test_numeric(orig_arr, aug_arr, args.alpha)
            fit_res  = fit_distribution(orig_arr, top_n=args.top_fits)
            if not fit_res:
                disc    = fit_discrete(orig_arr)
                fit_res = [disc] if disc else []
        else:
            results = test_categorical(orig_s, aug_s, args.alpha)
            fit_res = []

        all_test_keys.update(results.keys())
        print_feature_report(feature, ftype, desc, results, fit_res, args.alpha, orig_s)

        if args.plots:
            plot_feature(feature, orig_s, aug_s, ftype,
                         results, fit_res, args.alpha, plot_dir)

        report_rows.append(
            build_report_row(feature, ftype, desc, results, fit_res,
                             orig_name=Path(args.original).name,
                             aug_name=Path(args.augmented).name,
                             orig_series=orig_s)
        )

    # ── Riepilogo ────────────────────────────────────────────────────────────
    print_summary(report_rows, args.alpha, list(all_test_keys))

    # ── CSV ──────────────────────────────────────────────────────────────────
    pd.DataFrame(report_rows).to_csv(csv_path, index=False)
    cprint(f"  Report CSV salvato in : {csv_path.resolve()}", CYAN)
    cprint(f"  Log salvato in        : {log_path.resolve()}", CYAN)
    print()

    # Ripristina stdout e chiudi il log
    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    main()