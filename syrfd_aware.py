"""
syrfd_aware.py
==============
Distribution-Aware SyRFD: extends SyRFD_optimized_dual.SyRFD by running
statistical distribution fitting on each feature at startup, then using the
identified per-feature distributions when generating synthetic values.

Key additions over the dual baseline
--------------------------------------
1. At __init__ time, every feature's best-fit distribution is identified via
   KS-test fitting across an expanded catalog (numpy.random + scipy.stats).
2. A per-feature config dict is stored in self._dist_configs.
3. All free-random generation points in _get_attr_value,
   _get_safe_fallback_value, _get_rhs_constrained_value, and
   generate_safe_value are replaced with calls to
   self._aware_sample(attr, lo, hi), which samples from the fitted
   distribution clipped to [lo, hi].
4. Full fitting results are logged to
   ./distribution_analysis/logsyrfd/{dataset}_thr{threshold}_{timestamp}.log
"""

import os
import re
import math
import random
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import kstest

from SyRFD_optimized_dual import SyRFD

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Extended distribution catalog
# (scipy_name, readable_label, numpy/scipy generator template, param note)
# ─────────────────────────────────────────────────────────────────────────────
DISTRIBUTION_CATALOG = [
    # ── Continuous ───────────────────────────────────────────────────────────
    ("norm",        "Normale (Gaussiana)",          "numpy.random.normal(loc, scale)",               "loc=μ, scale=σ"),
    ("lognorm",     "Log-Normale",                  "numpy.random.lognormal(mean, sigma)",            "mean=log(μ), sigma=log(σ)"),
    ("expon",       "Esponenziale",                 "numpy.random.exponential(scale)",                "scale=1/λ"),
    ("gamma",       "Gamma",                        "numpy.random.gamma(shape, scale)",               "shape=α, scale=β"),
    ("beta",        "Beta",                         "numpy.random.beta(a, b)",                        "a=α, b=β  (valori in [0,1])"),
    ("uniform",     "Uniforme Continua",            "numpy.random.uniform(low, high)",                "low=a, high=b"),
    ("triang",      "Triangolare",                  "numpy.random.triangular(left, mode, right)",     "c=(mode-left)/(right-left)"),
    ("chi2",        "Chi-quadro",                   "numpy.random.chisquare(df)",                     "df=gradi di libertà"),
    ("t",           "t di Student",                 "numpy.random.standard_t(df)",                    "df=gradi di libertà"),
    ("f",           "F di Fisher",                  "numpy.random.f(dfn, dfd)",                       "dfn, dfd=gradi libertà"),
    ("weibull_min", "Weibull",                       "numpy.random.weibull(a)",                        "a=shape (c in scipy)"),
    ("pareto",      "Pareto",                       "numpy.random.pareto(a)",                         "a=shape"),
    ("laplace",     "Laplace (Double Exp.)",        "numpy.random.laplace(loc, scale)",               "loc=μ, scale=b"),
    ("logistic",    "Logistica",                    "numpy.random.logistic(loc, scale)",              "loc=μ, scale=s"),
    ("cauchy",      "Cauchy",                       "scipy.stats.cauchy.rvs(loc, scale)",             "loc=x0, scale=γ  (nessuna media)"),
    ("rayleigh",    "Rayleigh",                     "numpy.random.rayleigh(scale)",                   "scale=σ"),
    ("gumbel_r",    "Gumbel (estremi massimi)",     "numpy.random.gumbel(loc, scale)",                "loc=μ, scale=β"),
    ("invgauss",    "Wald / Gauss Inversa",         "numpy.random.wald(mean, scale)",                 "mean=μ, scale=λ"),
    ("halfnorm",    "Semi-Normale",                 "scipy.stats.halfnorm.rvs(loc, scale)",           "loc=0, scale=σ"),
    ("genextreme",  "GEV (Gumbel generalizzata)",   "scipy.stats.genextreme.rvs(c, loc, scale)",      "c=ξ (shape)"),
    ("powerlaw",    "Power Law",                    "numpy.random.power(a)",                          "a=shape (a>0)"),
    ("erlang",      "Erlang",                       "numpy.random.gamma(shape=k, scale)",             "k intero, scale=1/λ"),
    ("skewnorm",    "Normale Asimmetrica",          "scipy.stats.skewnorm.rvs(a, loc, scale)",        "a=parametro asimmetria"),
    ("exponnorm",   "Gaussiana-Esponenziale",       "scipy.stats.exponnorm.rvs(K, loc, scale)",       "K=1/(σλ)"),
    ("loggamma",    "Log-Gamma",                    "scipy.stats.loggamma.rvs(c, loc, scale)",        "c=shape"),
    # ── Discrete ─────────────────────────────────────────────────────────────
    ("poisson",     "Poisson (discreta)",           "numpy.random.poisson(lam)",                      "lam=λ  (dati interi non-neg.)"),
    ("nbinom",      "Neg. Binomiale (discreta)",    "numpy.random.negative_binomial(n, p)",           "n, p"),
    ("geom",        "Geometrica (discreta)",        "numpy.random.geometric(p)",                      "p=probabilità successo (k≥1)"),
]

DISCRETE_DISTS = {"poisson", "nbinom", "geom"}


# ─────────────────────────────────────────────────────────────────────────────
# Feature-type detection (from check_distribution.py)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_feature_type(series: pd.Series, max_categories: int = 20) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# Distribution fitting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_continuous(data: np.ndarray, top_n: int = 1) -> list:
    """Fit all continuous catalog distributions; return top_n by KS p-value."""
    results = []
    data = data[np.isfinite(data)]
    if len(data) < 10:
        return []
    for dist_name, label, generator, param_note in DISTRIBUTION_CATALOG:
        if dist_name in DISCRETE_DISTS:
            continue
        try:
            dist_obj = getattr(stats, dist_name)
            params = dist_obj.fit(data)
            ks_stat, ks_p = kstest(data, dist_name, args=params)
            results.append({
                "dist_name":  dist_name,
                "label":      label,
                "params":     params,
                "ks_stat":    ks_stat,
                "ks_p":       ks_p,
                "generator":  generator,
                "param_note": param_note,
                "ftype":      "continuous",
            })
        except Exception:
            continue
    results.sort(key=lambda x: x["ks_p"], reverse=True)
    return results[:top_n]


def _fit_discrete(data: np.ndarray) -> dict | None:
    """Attempt Poisson, Negative-Binomial, and Geometric fitting for integer ≥ 0 data."""
    data = data[np.isfinite(data)].astype(int)
    if data.min() < 0 or len(data) < 10:
        return None

    best      = None
    best_p    = -1.0
    max_val   = int(data.max())

    # ── Poisson ──────────────────────────────────────────────────────────────
    lam = float(data.mean())
    if lam > 0:
        expected_p = stats.poisson.pmf(np.arange(max_val + 1), lam)
        observed_c = np.bincount(data, minlength=max_val + 1)
        mask = expected_p * len(data) >= 5
        if mask.sum() >= 2:
            try:
                _, p_val = stats.chisquare(
                    observed_c[mask], f_exp=expected_p[mask] * len(data)
                )
                if p_val > best_p:
                    best_p = p_val
                    best = {
                        "dist_name":  "poisson",
                        "label":      "Poisson",
                        "params":     (lam,),
                        "ks_p":       p_val,
                        "generator":  f"numpy.random.poisson(lam={lam:.4f})",
                        "param_note": f"lam={lam:.4f}",
                        "ftype":      "discrete",
                    }
            except Exception:
                pass

    # ── Negative Binomial ────────────────────────────────────────────────────
    mu  = float(data.mean())
    var = float(data.var())
    if var > mu > 0:
        p_nb = mu / var
        n_nb = mu * p_nb / (1.0 - p_nb)
        if n_nb > 0 and 0.0 < p_nb < 1.0:
            try:
                expected_p = stats.nbinom.pmf(np.arange(max_val + 1), n_nb, p_nb)
                observed_c = np.bincount(data, minlength=max_val + 1)
                mask = expected_p * len(data) >= 5
                if mask.sum() >= 2:
                    _, p_val = stats.chisquare(
                        observed_c[mask], f_exp=expected_p[mask] * len(data)
                    )
                    if p_val > best_p:
                        best_p = p_val
                        best = {
                            "dist_name":  "nbinom",
                            "label":      "Neg. Binomiale",
                            "params":     (n_nb, p_nb),
                            "ks_p":       p_val,
                            "generator":  f"numpy.random.negative_binomial(n={n_nb:.4f}, p={p_nb:.4f})",
                            "param_note": f"n={n_nb:.4f}, p={p_nb:.4f}",
                            "ftype":      "discrete",
                        }
            except Exception:
                pass

    # ── Geometric ────────────────────────────────────────────────────────────
    if mu > 1:
        p_geom = 1.0 / mu
        if 0.0 < p_geom < 1.0:
            try:
                # scipy geom: k ≥ 1; shift data by +1 for fitting
                data_shifted = data + 1
                max_s = int(data_shifted.max())
                expected_p = stats.geom.pmf(np.arange(1, max_s + 1), p_geom)
                observed_c = np.bincount(data_shifted, minlength=max_s + 1)[1:]
                mask = expected_p[:len(observed_c)] * len(data) >= 5
                if mask.sum() >= 2:
                    _, p_val = stats.chisquare(
                        observed_c[mask],
                        f_exp=expected_p[:len(observed_c)][mask] * len(data)
                    )
                    if p_val > best_p:
                        best_p = p_val
                        best = {
                            "dist_name":  "geom",
                            "label":      "Geometrica",
                            "params":     (p_geom, -1, 1),  # loc=-1 shifts k≥1 → k≥0
                            "ks_p":       p_val,
                            "generator":  f"numpy.random.geometric(p={p_geom:.4f}) - 1",
                            "param_note": f"p={p_geom:.4f}  (sottrarre 1 per k≥0)",
                            "ftype":      "discrete",
                        }
            except Exception:
                pass

    return best


def _categorical_config(series: pd.Series) -> dict:
    """Build generator config for binary/categorical features."""
    counts     = series.dropna().value_counts()
    total      = counts.sum()
    categories = counts.index.tolist()
    probs      = (counts / total).tolist()
    return {
        "dist_name":  "categorical",
        "label":      f"Categorica ({len(categories)} classi)",
        "params":     None,
        "ks_p":       float("nan"),
        "generator":  f"numpy.random.choice({categories}, p={[round(p,4) for p in probs]})",
        "param_note": f"categories={categories}, probs={[round(p,4) for p in probs]}",
        "ftype":      "categorical",
        "categories": [float(c) if isinstance(c, (int, np.integer, float, np.floating)) else c
                       for c in categories],
        "probs":      probs,
    }


def _fallback_uniform_config(data_min: float, data_max: float) -> dict:
    return {
        "dist_name":  "uniform",
        "label":      "Uniforme (fallback)",
        "params":     (data_min, data_max - data_min),
        "ks_p":       float("nan"),
        "generator":  f"numpy.random.uniform({data_min:.4f}, {data_max:.4f})",
        "param_note": "fallback",
        "ftype":      "fallback",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SyRFDAware
# ─────────────────────────────────────────────────────────────────────────────

class SyRFDAware(SyRFD):
    """
    Distribution-Aware SyRFD.

    Extends SyRFD (dual strategy) by fitting the statistically best-matching
    distribution to every feature before augmentation starts, then using those
    fitted distributions as the random generators throughout.
    """

    def __init__(self, imbalance_dataset_path, rfd_file_path, oversampling,
                 threshold=4, max_iter=100, selected_rfds=None,
                 dist_alpha: float = 0.05):
        # Must be set before super().__init__() in case any override is reached
        self._dist_alpha   = dist_alpha
        self._dist_configs: dict = {}   # populated by _run_distribution_analysis

        super().__init__(
            imbalance_dataset_path=imbalance_dataset_path,
            rfd_file_path=rfd_file_path,
            oversampling=oversampling,
            threshold=threshold,
            max_iter=max_iter,
            selected_rfds=selected_rfds,
        )

        self._run_distribution_analysis()

    # ── Distribution fitting ─────────────────────────────────────────────────

    def _run_distribution_analysis(self):
        """Fit a distribution to every feature; populate self._dist_configs."""
        print("\n=== SyRFDAware: Distribution Analysis ===")

        for attr in self.all_attrs:
            series = self.imbalance_df_min[attr].dropna()
            data   = series.values.astype(float)
            ftype  = _detect_feature_type(series)

            if ftype in ("binary", "categorical"):
                config = _categorical_config(series)

            elif attr in self.integer_attrs:
                disc = _fit_discrete(data)
                if disc is not None:
                    config = disc
                else:
                    fits = _fit_continuous(data, top_n=1)
                    config = fits[0] if fits else _fallback_uniform_config(data.min(), data.max())

            else:
                fits = _fit_continuous(data, top_n=1)
                config = fits[0] if fits else _fallback_uniform_config(data.min(), data.max())

            self._dist_configs[attr] = config

            ks_str = (f"{config['ks_p']:.4f}"
                      if not math.isnan(config.get("ks_p", float("nan"))) else "N/A")
            print(f"  {attr:<18}  {config['label']:<30}  KS p={ks_str}")

        self._save_distribution_log()
        print("=== Distribution Analysis complete ===\n")

    def _save_distribution_log(self):
        log_dir = os.path.join("distribution_analysis", "logsyrfd")
        os.makedirs(log_dir, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"{self.base}_thr{self.threshold}_{ts}.log"
        log_path = os.path.join(log_dir, log_name)

        sep  = "=" * 60
        dash = "-" * 60

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"{sep}\n")
            f.write("  SyRFDAware  Distribution Configuration Log\n")
            f.write(f"{sep}\n")
            f.write(f"  Dataset   : {self.base}\n")
            f.write(f"  Threshold : {self.threshold}\n")
            f.write(f"  Alpha     : {self._dist_alpha}\n")
            f.write(f"  Timestamp : {ts}\n")
            f.write(f"  Features  : {len(self._dist_configs)}\n")
            f.write(f"{sep}\n\n")

            for attr, cfg in self._dist_configs.items():
                ks_p = cfg.get("ks_p", float("nan"))
                ks_str = f"{ks_p:.6f}" if not math.isnan(ks_p) else "N/A"
                params = cfg.get("params")
                f.write(f"Feature   : {attr}\n")
                f.write(f"  Type    : {cfg.get('ftype', '—')}\n")
                f.write(f"  Dist    : {cfg.get('label', '—')}\n")
                f.write(f"  KS p    : {ks_str}\n")
                if params is not None:
                    f.write(f"  Params  : {params}\n")
                f.write(f"  Note    : {cfg.get('param_note', '—')}\n")
                f.write(f"  Call    : {cfg.get('generator', '—')}\n")
                if cfg.get("ftype") in ("binary", "categorical"):
                    cats  = cfg.get("categories", [])
                    probs = [round(p, 4) for p in cfg.get("probs", [])]
                    f.write(f"  Cats    : {cats}\n")
                    f.write(f"  Probs   : {probs}\n")
                f.write("\n")

            f.write(f"{sep}\n")
            f.write("  SUMMARY TABLE\n")
            f.write(f"{dash}\n")
            f.write(f"  {'Feature':<18}  {'Distribution':<28}  {'KS p-value':>12}\n")
            f.write(f"{dash}\n")
            for attr, cfg in self._dist_configs.items():
                ks_p = cfg.get("ks_p", float("nan"))
                ks_str = f"{ks_p:.4f}" if not math.isnan(ks_p) else "N/A"
                f.write(f"  {attr:<18}  {cfg.get('label','—'):<28}  {ks_str:>12}\n")
            f.write(f"{sep}\n")

        print(f"  Log saved: {log_path}")

    # ── Low-level sampling ───────────────────────────────────────────────────

    def _rejection_sample(self, dist_name: str, params: tuple,
                          lo: float, hi: float) -> float:
        """
        Sample from scipy distribution within [lo, hi] via rejection sampling
        (up to 30 tries), then clip as last resort.
        """
        dist_obj = getattr(stats, dist_name, None)
        if dist_obj is None:
            return random.uniform(lo, hi)
        for _ in range(30):
            try:
                v = float(dist_obj.rvs(*params))
                if lo <= v <= hi:
                    return v
            except Exception:
                break
        try:
            return float(np.clip(float(dist_obj.rvs(*params)), lo, hi))
        except Exception:
            return random.uniform(lo, hi)

    def _aware_sample(self, attr: str, lo: float, hi: float) -> float:
        """
        Sample a value for attr in [lo, hi] using the fitted distribution.
        Returns an int for integer attributes, float otherwise.
        Falls back to uniform if distribution cannot produce an in-range value.
        """
        lo, hi = float(lo), float(hi)
        if lo > hi:
            lo, hi = hi, lo
        if math.isclose(lo, hi, rel_tol=1e-9):
            return int(round(lo)) if attr in self.integer_attrs else round(lo, 2)

        config = self._dist_configs.get(attr)

        if config is None or config.get("ftype") == "fallback":
            val = random.uniform(lo, hi)

        elif config["ftype"] in ("binary", "categorical"):
            categories = config["categories"]
            probs      = config["probs"]
            val = float(np.random.choice(categories, p=probs))
            val = float(np.clip(val, lo, hi))

        else:
            dist_name = config["dist_name"]
            params    = config["params"]
            val = self._rejection_sample(dist_name, params, lo, hi)

        val = float(np.clip(val, lo, hi))

        if attr in self.integer_attrs:
            lo_i = int(math.ceil(lo))
            hi_i = int(math.floor(hi))
            if lo_i > hi_i:
                lo_i, hi_i = int(round(lo)), int(round(hi))
            return int(np.clip(int(round(val)), lo_i, hi_i))

        return round(val, 2)

    def _aware_sample_float(self, attr: str, lo: float, hi: float) -> float:
        """Like _aware_sample but always returns a float (ignores integer_attrs).
        Used when use_decimal=True forces a decimal value."""
        lo, hi = float(lo), float(hi)
        if lo > hi:
            lo, hi = hi, lo
        if math.isclose(lo, hi, rel_tol=1e-9):
            return round(lo, 2)

        config = self._dist_configs.get(attr)

        if config is None or config.get("ftype") == "fallback":
            val = random.uniform(lo, hi)
        elif config["ftype"] in ("binary", "categorical"):
            val = random.uniform(lo, hi)
        else:
            dist_name = config["dist_name"]
            params    = config["params"]
            val = self._rejection_sample(dist_name, params, lo, hi)

        return round(float(np.clip(val, lo, hi)), 2)

    # ── Overridden generation methods ────────────────────────────────────────

    def _get_safe_fallback_value(self, attr, use_decimal=False):
        """
        Distribution-aware override.

        rhs_only / free  → sample from fitted distribution inside data range.
        lhs_only / both  → unchanged from dual (data-bounded far value that
                           prevents accidental LHS similarity with existing tuples).
        """
        role     = self._role_of(attr)
        stats_   = self._attr_stats.get(attr, {})
        data_min = stats_.get("min", 0.0)
        data_max = stats_.get("max", 1.0)

        if role == "rhs_only":
            val = self._aware_sample(attr, data_min, data_max)
            print(f"  {attr} [RHS-only aware fallback]: {val}")
            return val

        if role == "free":
            val = self._aware_sample(attr, data_min, data_max)
            print(f"  {attr} [free aware fallback]: {val}")
            return val

        # lhs_only or both: keep dual far-value logic unchanged
        jitter    = random.uniform(0, self.threshold)
        upper     = data_max + self.threshold + 1 + jitter
        lower     = data_min - self.threshold - 1 - jitter
        safe_base = random.choice([upper, lower])
        print(f"  {attr} [{role} fallback]: data-bounded far value -> {safe_base:.3f}")
        if attr in self.integer_attrs:
            return int(round(safe_base))
        if use_decimal:
            return round(safe_base + round(random.random(), 2), 2)
        return round(safe_base, 2)

    def _get_rhs_constrained_value(self, rhs_attr, generated_values, use_decimal=False):
        """
        Distribution-aware override.

        Free sampling (no LHS constraint found) and feasible-window sampling
        both use the fitted distribution for rhs_attr, clipped to the window.
        """
        stats_   = self._attr_stats.get(rhs_attr, {})
        data_min = stats_.get("min", 0.0)
        data_max = stats_.get("max", 1.0)

        lo = data_min
        hi = data_max
        constraint_applied = False

        relevant_rfds = [(lhs, rhs) for lhs, rhs in self.dependencies if rhs == rhs_attr]

        for lhs_list, _ in relevant_rfds:
            if not all(a in generated_values for a in lhs_list):
                continue
            lhs_gen = np.array([float(generated_values[a]) for a in lhs_list])
            for _, row in self.imbalance_df_min.iterrows():
                lhs_existing = np.array([float(row[a]) for a in lhs_list])
                if np.all(np.abs(lhs_gen - lhs_existing) <= self.threshold):
                    rhs_val = float(row[rhs_attr])
                    lo = max(lo, rhs_val - self.threshold)
                    hi = min(hi, rhs_val + self.threshold)
                    constraint_applied = True

        if not constraint_applied:
            val = self._aware_sample(rhs_attr, data_min, data_max)
            print(f"  {rhs_attr} [RHS constrained, no similar LHS, aware]: {val}")
            return val

        if lo > hi:
            print(f"  {rhs_attr} [RHS constrained]: infeasible [{lo:.3f},{hi:.3f}], using fallback")
            return self._get_safe_fallback_value(rhs_attr, use_decimal)

        val = self._aware_sample(rhs_attr, lo, hi)
        print(f"  {rhs_attr} [RHS constrained, aware]: feasible=[{lo:.3f},{hi:.3f}] -> {val}")
        return val

    def _get_attr_value(self, attr, i1, i2, use_decimal=False,
                        generated_values=None, missing_attrs=None):
        """
        Distribution-aware override.

        All random generation inside a computed [lo, hi] window uses
        _aware_sample (or _aware_sample_float when use_decimal=True).
        The diff=0 shortcut, diff=1 euclidean-nearest heuristic, and the
        LHS-far dual-fallback remain unchanged.
        """
        role = self._role_of(attr)

        if missing_attrs and attr in missing_attrs:
            print(f"  {attr} [missing, role={role}]: dual routing")
            return self._dual_fallback(attr, role, generated_values, use_decimal)

        row = self.attrs_df[
            (self.attrs_df["attribute"] == attr) &
            (self.attrs_df["idx1"] == i1) &
            (self.attrs_df["idx2"] == i2)
        ]

        if not row.empty:
            if attr in self.no_dependency_attrs:
                print(f"{attr} not in any RFD, generate aware(min_val, min_val+diff)")
                r        = row.iloc[0]
                val1, val2 = r["val1"], r["val2"]
                min_val  = min(val1, val2)
                diff_val = float(r["diff"])
                random_val = self._aware_sample(attr, min_val, min_val + diff_val)
                print(f"  {attr}: No dependency, min={min_val}, diff={diff_val}, -> {random_val}")
                return random_val

            r = row.iloc[0]
            val1, val2 = r["val1"], r["val2"]
            diff = r["diff"]
            print(f"  {attr}: val1={val1}, val2={val2}, diff={diff}, threshold={self.threshold}")

            if diff <= self.threshold:
                min_val = min(val1, val2)
                max_val = max(val1, val2)

                if diff == 0:
                    print(f"Identical values, using {val1}")
                    return val1

                if diff == 1:
                    # Try euclidean-nearest heuristic first (unchanged)
                    try:
                        if generated_values and len(generated_values) > 0:
                            keys = [k for k in generated_values.keys()
                                    if k != attr and k in self.all_attrs]
                            if keys:
                                vec_gen = np.array([float(generated_values[k]) for k in keys])
                                vec_i1  = np.array([float(self.imbalance_df_min.at[i1, k]) for k in keys])
                                vec_i2  = np.array([float(self.imbalance_df_min.at[i2, k]) for k in keys])
                                d1 = np.linalg.norm(vec_gen - vec_i1)
                                d2 = np.linalg.norm(vec_gen - vec_i2)
                                chosen = val1 if d1 <= d2 else val2
                                print(f"diff==1: euclidean d1={d1:.2f}, d2={d2:.2f}, choosing: {chosen}")
                                return chosen
                    except Exception as e:
                        print(f"Warning computing euclidean distance: {e}")

                    chosen = self._aware_sample(attr, int(min_val), int(max_val))
                    print(f"diff==1 aware sample: {chosen}")
                    return chosen

                else:
                    if use_decimal:
                        generated_val = self._aware_sample_float(attr, float(min_val), float(max_val))
                        print(f"Similar values (decimal aware), generating: {generated_val:.2f}")
                        return generated_val
                    else:
                        generated_val = self._aware_sample(attr, min_val, max_val)
                        print(f"Aware sample for {attr}: {generated_val}")
                        return generated_val
            else:
                print(f"  {attr} [dissimilar diff={diff}>{self.threshold}, role={role}]: dual routing")
                return self._dual_fallback(attr, role, generated_values, use_decimal)
        else:
            print(f"  {attr} [no pair data, role={role}]: dual routing")
            return self._dual_fallback(attr, role, generated_values, use_decimal)

    def generate_safe_value(self, attr, avoid_similarity=True):
        """
        Distribution-aware override.
        Samples within the pre-computed safe ranges using the fitted distribution.
        """
        if avoid_similarity:
            safe_ranges = self.get_safe_value_ranges(attr)
            if safe_ranges:
                chosen_range = random.choice(safe_ranges)
                min_val, max_val = chosen_range
                safe_val = self._aware_sample(attr, min_val, max_val)
                print(f"Generated safe value for {attr}: {safe_val} in range {chosen_range}")
                return safe_val
        return self._get_safe_fallback_value(attr, use_decimal=True)
