"""
──────────────────────
Reads metriche_per_dataset_final.csv and, for every dataset and every metric,
shows whether each SYRFD threshold (phi=2/4/8) is BETTER or WORSE than every
other method, printing a compact ranked/coloured console report.

Run:
    python analyse_per_dataset.py [path/to/metriche_per_dataset_final.csv]
"""

import sys
import math
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH = (sys.argv[1] if len(sys.argv) > 1
            else "1779894621870_metriche_per_dataset_final.csv")

# Metrics to analyse and their optimality direction
#   'low'     → smaller is better
#   'high'    → larger is better
#   'approx1' → closer to 1 is better
METRICS = {
    "Silhouette":                   "low",
    "Davies-Bouldin":               "high",
    "KL_mean":                      "low",
    "JS_mean":                      "low",
    "Q_multi_attribute_similarity": "high",
    "KS_mean":                      "low",
    "pMSE":                         "low",
    "MMD":                          "low",
    "Precision":                    "high",
    "Recall":                       "high",
    "Density":                      "high",
    "Coverage":                     "high",
    "mDCR":                         "approx1",
    "Coverage_div":                 "high",
    "Moment_skew_diff":             "low",
}

METRIC_LABEL = {
    "Silhouette":                   "Silhouette ↓",
    "Davies-Bouldin":               "DBI ↑",
    "KL_mean":                      "KL ↓",
    "JS_mean":                      "JS ↓",
    "Q_multi_attribute_similarity": "Q-multi ↑",
    "KS_mean":                      "KS ↓",
    "pMSE":                         "pMSE ↓",
    "MMD":                          "MMD ↓",
    "Precision":                    "Prec. ↑",
    "Recall":                       "Rec. ↑",
    "Density":                      "Dens. ↑",
    "Coverage":                     "Cov. ↑",
    "mDCR":                         "mDCR ≈1",
    "Coverage_div":                 "Cov.div ↑",
    "Moment_skew_diff":             "ΔSkew ↓",
}

SYRFD_KEYS = ["SYRFD_thr2", "SYRFD_thr4", "SYRFD_thr8"]
SYRFD_DISP = {"SYRFD_thr2": "φ=2", "SYRFD_thr4": "φ=4", "SYRFD_thr8": "φ=8"}

OVERFLOW = 1e10   # values above this are treated as NaN

# ANSI colour codes (fall back to plain text on non-colour terminals)
try:
    import os
    _colours = os.get_terminal_size()   # will raise if not a TTY
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    DIM    = "\033[2m"
except Exception:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = DIM = ""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_val(s):
    """Parse '0.207 +/- 0.000' → float; overflow/NaN → np.nan."""
    try:
        v = float(str(s).split("+/-")[0].strip())
        return v if abs(v) < OVERFLOW else np.nan
    except Exception:
        return np.nan


def score(value, direction):
    """Transform value to a 'score' where HIGHER is always better."""
    if np.isnan(value):
        return np.nan
    if direction == "low":
        return -value
    if direction == "high":
        return value
    # approx1
    return -abs(value - 1.0)


def rank_label(syrfd_val, syrfd_dir, all_vals):
    """
    Given the SYRFD value and all method values for one metric/dataset,
    return (rank_1based, total_valid, pct_beaten) where:
      rank=1  → best of all methods
      rank=N  → worst of all methods
    """
    s_score = score(syrfd_val, syrfd_dir)
    if np.isnan(s_score):
        return None, None, None
    all_scores = [score(v, syrfd_dir) for v in all_vals if not np.isnan(v)]
    if not all_scores:
        return None, None, None
    n_total  = len(all_scores)
    n_beaten = sum(1 for x in all_scores if x < s_score)
    n_equal  = sum(1 for x in all_scores if abs(x - s_score) < 1e-9)
    rank     = n_total - n_beaten - n_equal + 1   # 1 = best
    return rank, n_total, n_beaten / n_total * 100


def verdict_str(rank, n_total, pct_beaten, syrfd_val, best_val, worst_val, direction):
    """One-line human-readable verdict."""
    if rank is None:
        return f"{DIM}  N/A (no data){RESET}"

    fmtv = lambda x: f"{x:.4f}" if not np.isnan(x) else "nan"

    if rank == 1:
        colour = GREEN
        tag    = "★ BEST"
    elif rank <= max(1, math.ceil(n_total * 0.25)):
        colour = GREEN
        tag    = f"TOP {rank}/{n_total}"
    elif rank >= n_total:
        colour = RED
        tag    = "✗ WORST"
    elif rank >= math.floor(n_total * 0.75):
        colour = RED
        tag    = f"BOT {rank}/{n_total}"
    else:
        colour = YELLOW
        tag    = f"MID {rank}/{n_total}"

    val_str = fmtv(syrfd_val)
    gap_best  = abs(syrfd_val - best_val)  if not np.isnan(best_val)  else float("nan")
    gap_worst = abs(syrfd_val - worst_val) if not np.isnan(worst_val) else float("nan")

    return (f"{colour}{BOLD}{tag}{RESET}{colour}"
            f"  val={val_str}"
            f"  beats {pct_beaten:.0f}% of methods"
            f"  [best={fmtv(best_val)}, worst={fmtv(worst_val)}]"
            f"{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY ACCUMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class SummaryAccum:
    """Tracks wins/losses across all datasets and metrics."""
    def __init__(self):
        # ranks[syrfd_key][metric] = list of (rank, n_total)
        self.data = {k: {m: [] for m in METRICS} for k in SYRFD_KEYS}

    def add(self, syrfd_key, metric, rank, n_total):
        if rank is not None:
            self.data[syrfd_key][metric].append((rank, n_total))

    def print_summary(self):
        W = 110
        print("\n" + "═" * W)
        print(f"{BOLD}{'GLOBAL SUMMARY':^{W}}{RESET}")
        print("═" * W)
        print(f"\n  For each SYRFD threshold, % of (dataset×metric) combinations where it ranks:")
        print(f"  {GREEN}TOP 25%{RESET}  |  {YELLOW}MID 50%{RESET}  |  {RED}BOT 25%{RESET}  (rank 1 = best)\n")

        for sk in SYRFD_KEYS:
            top, mid, bot, na = 0, 0, 0, 0
            best_count = 0
            worst_count = 0
            for m, records in self.data[sk].items():
                for rank, n_total in records:
                    thresh_top = max(1, math.ceil(n_total * 0.25))
                    thresh_bot = math.floor(n_total * 0.75)
                    if rank == 1:
                        top += 1; best_count += 1
                    elif rank <= thresh_top:
                        top += 1
                    elif rank >= n_total:
                        bot += 1; worst_count += 1
                    elif rank >= thresh_bot:
                        bot += 1
                    else:
                        mid += 1
                if not records:
                    na += 1

            total = top + mid + bot
            if total == 0:
                continue

            bar_len  = 40
            t_bar = int(top / total * bar_len)
            m_bar = int(mid / total * bar_len)
            b_bar = bar_len - t_bar - m_bar

            bar = (f"{GREEN}{'█' * t_bar}{RESET}"
                   f"{YELLOW}{'█' * m_bar}{RESET}"
                   f"{RED}{'█' * b_bar}{RESET}")

            disp = SYRFD_DISP[sk]
            print(f"  SyRFD {disp:<4}  [{bar}]  "
                  f"{GREEN}top:{top:3d}({top/total*100:4.0f}%){RESET}  "
                  f"{YELLOW}mid:{mid:3d}({mid/total*100:4.0f}%){RESET}  "
                  f"{RED}bot:{bot:3d}({bot/total*100:4.0f}%){RESET}  "
                  f"(★best:{best_count}  ✗worst:{worst_count})")

        print()

        # Per-metric breakdown
        print(f"  {BOLD}Per-metric summary (% in TOP 25% across all datasets):{RESET}")
        header = f"  {'Metric':<18}" + "".join(f"  {SYRFD_DISP[k]:<8}" for k in SYRFD_KEYS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for m in METRICS:
            row = f"  {METRIC_LABEL[m]:<18}"
            for sk in SYRFD_KEYS:
                records = self.data[sk][m]
                if not records:
                    row += f"  {'N/A':<8}"
                    continue
                n_top = sum(1 for rank, nt in records
                            if rank <= max(1, math.ceil(nt * 0.25)))
                pct = n_top / len(records) * 100
                colour = GREEN if pct >= 50 else (YELLOW if pct >= 25 else RED)
                row += f"  {colour}{pct:5.0f}%{RESET}  "
            print(row)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH, sep=";")
    for col in METRICS:
        if col in df.columns:
            df[col] = df[col].apply(parse_val)

    datasets = sorted(df["dataset"].unique())
    accum    = SummaryAccum()
    W        = 110    # console width

    # ── Per dataset ──────────────────────────────────────────────────────────
    for ds in datasets:
        df_ds = df[df["dataset"] == ds].copy()

        # Build dict: method_key → {metric: value}
        method_vals = {}
        for _, row in df_ds.iterrows():
            mk = str(row["metodo"])
            method_vals[mk] = {m: row[m] if m in row.index else np.nan
                               for m in METRICS}

        # Only proceed if at least one SYRFD key has data
        syrfd_present = [k for k in SYRFD_KEYS if k in method_vals]
        if not syrfd_present:
            continue

        baselines = [k for k in method_vals if k not in SYRFD_KEYS]
        all_methods = list(method_vals.keys())

        # ── Dataset header ────────────────────────────────────────────────────
        print("\n" + "═" * W)
        print(f"{BOLD}{CYAN}  DATASET: {ds}   "
              f"({len(all_methods)} methods, {len(syrfd_present)} SYRFD variants){RESET}")
        print("═" * W)

        # ── Per metric ────────────────────────────────────────────────────────
        for metric, direction in METRICS.items():
            if metric not in df.columns:
                continue

            all_vals = [method_vals[mk][metric] for mk in all_methods]
            valid_vals = [v for v in all_vals if not np.isnan(v)]

            if not valid_vals:
                continue

            best_v  = max(valid_vals, key=lambda x: score(x, direction))
            worst_v = min(valid_vals, key=lambda x: score(x, direction))

            print(f"\n  {BOLD}{METRIC_LABEL[metric]:<14}{RESET}", end="")

            # Best baseline for comparison
            base_vals = [method_vals[mk][metric]
                         for mk in baselines
                         if not np.isnan(method_vals[mk].get(metric, np.nan))]
            best_base = (max(base_vals, key=lambda x: score(x, direction))
                         if base_vals else np.nan)

            for sk in syrfd_present:
                sv   = method_vals[sk][metric]
                rank, n_total, pct_beaten = rank_label(sv, direction, all_vals)
                accum.add(sk, metric, rank, n_total)

                s_score = score(sv, direction)
                b_score = score(best_base, direction) if not np.isnan(best_base) else np.nan

                # vs-best-baseline indicator
                if not np.isnan(s_score) and not np.isnan(b_score):
                    if s_score > b_score + 1e-9:
                        vs_base = f"{GREEN}+{RESET}"
                    elif s_score < b_score - 1e-9:
                        vs_base = f"{RED}-{RESET}"
                    else:
                        vs_base = f"{YELLOW}={RESET}"
                else:
                    vs_base = " "

                verd = verdict_str(rank, n_total, pct_beaten,
                                   sv, best_v, worst_v, direction)
                print(f"\n    {SYRFD_DISP[sk]:<5} {vs_base}base  {verd}")

        print()

    # ── Global summary ───────────────────────────────────────────────────────
    accum.print_summary()


if __name__ == "__main__":
    main()