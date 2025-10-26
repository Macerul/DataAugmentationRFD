import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import numpy as np

# ==========================
# Carica CSV
# ==========================
csv_file = "metriche_per_dataset.csv"
df = pd.read_csv(csv_file, sep=";")

# Filtra solo SYRFD
df_syrfd = df[df["metodo"].str.startswith("SYRFD")].copy()

# Estrai solo valore medio
metrics = ["Silhouette", "Davies-Bouldin", "Compactness_class_1",
           "KL_mean", "JS_mean"]

for m in metrics:
    df_syrfd[m] = df_syrfd[m].str.split(" ± ").str[0].astype(float)

# Crea colonna soglia
df_syrfd['soglia'] = df_syrfd['metodo'].str.extract(r'thr(\d+)').astype(int)

# ==========================
# Definisci confronti
# ==========================
comparisons = [(2, 4), (2, 8), (4, 8)]
comparison_labels = ["2 vs 4", "2 vs 8", "4 vs 8"]

# ==========================
# Calcola p-value Wilcoxon
# ==========================
p_wilcoxon = pd.DataFrame(index=metrics, columns=comparison_labels)

for metric in metrics:
    for (a, b), label in zip(comparisons, comparison_labels):
        vals_a = df_syrfd[df_syrfd['soglia'] == a][metric]
        vals_b = df_syrfd[df_syrfd['soglia'] == b][metric]

        try:
            w_stat, w_p = wilcoxon(vals_a, vals_b)
        except ValueError:
            w_p = np.nan
        p_wilcoxon.loc[metric, label] = w_p

p_wilcoxon = p_wilcoxon.astype(float)
print("\n=== WILCOXON SIGNED-RANK TEST ===")
for metric, res in p_wilcoxon.iterrows():
    print(f"\n{metric}:")
    for comp in comparison_labels:
        print(f"{comp} -> p={res[comp]:.3f}")
# ==========================
# Rinomina metriche per grafico
# ==========================
metric_labels = {
    "Silhouette": "Silhouette",
    "Davies-Bouldin": "DBI",
    "Compactness_class_1": "Compactness",
    "KL_mean": "KL",
    "JS_mean": "JS"
}
p_wilcoxon.rename(index=metric_labels, inplace=True)

# ==========================
# Plot heatmap Wilcoxon
# ==========================
plt.figure(figsize=(8, 5))
sns.heatmap(
    p_wilcoxon, annot=True, fmt=".3f",
    cmap="Blues",  # scala solo blu
    cbar_kws={'label': 'p-value'},
    vmin=0, vmax=0.05, linewidths=0.5, linecolor='white'
)
plt.title("Wilcoxon Signed-Rank Test (SYRFD Threshold Comparison)", fontsize=12, pad=15,fontweight="bold")
#plt.ylabel("Metrics", fontsize=10,fontweight="bold")
plt.xlabel("Comparison between different thresholds (φ)", fontsize=10,fontweight="bold")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()

# Salva il grafico in alta qualità
#plt.savefig("wilcoxon_heatmap_SYRFD.pdf", bbox_inches="tight")
plt.show()

print("✅ Heatmap salvata come 'wilcoxon_heatmap_SYRFD.png'")
