import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ===========================
# 1️⃣ Carica i dati
# ===========================
real_csv_path = ".\\Test/LiverCirrhosis_onevsrest_0.csv"
synthetic_csv_path = ".\\Test/LiverCirrhosis_onevsrest_0.csv"  # altrimenti lascia None

# Colonna che indica la classe
class_column = "class"  # cambialo con il nome della tua colonna

# Leggi CSV reale
df_real = pd.read_csv(real_csv_path)

# Colonne delle feature (senza la classe)
feature_columns = [c for c in df_real.columns if c != class_column]

X_real = df_real[feature_columns].values
y_real = df_real[class_column].values

if synthetic_csv_path is not None:
    df_syn = pd.read_csv(synthetic_csv_path)

    # 🔎 Controllo colonne
    missing_cols = set(feature_columns) - set(df_syn.columns)
    extra_cols = set(df_syn.columns) - set(feature_columns) - {class_column}

    if missing_cols:
        raise ValueError(f"Mancano le colonne nel sintetico: {missing_cols}")
    if extra_cols:
        print(f"⚠️ Attenzione: colonne extra nel sintetico (ignorate): {extra_cols}")

    # Allinea l'ordine delle colonne
    X_syn = df_syn[feature_columns].values
    y_syn = df_syn[class_column].values

    # Unisci reali e sintetici
    X_all = np.vstack([X_real, X_syn])
    y_all = np.hstack([y_real, y_syn])
else:
    X_all = X_real
    y_all = y_real

# ===========================
# 2️⃣ Standardizza le feature
# ===========================
scaler = StandardScaler()
X_all_std = scaler.fit_transform(X_all)

# ===========================
# 3️⃣ Silhouette Score
# ===========================
sil_score = silhouette_score(X_all_std, y_all)
print(f"Silhouette Score: {sil_score:.4f}")

# ===========================
# 4️⃣ Compactness intra-classe
# ===========================
print("\nCompactness intra-classe:")
for c in np.unique(y_all):
    Xc = X_all_std[y_all == c]
    mu_c = Xc.mean(axis=0)
    compactness = np.mean(np.linalg.norm(Xc - mu_c, axis=1)**2)
    print(f"Classe {c}: {compactness:.4f}")

# ===========================
# 5️⃣ Davies–Bouldin Index
# ===========================
dbi = davies_bouldin_score(X_all_std, y_all)
print(f"\nDavies-Bouldin Index: {dbi:.4f}")
