"""
replace_outliers.py
-------------------
Per ogni threshold in {2, 4, 8}, entra nella cartella:
    classification_results_SYRFD_thr{thr}/new_tuples/
legge tutti i file CSV/Parquet presenti, individua i valori
"enormi" (|valore| > THRESHOLD_VALUE) e li sostituisce con:
    - Strategia 1: NaN  (null)
    - Strategia 2: media della colonna  (calcolata sui valori non-outlier)
    - Strategia 3: mediana della colonna (calcolata sui valori non-outlier)

L'utente sceglie la strategia prima che inizi l'elaborazione.
I file modificati vengono salvati in una sottocartella "cleaned/".
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Configurazione ──────────────────────────────────────────────────────────
THRESHOLDS      = [2, 4, 8]          # valori di thr
OUTLIER_LIMIT   = 100_000            # |valore| > questo → outlier
SUPPORTED_EXTS  = {".csv", ".parquet", ".tsv"}
# ────────────────────────────────────────────────────────────────────────────


def scegli_strategia() -> str:
    """Chiede all'utente quale strategia usare e restituisce 'null'|'mean'|'median'."""
    print("\n" + "=" * 60)
    print("  SOSTITUZIONE VALORI OUTLIER  (|valore| > {:,})".format(OUTLIER_LIMIT))
    print("=" * 60)
    print("\nScegli la strategia di sostituzione:\n")
    print("  [1]  NULL    — sostituisce con NaN (valore mancante)")
    print("  [2]  MEDIA   — sostituisce con la media della colonna")
    print("                  (calcolata escludendo gli outlier)")
    print("  [3]  MEDIANA — sostituisce con la mediana della colonna")
    print("                  (calcolata escludendo gli outlier)")
    print()

    mapping = {"1": "null", "2": "mean", "3": "median"}
    while True:
        scelta = input("Inserisci 1, 2 o 3 e premi Invio: ").strip()
        if scelta in mapping:
            strategia = mapping[scelta]
            print(f"\n✔  Strategia scelta: {strategia.upper()}\n")
            return strategia
        print("  ⚠  Scelta non valida. Riprova.")


def carica_file(path: str) -> pd.DataFrame | None:
    """Carica un file CSV, TSV o Parquet in un DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        elif ext == ".parquet":
            return pd.read_parquet(path)
    except Exception as e:
        print(f"    ⚠  Impossibile leggere {path}: {e}")
    return None


def salva_file(df: pd.DataFrame, path_originale: str, cartella_output: str) -> None:
    """Salva il DataFrame nella cartella cleaned/ con lo stesso formato del file originale."""
    os.makedirs(cartella_output, exist_ok=True)
    nome_file = os.path.basename(path_originale)
    path_out  = os.path.join(cartella_output, nome_file)
    ext       = os.path.splitext(nome_file)[1].lower()

    try:
        if ext == ".csv":
            df.to_csv(path_out, index=False)
        elif ext == ".tsv":
            df.to_csv(path_out, sep="\t", index=False)
        elif ext == ".parquet":
            df.to_parquet(path_out, index=False)
        print(f"    ✔  Salvato → {path_out}")
    except Exception as e:
        print(f"    ⚠  Errore nel salvataggio di {path_out}: {e}")


def sostituisci_outlier(df: pd.DataFrame, strategia: str, limite: float) -> tuple[pd.DataFrame, dict]:
    """
    Individua i valori |v| > limite nelle colonne numeriche e li sostituisce
    secondo la strategia scelta.
    Restituisce il DataFrame modificato e un report con i conteggi per colonna.
    """
    df      = df.copy()
    report  = {}

    col_numeriche = df.select_dtypes(include=[np.number]).columns

    for col in col_numeriche:
        maschera_outlier = df[col].abs() > limite
        n_outlier        = maschera_outlier.sum()

        if n_outlier == 0:
            continue

        report[col] = n_outlier

        if strategia == "null":
            df.loc[maschera_outlier, col] = np.nan

        elif strategia == "mean":
            valori_validi    = df.loc[~maschera_outlier, col]
            valore_sostituto = valori_validi.mean()
            df.loc[maschera_outlier, col] = valore_sostituto

        elif strategia == "median":
            valori_validi    = df.loc[~maschera_outlier, col]
            valore_sostituto = valori_validi.median()
            df.loc[maschera_outlier, col] = valore_sostituto

    return df, report


def processa_cartella(thr: int, strategia: str) -> None:
    """Elabora tutti i dataset nella cartella new_tuples per una data threshold."""
    base_dir    = f"classification_results_SYRFD_thr{thr}"
    input_dir   = os.path.join(base_dir, "new_tuples")
    output_dir  = os.path.join(input_dir, "cleaned")

    print(f"\n{'─'*60}")
    print(f"  thr = {thr}  │  {input_dir}")
    print(f"{'─'*60}")

    if not os.path.isdir(input_dir):
        print(f"  ⚠  Cartella non trovata, salto: {input_dir}")
        return

    file_trovati = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]

    if not file_trovati:
        print(f"  ⚠  Nessun file supportato trovato in {input_dir}")
        return

    totale_outlier_globale = 0

    for nome_file in sorted(file_trovati):
        path_file = os.path.join(input_dir, nome_file)
        print(f"\n  📄 {nome_file}")

        df = carica_file(path_file)
        if df is None:
            continue

        print(f"     Righe: {len(df):,}  │  Colonne: {len(df.columns)}")

        df_pulito, report = sostituisci_outlier(df, strategia, OUTLIER_LIMIT)

        if report:
            for col, n in report.items():
                print(f"     • {col}: {n:,} outlier sostituiti")
            totale_outlier_globale += sum(report.values())
        else:
            print(f"     ✔  Nessun outlier trovato")

        salva_file(df_pulito, path_file, output_dir)

    print(f"\n  ✅ thr={thr}: outlier totali sostituiti → {totale_outlier_globale:,}")


def main() -> None:
    strategia = scegli_strategia()

    print(f"Elaborazione in corso con strategia: {strategia.upper()}")
    print(f"Limite outlier: |valore| > {OUTLIER_LIMIT:,}\n")

    for thr in THRESHOLDS:
        processa_cartella(thr, strategia)

    print(f"\n{'='*60}")
    print("  ELABORAZIONE COMPLETATA")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()