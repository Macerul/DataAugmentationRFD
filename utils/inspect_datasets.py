import os
import sys
import pandas as pd
from collections import defaultdict

# Mapping: suffisso file -> nome leggibile
REQUIRED_VERSIONS = {
    "schema_constraints": "Explicit Schema Constraints",
    "distribution_guidance": "Distribution Guidance",
    "cot_hierarchical": "CoT Gerarchica (Multi-livello) [3]",
    "cot_self_consistency": "CoT con Verifica (Self-Consistency) [4]",
    "cot_react": "CoT con ReAct Pattern [5]",
    "cot_ensemble": "CoT Ensemble (Multi-Perspective) [6]",
    "cot_decision_tree": "CoT con Albero Decisionale [7]",
    "cot_confidence": "CoT con Calcolo della Confidenza [8]",
}


def extract_dataset_and_version(filename: str):
    """
    Esempio:
        yeast3_cot_confidence.csv
    -> dataset = yeast3
    -> version = cot_confidence
    """

    name = filename.replace(".csv", "")

    for version_key in REQUIRED_VERSIONS.keys():
        suffix = f"_{version_key}"
        if name.endswith(suffix):
            dataset_name = name[:-len(suffix)]
            return dataset_name, version_key

    return None, None


def inspect_directory(directory: str):

    csv_files = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])

    if not csv_files:
        print(f"Nessun file CSV trovato in: {directory}")
        return

    print(f"\n{'=' * 60}")
    print(f"Directory: {directory}")
    print(f"File CSV trovati: {len(csv_files)}")
    print(f"{'=' * 60}\n")

    # dataset -> set(versioni trovate)
    dataset_versions = defaultdict(set)

    # Lista file con classi sbilanciate
    unbalanced_files = []

    for fname in csv_files:

        fpath = os.path.join(directory, fname)

        # ---------------------------
        # Controllo versioni dataset
        # ---------------------------
        dataset_name, version_key = extract_dataset_and_version(fname)

        if dataset_name and version_key:
            dataset_versions[dataset_name].add(version_key)

        try:
            df = pd.read_csv(fpath)

            print(
                f"📄 {fname}",
                f" Rows: {len(df)}",
                f" Cols: {len(df.columns)}",
                end=" "
            )

            class_counts = None

            if 'class' in df.columns:
                class_counts = df['class'].value_counts().sort_index()

                print(f"   Classes  :", end=" ")

                for cls, cnt in class_counts.items():
                    print(f"     [{cls}] → {cnt} righe", end=" ")

            elif 'label' in df.columns:
                class_counts = df['label'].value_counts().sort_index()

                print(f"   Labels   :", end=" ")

                for cls, cnt in class_counts.items():
                    print(f"     [{cls}] → {cnt} righe", end=" ")

            else:
                print("   (nessuna colonna 'class' o 'label' trovata)", end=" ")

            # --------------------------------------
            # Controllo bilanciamento delle classi
            # --------------------------------------
            if class_counts is not None:

                values = list(class_counts.values)

                if len(set(values)) > 1:
                    unbalanced_files.append({
                        "file": fname,
                        "counts": dict(class_counts)
                    })

            print()

        except Exception as e:
            print(f"\n⚠️  Errore nella lettura di {fname}: {e}\n")

    # =====================================================
    # CONTROLLO VERSIONI MANCANTI
    # =====================================================

    print(f"\n{'=' * 60}")
    print("CONTROLLO VERSIONI DATASET")
    print(f"{'=' * 60}")

    missing_any = False

    for dataset_name in sorted(dataset_versions.keys()):

        found_versions = dataset_versions[dataset_name]

        missing_versions = [
            REQUIRED_VERSIONS[v]
            for v in REQUIRED_VERSIONS
            if v not in found_versions
        ]

        if missing_versions:
            missing_any = True

            print(f"\n❌ Dataset: {dataset_name}")
            print("   Versioni mancanti:")

            for mv in missing_versions:
                print(f"   - {mv}")

    if not missing_any:
        print("\n✅ Tutti i dataset contengono tutte le versioni richieste.")

    # =====================================================
    # CONTROLLO CLASSI SBILANCIATE
    # =====================================================

    print(f"\n{'=' * 60}")
    print("CONTROLLO CLASSI SBILANCIATE")
    print(f"{'=' * 60}")

    if unbalanced_files:

        for item in unbalanced_files:

            print(f"\n❌ {item['file']}")

            for cls, cnt in item["counts"].items():
                print(f"   Classe [{cls}] -> {cnt} righe")

    else:
        print("\n✅ Tutti i file hanno classi bilanciate.")


if __name__ == "__main__":

    target_dir = "augmented_datasets/gemma4_31b-cloud"
    target_dir = "augmented_datasets/gemma3_12b"

    if not os.path.isdir(target_dir):
        print(f"Errore: la directory '{target_dir}' non esiste.")
        sys.exit(1)

    inspect_directory(target_dir)