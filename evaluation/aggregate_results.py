import os
import pandas as pd

def parse_filename_using_classification(fname):
    """
    Estrae dataset e metodo a partire da un nome file che contiene "_classification"
    Esempio: "Migraine_onevsrest_3_classification_results_aug_gridsearch_smote.csv"
    → dataset = "Migraine_onevsrest_3", method = ultimo segmento dopo '_'
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    # cerca la posizione di "_classification"
    idx = base.find("_classification")
    if idx != -1:
        dataset = base[:idx]  # tutto quello prima di "_classification"
    else:
        # fallback: se non lo trovi, usa tutto fino al primo underscore
        dataset = base.split('_')[0]
    # il metodo lo prendi come parte finale dopo l’ultimo underscore
    method = base.split('_')[-1]
    return dataset, method

def collect_metrics(root_dir, output_csv):
    records = []
    method_dirs = [
        "classification_results_casTGAN",
        "classification_results_SMOTE",
        "classification_results_SMOTECDNN",
        "classification_results_SYRFD_thr2",
        "classification_results_SYRFD_thr4",
        "classification_results_SYRFD_thr8",
        "classification_results_llama",
        "classification_results_deepseek"
    ]
    for method_folder in method_dirs:
        method_path = os.path.join(root_dir, method_folder)
        if not os.path.isdir(method_path):
            continue
        metrics_dir = os.path.join(method_path, "metrics_results")
        if not os.path.isdir(metrics_dir):
            continue
        for fname in os.listdir(metrics_dir):
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(metrics_dir, fname)
            dataset, method = parse_filename_using_classification(fname)
            df = pd.read_csv(fpath)
            df.columns = [c.strip() for c in df.columns]
            for _, row in df.iterrows():
                rec = {
                    "dataset": dataset,
                    "method": method,
                    "model": row.get("Model"),
                    "Precision": row.get("Precision"),
                    "Recall": row.get("Recall"),
                    "F1-Score": row.get("F1-score"),
                    "G mean": row.get("G-mean"),
                    "Balanced Accuracy": row.get("Balanced-Accuracy")
                }
                records.append(rec)
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False, sep=";")
    return out_df

if __name__ == "__main__":
    root = r"C:/Users/gianp/Desktop/Codes/github/DataAugmentationRFD"
    output = "aggregated.csv"
    df = collect_metrics(root, output)
    print("Aggregated CSV saved at:", output)
    print(df.head(), df.shape)
