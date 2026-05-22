import os
import pandas as pd


def parse_filename_using_classification(fname):
    """
    Estrae dataset e metodo a partire da un nome file che contiene "_classification"
    Esempio: "Migraine_onevsrest_3_classification_results_aug_gridsearch_smote.csv"
    → dataset = "Migraine_onevsrest_3", method = ultimo segmento dopo '_'
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    idx = base.find("_classification")
    if idx != -1:
        dataset = base[:idx]
    else:
        dataset = base.split('_')[0]
    method = base.split('_')[-1]
    return dataset, method


def collect_metrics(root_dir, output_csv):
    records = []

    # ------------------------------------------------------------------
    # Metodi standard: un'unica cartella metrics_results per metodo
    # ------------------------------------------------------------------
    method_dirs = [
        "classification_results_casTGAN",
        "classification_results_ddpm",
        "classification_results_SMOTE",
        "classification_results_SMOTECDNN",
        "classification_results_SYRFD_thr2",
        "classification_results_SYRFD_thr4",
        "classification_results_SYRFD_thr8",
        "classification_results_GOGGLE",
        "classification_results_tabdiff",
        "classification_results_tvae",
        "classification_results_llama",
        "classification_results_deepseek",
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
                    "strategy": None,                   # non applicabile
                    "model": row.get("Model"),
                    "Precision": row.get("Precision"),
                    "Recall": row.get("Recall"),
                    "F1-Score": row.get("F1-score"),
                    "G mean": row.get("G-mean"),
                    "Balanced Accuracy": row.get("Balanced-Accuracy")
                }
                records.append(rec)

    # ------------------------------------------------------------------
    # LLM con strategie multiple:
    #   classification_results_{model}/{strategy}/metrics_results/*.csv
    # ------------------------------------------------------------------
    llm_models = [
        "devstral-small-2_24b-cloud",
        "gemma3_12b",
        "gemma4_31b_cloud",
        "gpt-oss_20b-cloud",
    ]

    llm_strategies = [
        "confidence",
        "decision_tree",
        "distribution_guidance",
        "ensemble",
        "hierarchical",
        "react",
        "schema_constraints",
        "self_consistency",
    ]

    for model in llm_models:
        model_path = os.path.join(root_dir, f"classification_results_{model}")
        if not os.path.isdir(model_path):
            continue
        for strategy in llm_strategies:
            metrics_dir = os.path.join(model_path, strategy, "metrics_results")
            if not os.path.isdir(metrics_dir):
                continue
            for fname in os.listdir(metrics_dir):
                if not fname.lower().endswith(".csv"):
                    continue
                fpath = os.path.join(metrics_dir, fname)
                dataset, _ = parse_filename_using_classification(fname)
                df = pd.read_csv(fpath)
                df.columns = [c.strip() for c in df.columns]
                for _, row in df.iterrows():
                    rec = {
                        "dataset": dataset,
                        "method": model,
                        "strategy": strategy,
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
    root = r"C:/Users/mary_/PycharmProjects/DataAugmentationRFD"
    output = "aggregated.csv"
    df = collect_metrics(root, output)
    print("Aggregated CSV saved at:", output)
    print(df.head(), df.shape)