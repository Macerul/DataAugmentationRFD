'''
import os
import pandas as pd

def find_filename_for_dataset(metrics_dir, dataset_names):
    """
    Cerca, in metrics_dir, un file che contenga (come sottostringa) uno qualsiasi
    dei nomi in dataset_names. Restituisce il nome del file trovato, o None.
    """
    for fname in os.listdir(metrics_dir):
        if not fname.lower().endswith(".csv"):
            continue
        for ds in dataset_names:
            if ds in fname:
                return fname
    return None

def collect_metrics(root_dir, dataset_names, output_csv):
    records = []
    method_dirs = [
        "classification_results_casTGAN",
        "classification_results_SMOTE",
        "classification_results_SMOTECDNN",
        "classification_results_system"
    ]
    for method_folder in method_dirs:
        method_path = os.path.join(root_dir, method_folder)
        if not os.path.isdir(method_path):
            continue
        metrics_dir = os.path.join(method_path, "metrics_results")
        if not os.path.isdir(metrics_dir):
            continue
        # Cerchiamo solo i file relativi ai dataset che ci interessano
        fname = find_filename_for_dataset(metrics_dir, dataset_names)
        if fname is None:
            # nessun file trovato per quei dataset in questa cartella, salta
            continue
        fpath = os.path.join(metrics_dir, fname)
        # scarta se non è csv
        if not fpath.lower().endswith(".csv"):
            continue
        # determiniamo il nome dataset usato: prendilo da dataset_names che è substring
        dataset = next(ds for ds in dataset_names if ds in fname)
        # metodo: lo prendi come ultima parte del nome del file
        base = os.path.splitext(os.path.basename(fname))[0]
        method = base.split('_')[-1]
        df = pd.read_csv(fpath)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            rec = {
                "dataset": dataset,
                "method": method,
                "model": row.get("Model"),
                "precision": row.get("Precision"),
                "recall": row.get("Recall"),
                "F1-Score": row.get("F1-score"),
                "g_mean": row.get("G-mean"),
                "balanced_accuracy": row.get("Balanced-Accuracy")
            }
            records.append(rec)
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False,sep=";")
    return out_df

if __name__ == "__main__":
    root = "C:/Users/Utente\Desktop\lavoro\github\DataAugmentationRFD"
    # Qui passi l'array con i nomi dei dataset (in stringhe che corrispondono ai nomi nei file)
    dataset_names = ["Migraine_onevsrest_0","Migraine_onevsrest_1","Migraine_onevsrest_2",
                     "Migraine_onevsrest_3","Migraine_onevsrest_4","Migraine_onevsrest_5",
                     "Obesity_onevsrest_0","Obesity_onevsrest_1","Obesity_onevsrest_2",
                     "Obesity_onevsrest_3","Obesity_onevsrest_4","Obesity_onevsrest_5",
                     "Obesity_onevsrest_6","Abalone19","Cleveland-0_vs_4","Dermatology-6",
                     "Ecoli1","ecoli-0_vs_1","Iris0","kddcup-guess_passwd_vs_satan","New_Thyroid1","Newthyroid2",
                     "Page_blocks0","Page_blocks-1-3_vs_4","Pima","Transfusion","Vehicle0",
                     "Vehicle1","Vehicle2","Vehicle3","Vowel0","Yeast1","Yeast3"]
    output = "aggregated_selected_datasets.csv"
    df = collect_metrics(root, dataset_names, output)
    print("CSV aggregato salvato:", output)
    print(df.head())
'''

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
        "classification_results_system"
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
                    "precision": row.get("Precision"),
                    "recall": row.get("Recall"),
                    "F1-Score": row.get("F1-score"),
                    "g_mean": row.get("G-mean"),
                    "balanced_accuracy": row.get("Balanced-Accuracy")
                }
                records.append(rec)
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False, sep=";")
    return out_df

if __name__ == "__main__":
    root = r"C:/Users/Utente/Desktop/lavoro/github/DataAugmentationRFD"
    output = "aggregated.csv"
    df = collect_metrics(root, output)
    print("Aggregated CSV saved at:", output)
    print(df.head(), df.shape)
