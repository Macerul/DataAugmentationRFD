import os
import shutil


cartella_dataset = "./imbalanced_datasets"
# folder to organize
cartella_risultati = "./classification_results_deepseek"

cartella_new_tuples = os.path.join(cartella_risultati, "new_tuples")
cartella_metrics = os.path.join(cartella_risultati, "metrics_results")

os.makedirs(cartella_new_tuples, exist_ok=True)
os.makedirs(cartella_metrics, exist_ok=True)

# prendo i nomi dei dataset dai file csv
dataset_names = [
    os.path.splitext(f)[0]
    for f in os.listdir(cartella_dataset)
    if f.endswith(".csv")
]

for dataset in dataset_names:
    destinazione_dataset = os.path.join(cartella_risultati, dataset)
    os.makedirs(destinazione_dataset, exist_ok=True)

    for file in os.listdir(cartella_risultati):
        file_path = os.path.join(cartella_risultati, file)

        if not os.path.isfile(file_path):
            continue

        if dataset in file:
            if "new_tuples" in file:
                shutil.move(file_path, os.path.join(cartella_new_tuples, file))
            elif "classification_results" in file:
                shutil.move(file_path, os.path.join(cartella_metrics, file))
            else:
                shutil.move(file_path, os.path.join(destinazione_dataset, file))

print("Complete")
