import re
import os
import pandas as pd

DISCOVERED_RFDs_PATH = 'discovered_rfds'
DISCOVERED_RFDs_PROCESSED_PATH = 'discovered_rfds/discovered_rfds_processed'
DATASET_PATH = 'imbalanced_datasets'


dep_pattern = re.compile(r"^\s*->\s*(\w+@[\d.]+)")

for dataset_file in os.listdir(DATASET_PATH):
    if not dataset_file.endswith('_min.csv'):
        continue

    base_name = dataset_file.split('_')[0]
    df = pd.read_csv(os.path.join(DATASET_PATH, dataset_file))
    attrs = df.columns.tolist()

    for rfd_file in os.listdir(DISCOVERED_RFDs_PATH):
        if not rfd_file.endswith(f'{base_name}_min.txt'):
            continue

        in_path = os.path.join(DISCOVERED_RFDs_PATH, rfd_file)
        out_path = os.path.join(DISCOVERED_RFDs_PROCESSED_PATH, rfd_file)

        with open(in_path, 'r') as f:
            lines = f.readlines()

        root_deps = []
        unchanged = []
        for line in lines:
            m = dep_pattern.match(line)
            # root-dep if it's exactly " -> X@thr"
            if m and not line.split('->')[0].strip():
                root_deps.append(m.group(1))
            else:
                unchanged.append(line)

        with open(out_path, 'w') as out:
            out.writelines(unchanged)
            for dep in root_deps:
                name, thr = dep.split('@')
                for attr in attrs:
                    if attr == name:
                        continue
                    out.write(f"{attr}@{thr} -> {dep}\n")
        print(f"Processed RFD file {in_path}")
