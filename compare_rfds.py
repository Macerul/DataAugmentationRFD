import pandas as pd
import re
import os

def create_rfd_csv_with_header_from_dataset(dataset_path,rfd_txt_path,output_csv_path):
    # Leggi header dal dataset
    dataset_df = pd.read_csv(dataset_path, sep=',', nrows=0)
    attributes = list(dataset_df.columns)

    rows = []

    with open(rfd_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or '->' not in line:
                continue

            lhs_raw, rhs_raw = map(str.strip, line.split('->'))

            # RHS parsing
            rhs_match = re.match(r'(\w+)@([\d.]+)', rhs_raw)
            if not rhs_match:
                continue
            rhs_attr, rhs_thresh = rhs_match.groups()

            if lhs_raw == '':
                # Caso: -> attr@X  → espansione
                for attr in attributes:
                    if attr == rhs_attr:
                        continue
                    row = {'RHS': rhs_attr}
                    for a in attributes:
                        if a == attr or a == rhs_attr:
                            row[a] = rhs_thresh
                        else:
                            row[a] = '?'
                    rows.append(row)
            else:
                row = {'RHS': rhs_attr}
                for a in attributes:
                    row[a] = '?'

                for part in lhs_raw.split(','):
                    match = re.match(r'(\w+)@([\d.]+)', part.strip())
                    if match:
                        attr, thresh = match.groups()
                        if attr in attributes:
                            row[attr] = thresh

                if rhs_attr in attributes:
                    row[rhs_attr] = rhs_thresh
                rows.append(row)

    # Crea DataFrame e salva
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False, sep=';')


dataset = "vehicle0_1_test"
thr = "RFD12"
rfds_aug = dataset + "_min_aug"

create_rfd_csv_with_header_from_dataset(f"./imbalanced_datasets/{dataset}.csv",f"./discovered_rfds/{thr}_E0.0_{dataset}.txt", f'./discovered_rfds/parsed_rfds/{thr}_E0.0_{dataset}.csv')
create_rfd_csv_with_header_from_dataset(f"./imbalanced_datasets/{dataset}.csv",f"./discovered_rfds/rfds_after_augmentation/{thr}_E0.0_{rfds_aug}.txt", f'./discovered_rfds/rfds_after_augmentation/parsed_rfds/{thr}_E0.0_{rfds_aug}.csv')
