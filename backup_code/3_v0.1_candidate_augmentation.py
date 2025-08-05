import re
import random
import os
import numpy as np
import pandas as pd

output_dir = '../augmentation_results'
os.makedirs(output_dir, exist_ok=True)

# 1) Parse RFDs from file
IMBALANCE_DATASET_PATH = 'imbalanced_datasets/shuttle-c2-vs-c4.csv'
imbalance_df = pd.read_csv(IMBALANCE_DATASET_PATH)
ATTR_PATH_DIFF = 'diff_tuples/diff_tuples_shuttle-c2-vs-c4_min.csv'
rfd_file = '../discovered_rfds/discovered_rfds_processed/RFD4_E0.0_shuttle-c2-vs-c4_min.txt'
dependencies = []
lhs_attrs = set()
rhs_attrs = set()

rhs_break = []

with open(rfd_file, 'r') as f:
    for line in f:
        line = line.strip()
        if '->' in line:
            lhs, rhs = line.split('->')
            lhs_list = [a.split('@')[0] for a in re.split('[, ]+', lhs) if a and '@' in a]
            rhs_attr = re.match(r"(\w+)@", rhs.strip()).group(1)
            dependencies.append((lhs_list, rhs_attr))
            lhs_attrs.update(lhs_list)
            rhs_attrs.add(rhs_attr)

print('RFDs:\n', dependencies)

print("LHS attributes:", sorted(lhs_attrs))
print("RHS attributes:", sorted(rhs_attrs))

both = lhs_attrs & rhs_attrs
print("Attributes appearing both in LHS and RHS:", sorted(both))

solo_lhs = list(lhs_attrs - rhs_attrs)
print("Attributes appearing only in LHS:", sorted(solo_lhs))

solo_rhs = rhs_attrs - lhs_attrs
print("Attributes appearing only in RHS:", sorted(solo_rhs))


basename = os.path.basename(ATTR_PATH_DIFF)
attrs_df = pd.read_csv(ATTR_PATH_DIFF)


solo_lhs_df = attrs_df[attrs_df['attribute'].isin(both)]
print(solo_lhs_df)

freq_df = (solo_lhs_df
           .groupby(['idx1', 'idx2'])
           .agg(attribute_count=('attribute', 'nunique'))
           .reset_index())


total_attrs = solo_lhs_df['attribute'].nunique()
print('Total number of attributes:', total_attrs)


freq_df['in_all_attrs'] = freq_df['attribute_count'] == total_attrs
top_pairs = freq_df[freq_df['in_all_attrs']]
print(top_pairs)


def get_attr_row(attr, i1, i2):
    # exact match
    row = attrs_df[(attrs_df['attribute'] == attr) &
                   ((attrs_df['idx1'] == i1) & (attrs_df['idx2'] == i2))]
    if not row.empty:
        return row.iloc[0]
    # fallback to first occurrence
    return attrs_df[attrs_df['attribute'] == attr].iloc[0]

all_attrs = sorted(attrs_df['attribute'].unique(), key=lambda x: int(x.replace('Attr', '')))
oversampling_quantity = len(imbalance_df) - len(imbalance_df[imbalance_df['class']==1])
print("Quantity of oversampling required: ", oversampling_quantity)

oversampling_factor = oversampling_quantity / len(top_pairs)
if oversampling_factor > 1:
    oversampling_factor = int(oversampling_factor) + 1
print("Number of oversampling factor: ", oversampling_factor)

# Augmentation
new_rows = []
for _, pair in top_pairs.iterrows():
    i1, i2 = pair['idx1'], pair['idx2']
    for _ in range(oversampling_factor):
        row_data = {}
        for attr in all_attrs:
            r = get_attr_row(attr, i1, i2)
            print("ROW:\n", r)
            base = min(r['val1'], r['val2'])
            print("MIN VALUE:\n", base)
            if isinstance(base, (int, np.integer)):
                row_data[attr] = base + random.randint(0, int(r['diff']))
            else:
                row_data[attr] = base + random.uniform(0, float(r['diff']))
        print("NEW VALUE:\n",row_data)
        # Add class column
        row_data['class'] = 1
        new_rows.append(row_data)

new_df = pd.DataFrame(new_rows, columns=all_attrs + ['class'])
if len(new_df) > oversampling_quantity:
    new_df = new_df.sample(n=oversampling_quantity, random_state=42).reset_index(drop=True)
elif len(new_df) < oversampling_quantity:
    print(f"Warning: generated only {len(new_df)} samples but need {oversampling_quantity}.")

print(new_df.shape)
new_df.to_csv(os.path.join(output_dir, f'generated_{basename}'), index=False)











