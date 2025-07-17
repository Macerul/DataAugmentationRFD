import pandas as pd
import random
import os


PAIRS_PATH = 'diff_tuples/all_attr_common_pairs_vehicle0_1.csv'
ATTR_PATH = 'diff_tuples/diff_tuples_vehicle0_1.csv'
basename = os.path.basename(ATTR_PATH)
pairs_df = pd.read_csv(PAIRS_PATH)
common_pairs = pairs_df[pairs_df['in_all_attrs']==True]
#limited_pairs = pairs_df.head(100)
attrs_df = pd.read_csv(ATTR_PATH)


def get_attr_row(attr, i1, i2):
    # exact match
    row = attrs_df[(attrs_df['attribute'] == attr) &
                   ((attrs_df['idx1'] == i1) & (attrs_df['idx2'] == i2))]
    if not row.empty:
        return row.iloc[0]
    # fallback to first occurrence
    return attrs_df[attrs_df['attribute'] == attr].iloc[0]

output_dir = 'augmentation_results'
os.makedirs(output_dir, exist_ok=True)

all_attrs = sorted(attrs_df['attribute'].unique(), key=lambda x: int(x.replace('Attr', '')))

# Augmentation
new_rows = []
for _, pair in common_pairs.iterrows():
    i1, i2 = pair['idx1'], pair['idx2']
    row_data = {}
    for attr in all_attrs:
        r = get_attr_row(attr, i1, i2)
        print("ROW:\n", r)
        base = min(r['val1'], r['val2'])
        print("MIN VALUE:\n", base)
        row_data[attr] = base + random.randint(0, int(r['diff']))
        print("NEW VALUE:\n",row_data)
    # Add class column
    row_data['class'] = 1
    new_rows.append(row_data)

new_df = pd.DataFrame(new_rows, columns=all_attrs + ['class'])

# Save to CSV
new_df.to_csv(os.path.join(output_dir, f'generated_{basename}'), index=False)

