import pandas as pd
import os
import re


DIFF_PATH = '../diff_matrices/pw_diff_mx_wisconsin_min.csv'
ORIGINAL_PATH = '../imbalanced_datasets/wisconsin_min.csv'

orig_df = pd.read_csv(ORIGINAL_PATH)
diff_df = pd.read_csv(DIFF_PATH)

basename = os.path.basename(ORIGINAL_PATH)

def transform_pairwise(orig_df: pd.DataFrame,
                       diff_df: pd.DataFrame,
                       thr: float) -> pd.DataFrame:

    records = []
    for _, row in diff_df.iterrows():
        t1, t2 = row['tuple_pair'].strip('"').split(',')
        idx1, idx2 = int(t1[1:]), int(t2[1:])
        for attr in (c for c in diff_df.columns if c.startswith('Attr')):
            if row[attr] <= thr:
                records.append({
                    'attribute': attr,
                    'idx1': idx1,
                    'val1': orig_df.at[idx1, attr],
                    'idx2': idx2,
                    'val2': orig_df.at[idx2, attr],
                    'diff': row[attr]
                })

    df = pd.DataFrame(records)
    return df

output_dir = '../diff_tuples'
os.makedirs(output_dir, exist_ok=True)

'''                             
                SET THR
'''
result_df = transform_pairwise(orig_df, diff_df, 12)
result_df.to_csv(os.path.join(output_dir, f'diff_tuples_{basename}'), index=False)





