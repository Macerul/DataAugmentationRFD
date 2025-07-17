import pandas as pd
import os


DIFF_PATH = 'diff_matrices/pw_diff_mx_vehicle0_1.csv'
ORIGINAL_PATH = 'imbalanced_datasets/vehicle0_1.csv'

orig_df = pd.read_csv(ORIGINAL_PATH)
diff_df = pd.read_csv(DIFF_PATH)

basename = os.path.basename(ORIGINAL_PATH)

rfd_text = """
Attr1 -> class@12.0
Attr10 -> class@12.0
Attr11 -> class@12.0
Attr12 -> class@12.0
Attr13 -> class@12.0
Attr14 -> class@12.0
Attr15 -> class@12.0
Attr16 -> class@12.0
Attr17 -> class@12.0
Attr18 -> class@12.0
Attr2 -> class@12.0
Attr3 -> class@12.0
Attr4 -> class@12.0
Attr5 -> class@12.0
Attr6 -> class@12.0
Attr7 -> class@12.0
Attr8 -> class@12.0
Attr9 -> class@12.0
Attr1 -> Attr9@12.0
Attr10 -> Attr9@12.0
Attr11 -> Attr9@12.0
Attr12 -> Attr9@12.0
Attr13 -> Attr9@12.0
Attr14 -> Attr9@12.0
Attr15 -> Attr9@12.0
Attr16 -> Attr9@12.0
Attr17 -> Attr9@12.0
Attr18 -> Attr9@12.0
Attr2 -> Attr9@12.0
Attr3 -> Attr9@12.0
Attr4 -> Attr9@12.0
Attr5 -> Attr9@12.0
Attr6 -> Attr9@12.0
Attr7 -> Attr9@12.0
Attr8 -> Attr9@12.0
Attr10@12.0, -> Attr2@12.0
Attr12@12.0, -> Attr8@12.0
Attr12@12.0, -> Attr7@12.0
Attr12@12.0, -> Attr2@12.0
Attr11@12.0, -> Attr8@12.0
Attr13@12.0, -> Attr2@12.0
Attr7@12.0, -> Attr2@12.0
Attr7@12.0, -> Attr8@12.0
Attr3@12.0,Attr10@12.0, -> Attr8@12.0
Attr4@12.0,Attr12@12.0, -> Attr5@12.0
Attr4@12.0,Attr11@12.0, -> Attr5@12.0
Attr3@12.0,Attr13@12.0, -> Attr8@12.0
Attr4@12.0,Attr7@12.0, -> Attr5@12.0
Attr11@12.0,Attr12@12.0,Attr13@12.0,Attr15@12.0,Attr18@12.0, -> Attr17@12.0
Attr4@12.0,Attr12@12.0,Attr13@12.0,Attr15@12.0,Attr18@12.0, -> Attr17@12.0
Attr1@12.0,Attr4@12.0,Attr12@12.0,Attr13@12.0,Attr18@12.0, -> Attr17@12.0
Attr3@12.0,Attr12@12.0,Attr13@12.0,Attr14@12.0,Attr15@12.0,Attr18@12.0, -> Attr17@12.0
"""

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

output_dir = 'diff_tuples'
os.makedirs(output_dir, exist_ok=True)

result_df = transform_pairwise(orig_df, diff_df, 12)
result_df.to_csv(os.path.join(output_dir, f'diff_tuples_{basename}'), index=False)

#df_cluster = pd.read_csv('tuples_vis_clusters/vehicle0_1.csv')

# frequency of each pair across attributes
freq_df = (result_df
           .groupby(['idx1', 'idx2'])
           .agg(attribute_count=('attribute', 'nunique'))
           .reset_index())
total_attrs = result_df['attribute'].nunique()


freq_df['in_all_attrs'] = freq_df['attribute_count'] == total_attrs
top_pairs = freq_df.sort_values('attribute_count', ascending=False)
print(top_pairs)
top_pairs.to_csv(os.path.join(output_dir, f'all_attr_common_pairs_{basename}'), index=False)
print("Common pairs in all attributes:\n", top_pairs)

