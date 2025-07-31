import os.path
import pandas as pd
import numpy as np
from itertools import combinations

""" 
n! / (k! * (n-k)!) coppie 
"""

DIFF_MATRIX_PATH = 'diff_matrices'
IMBALANCE_DATASET = 'imbalanced_datasets/shuttle-c2-vs-c4_min.csv'
OVERSAMPLED_DATASET = 'oversampled_datasets/shuttle-c2-vs-c4_min_aug.csv'


if not os.path.exists("diff_matrices"):
    os.makedirs("diff_matrices")

df = pd.read_csv(IMBALANCE_DATASET)
basename = os.path.basename(IMBALANCE_DATASET)

df.reset_index(inplace=True)
df.rename(columns={'index': 'tuple_id'}, inplace=True)

attribute_columns = [col for col in df.columns if col not in ['tuple_id']]

diff_list = []

# Iterate over each unique pair of rows using combinations
for (_, row_i), (_, row_j) in combinations(df.iterrows(), 2):
    # Compute absolute difference for each attribute column
    diff_values = np.abs(row_i[attribute_columns] - row_j[attribute_columns])

    # Create a dictionary with pair identifiers and the differences
    diff_entry = {
        'tuple_pair': f"t{row_i['tuple_id']},t{row_j['tuple_id']}"
    }
    # Add each attribute's difference to the dictionary
    for col in attribute_columns:
        diff_entry[f"{col}"] = diff_values[col]

    diff_list.append(diff_entry)


diff_df = pd.DataFrame(diff_list)
diff_df.to_csv(os.path.join(DIFF_MATRIX_PATH, f'pw_diff_mx_{basename}'), index=False)
print(diff_df)
