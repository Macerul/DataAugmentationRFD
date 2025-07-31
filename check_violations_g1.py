import pandas as pd
import ast
import os

#TODO:
# dopo aver eseguito compare rfd so quali rfd vengono violate,
# queste andranno in dependencies, mi verranno restituiti i pairs (tuple_pair)
# che violano e li vado ad eliminare
# (stabilire un max iter come parametro per non generare loop)
# se non è possibile raggiungere la quantità di oversampling desiderata
# ossia se non è possibile generare tuple che non introducono violazioni
# allora si interviene sull'undersampling della classe maggioritaria per
# bilanciare

DIFF_MATRIX_PATH = 'diff_matrices/pw_diff_mx_shuttle-c2-vs-c4_min_aug.csv'
df = pd.read_csv(DIFF_MATRIX_PATH, index_col='tuple_pair')
basename = os.path.basename(DIFF_MATRIX_PATH).split('_')[3]

dependencies = []
with open('discovered_rfds/RFD4_E0.0_shuttle-c2-vs-c4_min_NF.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        lhs_raw, rhs_raw = ast.literal_eval(line)
        lhs_clean = [col.split('_', 1)[0] for col in lhs_raw]
        rhs_clean = rhs_raw.split('_', 1)[0]
        dependencies.append((lhs_clean, rhs_clean))


df_list = []

"""
COMPUTATION OF G1 MEASURE
"""
for lhs, rhs in dependencies:
    # Filter rows where all LHS diffs <= 12 and RHS diff > 12
    mask = (df[lhs] <= 12).all(axis=1) & (df[rhs] > 12)
    viol = df[mask].copy()
    viol_indices = df.index[mask].tolist()
    print([index.split(',') for index in viol_indices])
    if not viol.empty:
        viol['dependency'] = f"{','.join(lhs)} -> {rhs}"
        df_list.append(viol[['dependency'] + lhs + [rhs]])

if df_list:
    violations_df = pd.concat(df_list)
    #violations_df.to_csv('violations.csv')
    print("Violations found:")
    print(violations_df)
    # Optionally save:
    violations_df.to_csv(f'{basename}_violations.csv')
else:
    print("No violations found.")
