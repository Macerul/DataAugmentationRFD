import random
import pandas as pd
import numpy as np
import os
from collections import Counter
from argparse import ArgumentParser
from utils import rfdParsing

# Arguments
parser = ArgumentParser("RFD-based Oversampler")
parser.add_argument('--diff_csv',     default='diff_matrices/pw_diff_mx_vehicle_1.csv')
parser.add_argument('--rfd_file',     default='discovered_rfd_processed/RFD12_E0.0_vehicle0_1.txt')
parser.add_argument('--original_csv', default='Imbalance_Datasets/vehicle0_1.csv')
parser.add_argument('--oversample_source', default='Synthetic_datasets/synth_int_dataset.csv')
parser.add_argument('--output_dir',   default='rfd_outputs')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# -----------------------------
# Step 1: Load data
# -----------------------------
df_diff = pd.read_csv(args.diff_csv)
pair_count = len(df_diff)
orig = pd.read_csv(args.original_csv)

# -----------------------------
# Step 2: Parse all RFDs
# -----------------------------
rfds = rfdParsing(args.rfd_file)
print("RFDs:")
print(rfds)

# Build a set of attributes for each RFD
for r in rfds:
    # r['lhs'] and r['rhs'] are lists of (attr, threshold)
    r['attr_set'] = { a for a,_ in r['lhs']+r['rhs'] }

# -----------------------------
# Step 3: Compute support for each RFD
# -----------------------------
"""
f:      sottoinsieme di tuple che rispettano le condizioni
        dettate dagli attributi coinvolti nella rfd
a:      attributo
t:      threshold per l'attributo a
sup:    ranking     
"""
supports = []
for r in rfds:
    f = df_diff.copy()
    for a,t in r['lhs']+r['rhs']:
        f = f[f[a] <= t]
    print(f"sottoinsieme di tuple che rispettano le condizioni per rfd {r['lhs']+r['rhs']}:")
    print(f)
    safe_name = r['dep_str'].replace('@', '_at_').replace('->', '_to_').replace(',', '_')
    f.to_csv(f"rfd_outputs/filtered_{safe_name}.csv", index=False)
    sup = len(f) / pair_count
    if sup>0: supports.append({'dep':r['dep_str'],'support':sup,'filtered':f})
supports.sort(key=lambda x: x['support'], reverse=True)

# Save summary
summary_df = pd.DataFrame([{'dependency': s['dep'], 'rank': s['support']} for s in supports])
summary_df.to_csv(os.path.join(args.output_dir, 'rfd_ranking_summary.csv'), index=False)