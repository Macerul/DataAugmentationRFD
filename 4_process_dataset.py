import pandas as pd
import argparse
import os

DF_ORIG_PATH = 'imbalanced_datasets/shuttle-c2-vs-c4.csv'
DF_ORIG_MIN_PATH = 'imbalanced_datasets/shuttle-c2-vs-c4_min.csv'
GENERATED_TUPLES_PATH = 'augmentation_results/generated_diff_tuples_shuttle-c2-vs-c4_min.csv'
OVERSAMPLED_DATASET_PATH = 'oversampled_datasets'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type = str, default = DF_ORIG_PATH)
parser.add_argument('--dataset_min_path', type = str, default = DF_ORIG_MIN_PATH)
parser.add_argument('--generated_tuples_path', type = str, default = GENERATED_TUPLES_PATH)
args = parser.parse_args()

df = pd.read_csv(args.dataset_path)
df = df[df['class']==0]
df_basename = os.path.basename(args.dataset_path).split('.')[0]

df_min = pd.read_csv(args.dataset_min_path)
df_min['class'] = 1
df_basename_min = os.path.basename(args.dataset_min_path).split('.')[0]

df_oversampled = pd.concat([df, df_min], ignore_index=True)
df_oversampled.to_csv(os.path.join(OVERSAMPLED_DATASET_PATH, f'{df_basename}_aug.csv'), index = False)
