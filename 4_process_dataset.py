import pandas as pd
import argparse
import os

DF_ORIG_PATH = 'imbalanced_datasets/vehicle0.csv'
DF_ORIG_MIN_PATH = 'imbalanced_datasets/vehicle0_min.csv'
GENERATED_TUPLES_PATH = 'augmentation_results/generated_diff_tuples_vehicle0_min.csv'
OVERSAMPLED_DATASET_PATH = 'oversampled_datasets'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type = str, default = DF_ORIG_PATH)
parser.add_argument('--dataset_min_path', type = str, default = DF_ORIG_MIN_PATH)
parser.add_argument('--generated_tuples_path', type = str, default = GENERATED_TUPLES_PATH)
args = parser.parse_args()

df = pd.read_csv(args.dataset_path)
df_basename = os.path.basename(args.dataset_path).split('.')[0]

df_min = pd.read_csv(args.dataset_min_path)
df_basename_min = os.path.basename(args.dataset_min_path).split('.')[0]
df_tuples = pd.read_csv(args.generated_tuples_path)
print(df_tuples.shape)

ovs_source = len(df) - len(df_min)
print("Number of tuples required to balance classes: ", ovs_source)
df_augmented = df_tuples.head(ovs_source)
print(df_augmented.shape)


df_oversampled = pd.concat([df, df_augmented], ignore_index=True)
df_oversampled.to_csv(os.path.join(OVERSAMPLED_DATASET_PATH, f'{df_basename}_aug.csv'), index = False)
df_min_oversampled = pd.concat([df_min, df_augmented], ignore_index=True)
df_min_oversampled.to_csv(os.path.join(OVERSAMPLED_DATASET_PATH, f'{df_basename_min}_aug.csv'), index = False)