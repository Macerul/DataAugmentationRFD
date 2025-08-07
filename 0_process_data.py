from os import rename

import pandas as pd
import argparse
import os

DF_ORIG_PATH = 'imbalanced_datasets'

for file in os.listdir(DF_ORIG_PATH):
    if file.endswith('_min.csv') or file.endswith('.dat'):
        continue
    else:
        df = pd.read_csv(os.path.join(DF_ORIG_PATH, file))
        print(df.shape[1])
        basename = os.path.basename(file).split('.')[0]

        expected = [f'Attr{i}' for i in range(df.shape[1] - 1)] + ['class']
        if list(df.columns) != expected:
            df.columns = expected
            print(f"{file}: {df.columns}")

        if df['class'].isin(['positive', 'negative']).any():
            df['class'] = df['class'].map({'positive': 1, 'negative': 0})
            print(f"{file}: mappato positive/negative → 1/0")
        else:
            print(f"{file}: mapping non necessario, valori class già {df['class'].unique().tolist()}")


        #df.columns = [f'Attr{i}' for i in range(df.shape[1])]
        #df.rename(columns={df.columns[-1]: 'class'}, inplace=True)
        #df['class'] = df['class'].map({'positive': 1, 'negative': 0})
        #print(df.columns)
        dataset_min = df[df['class'] == 1]
        print(dataset_min.head(5))
        dataset_min_no_class = dataset_min.drop(['class'], axis=1)
        print(dataset_min_no_class.shape)
        dataset_min_no_class.to_csv(os.path.join(DF_ORIG_PATH, f'{basename}_min.csv'), index=False)
        df.to_csv(os.path.join(DF_ORIG_PATH, file), index=False)

