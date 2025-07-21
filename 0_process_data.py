from os import rename

import pandas as pd
import argparse
import os

DF_ORIG_PATH = 'imbalanced_datasets'

for file in os.listdir(DF_ORIG_PATH):
    if file.endswith('_min.csv') or file.endswith('.dat'):
        pass
    else:
        dataset = pd.read_csv(os.path.join(DF_ORIG_PATH, file))
        print(dataset.shape[1])
        basename = os.path.basename(file).split('.')[0]
        dataset.columns = [f'Attr{i}' for i in range(dataset.shape[1])]
        dataset.rename(columns={dataset.columns[-1]: 'class'}, inplace=True)
        print(dataset.columns)
        dataset_min = dataset[dataset['class'] == 1]
        print(dataset_min.shape)
        dataset_min.to_csv(os.path.join(DF_ORIG_PATH, f'{basename}_min.csv'), index=False)
        dataset.to_csv(os.path.join(DF_ORIG_PATH, file), index=False)





