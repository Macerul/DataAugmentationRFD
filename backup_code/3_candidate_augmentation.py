import pandas as pd
import numpy as np
import random
import os


PAIRS_PATH = 'diff_tuples/all_attr_common_pairs_vehicle0_min.csv'
ATTR_PATH = 'diff_tuples/diff_tuples_vehicle0_min.csv'
IMBALANCE_DATASET_PATH = 'imbalanced_datasets/vehicle0.csv'
basename = os.path.basename(ATTR_PATH)
pairs_df = pd.read_csv(PAIRS_PATH)
common_pairs = pairs_df[pairs_df['in_all_attrs']==True]
print("Number of common_pairs: ", len(common_pairs))
attrs_df = pd.read_csv(ATTR_PATH)
imbalance_df = pd.read_csv(IMBALANCE_DATASET_PATH)

output_dir = '../augmentation_results'
os.makedirs(output_dir, exist_ok=True)



# TODO: per gli attributi su RHS prendere le coppie in comune solo agli attributi dell'RHS,
#  per tutti gli altri prendere le coppie in comune a quelli facenti parte dell'LHS
# nei casi in cui io ho X -> Y e Y-> A la perdiamo, non si può risolvere la situazione
# nei casi in cui io ho X -> Y e Y,B-> A per Y rispetto a destra, B lo rompo esempio pratico:
#prendo le tuple che rixpettano X->Y e quelle vhe rispettano Z->X e

# se il massimo per B è 25 io mi sposto fuori dal range di thr+1 a partire da thr+1 io genero di max +thr,
# quindi parto da 38 e posso arrivare massimo a 50


# Range matrix per ogni tupla candidata (min diff value > della thr - thr) - 1
# e ottengo il range in cui mi posso muovere


# TODO
"""
Step 1 - Implementare la parte dove si fa il controllo su se la tupla inserita viola allora si scarta
quindi validare e inserire solo quelle che non violano, tramite g1 measure
(entropia logica, paper measuring AFD: a comparative study).

Step 2 - Implementare la parte dove se lo step 1 consente di effettuare un oversampling
non sufficiente allora va a bilanciare rimuovendo i campioni della classe
maggioritaria (stripped)

*** IDEA **** 
what if I generalize or specialize rfd that got violated? (????)

Se per ogni tupla genero due tuple, la prima seguendo la strategia e la seconda aumentando i valori di quella generata di max thr

Seleziono gli attributi dell'lhs (quelli che "attivano" la dipendenza) e devo trovare il modo di generare l'RHS in modo che se sono simili
lo siano anche a dx
"""





def get_attr_row(attr, i1, i2):
    # exact match
    row = attrs_df[(attrs_df['attribute'] == attr) &
                   ((attrs_df['idx1'] == i1) & (attrs_df['idx2'] == i2))]
    if not row.empty:
        return row.iloc[0]
    # fallback to first occurrence
    return attrs_df[attrs_df['attribute'] == attr].iloc[0]

all_attrs = sorted(attrs_df['attribute'].unique(), key=lambda x: int(x.replace('Attr', '')))
oversampling_quantity = len(imbalance_df) - len(imbalance_df[imbalance_df['class']==1])
print("Quantity of oversampling required: ", oversampling_quantity)

oversampling_factor = oversampling_quantity / len(common_pairs)
if oversampling_factor > 1:
    oversampling_factor = int(oversampling_factor) + 1
print("Number of oversampling factor: ", oversampling_factor)

# Augmentation
new_rows = []
for _, pair in common_pairs.iterrows():
    i1, i2 = pair['idx1'], pair['idx2']
    for _ in range(oversampling_factor):
        row_data = {}
        for attr in all_attrs:
            r = get_attr_row(attr, i1, i2)
            print("ROW:\n", r)
            base = min(r['val1'], r['val2'])
            print("MIN VALUE:\n", base)
            if isinstance(base, (int, np.integer)):
                row_data[attr] = base + random.randint(0, int(r['diff']))
            else:
                row_data[attr] = base + random.uniform(0, float(r['diff']))
            print("NEW VALUE:\n",row_data)
        # Add class column
        row_data['class'] = 1
        new_rows.append(row_data)

new_df = pd.DataFrame(new_rows, columns=all_attrs + ['class'])
if len(new_df) > oversampling_quantity:
    new_df = new_df.sample(n=oversampling_quantity, random_state=42).reset_index(drop=True)
elif len(new_df) < oversampling_quantity:
    print(f"Warning: generated only {len(new_df)} samples but need {oversampling_quantity}.")

print(new_df.shape)
new_df.to_csv(os.path.join(output_dir, f'generated_{basename}'), index=False)

