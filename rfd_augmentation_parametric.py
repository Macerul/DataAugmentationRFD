import re
import random
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations


class RFDAwareAugmenter:
    def __init__(self, imbalance_dataset_path,
                  rfd_file_path,  oversampling,
                  threshold=4, max_iter=100, selected_rfds=None):
        """
        Initialization

        Args:
            imbalance_dataset_path: Path to the imbalanced dataset
            attr_diff_path: Path to the attribute differences file
            rfd_file_path: Path to the RFD file
            diff_matrix_path: Path to the distance matrix file
            threshold: Similarity threshold for RFDs
            max_iter: Maximum iterations to try generating a valid tuple
            selected_rfds: List of specific RFDs to respect (None = all RFDs)
        """

        # Create output directory
        self.output_dir = 'augmentation_results'
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_diff_dir = 'diff_matrices'
        os.makedirs(self.output_diff_dir, exist_ok=True)
        self.output_diff_tuples_dir = 'diff_tuples'
        os.makedirs( self.output_diff_tuples_dir, exist_ok=True)
        self.imbalance_dir = 'imbalanced_datasets'

        # Initialize parameters
        self.threshold = threshold
        self.max_iter = max_iter
        self.oversampling_quantity = oversampling
        self.selected_rfds = selected_rfds
        self.last_safe_value = sys.maxsize - self.threshold - 1     # quando la strategia è sys.max (sys.max - thr - 1)
        #self.last_safe_value = {}  # quando la strategia è max value della colonna + thr + 1 (incrementale)


        # Load datasets
        self.imbalance_dataset_path = imbalance_dataset_path
        self.base = os.path.basename(self.imbalance_dataset_path).split('.')[0]
        self.imbalance_df = pd.read_csv(imbalance_dataset_path)
        ###### IDENTIFY IMB CLASS ######
        counts = self.imbalance_df['class'].value_counts()
        min_class = counts.idxmin()
        self.dataset_min = self.imbalance_df[self.imbalance_df['class']== min_class]
        self.out_min_path = os.path.join(self.imbalance_dir, f'{self.base}_min.csv')
        self.dataset_min.to_csv(self.out_min_path, index=False)

        # Calcola statistiche classe maggioritaria per ogni attributo
        self.majority_stats = {}
        majority_class = self.imbalance_df[self.imbalance_df['class'] == 0]
        for attr in self.imbalance_df.columns:
            self.majority_stats[attr] = {
                'mean': majority_class[attr].mean(),
                'min': majority_class[attr].min(),
                'max': majority_class[attr].max()
            }

        self.imbalance_df_min =  pd.read_csv(self.out_min_path)
        print(f'LETTURA DATASET INIZIALE:\n{self.imbalance_df_min.head()}')

        self.integer_attrs = set()
        for attr in self.imbalance_df_min.columns:
            if (self.imbalance_df_min[attr].dtype in ['int64', 'int32'] or
                    np.issubdtype(self.imbalance_df_min[attr].dtype, np.integer)):
                self.integer_attrs.add(attr)
                print(f"{attr} è un attributo intero")


        self.out_diff_path = os.path.join(self.output_diff_dir, f'pw_diff_mx_{self.base}_min.csv')
        #self.attrs_df = pd.read_csv(attr_diff_path)
        print("Computing difference matrix...")
        self.diff_df = self._compute_diff_matrix()

        self.original_diff_matrix = pd.read_csv(self.out_diff_path, index_col='tuple_pair')
        print("MATRICE DISTANZE INIZIALE:\n", self.original_diff_matrix)

        print("Filtering difference pairs...")
        self.attrs_df = self._filter_diff_pairs()

        # Parse RFDs
        self.dependencies = self._parse_rfds(rfd_file_path)
        self._analyze_attributes()


    def order_attributes(self):
        """
        Ordina gli attributi mettendo prima quelli non-booleani
        (con più di due valori possibili), poi i booleani.
        All'interno di ciascun gruppo li ordina per indice numerico.
        """
        #attrs = self.attrs_df['attribute'].unique()
        #attrs = self.attrs_df['attribute'].unique()
        attrs = self.dataset_min.columns[:-1]
        print('Attributes: \n', attrs)

        ordered = sorted(
            attrs,
            key=lambda x: (
                1 if self.check_bool_attr(x) else 0,  # 0 = non booleani, 1 = booleani
                int(x.replace('Attr', ''))  # ordinamento numerico
            )
        )
        return ordered

    def check_bool_attr(self, attr):
        if self.imbalance_df_min[attr].isin([0, 1]).all():
            print(f"{attr} is a boolean attribute")
            return True
        else:
            return False


    def _parse_rfds(self, rfd_file_path):
        """Parse RFDs from the file."""
        dependencies = []

        with open(rfd_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '->' in line:
                    lhs, rhs = line.split('->')
                    lhs_list = [a.split('@')[0] for a in re.split('[, ]+', lhs) if a and '@' in a]
                    rhs_attr = re.match(r"(\w+)@", rhs.strip()).group(1)
                    dependencies.append((lhs_list, rhs_attr))

        # Filter by selected RFDs if specified
        if self.selected_rfds is not None:
            filtered_deps = []
            for lhs, rhs in dependencies:
                dep_str = f"{','.join(lhs)} -> {rhs}"
                if dep_str in self.selected_rfds:
                    filtered_deps.append((lhs, rhs))
            dependencies = filtered_deps
        print('DEPENDENCIES:\n', dependencies)
        return dependencies

    def _compute_diff_matrix(self) -> pd.DataFrame:
        """
        Computes the pairwise absolute differences for each attribute
        and saves a CSV in diff_dir.
        Returns:
            diff_df (pd.DataFrame): DataFrame of pairwise differences.
        """
        self.imbalance_df_min.reset_index(inplace=True)
        self.imbalance_df_min.rename(columns={'index': 'tuple_id'}, inplace=True)
        attribute_columns = [col for col in self.imbalance_df_min.columns if col not in ['tuple_id']]
        diff_list = []
        for (_, row_i), (_, row_j) in combinations(self.imbalance_df_min.iterrows(), 2):
            diff_values = np.abs(row_i[attribute_columns] - row_j[attribute_columns])

            diff_entry = {
                'tuple_pair': f"t{int(row_i['tuple_id'])},t{int(row_j['tuple_id'])}"
            }
            # Add each attribute's difference to the dictionary
            for col in attribute_columns:
                diff_entry[f"{col}"] = diff_values[col]

            diff_list.append(diff_entry)

        diff_df = pd.DataFrame(diff_list)
        # Save to file

        diff_df.to_csv(self.out_diff_path, index=False)
        print(f"Saved initial difference matrix to {self.out_diff_path}")
        return diff_df

    def _filter_diff_pairs(self) -> pd.DataFrame:
        """
        Filters the pairwise difference matrix for attribute differences <= thr.
        Saves the result to a CSV in output_dir.

        Args:
            thr: Threshold for difference filtering.

        Returns:
            result_df: DataFrame of filtered pairs.
        """
        records = []
        # Iterate over diff rows
        for _, row in self.diff_df.iterrows():
            t1, t2 = row['tuple_pair'].split(',')
            idx1, idx2 = int(t1[1:]), int(t2[1:])
            for attr in (c for c in self.diff_df.columns if c.startswith('Attr')):
                if row[attr] <= self.threshold:
                    records.append({
                        'attribute': attr,
                        'idx1': idx1,
                        'val1': self.imbalance_df_min.at[idx1, attr],
                        'idx2': idx2,
                        'val2':self.imbalance_df_min.at[idx2, attr],
                        'diff': row[attr]
                    })

        result_df = pd.DataFrame(records)
        base = os.path.basename(self.imbalance_dataset_path).split('.')[0]
        out_path = os.path.join(self.output_diff_tuples_dir, f'diff_tuples_{base}_min.csv')
        result_df.to_csv(out_path, index=False)
        print(f"Saved filtered tuples to {out_path}")
        return result_df

    def _analyze_attributes(self):
        """Analyze which attributes appear in LHS and RHS of dependencies."""
        lhs_attrs = set()
        rhs_attrs = set()


        for lhs_list, rhs_attr in self.dependencies:
            lhs_attrs.update(lhs_list)
            rhs_attrs.add(rhs_attr)


        self.lhs_attrs = lhs_attrs
        self.rhs_attrs = rhs_attrs
        self.both_attrs = lhs_attrs & rhs_attrs
        #self.all_attrs = sorted(self.attrs_df['attribute'].unique(),  key=lambda x: int(x.replace('Attr', '')))
        self.all_attrs = self.order_attributes()
        # Attributes not found in any dependency
        self.dependency_attrs = lhs_attrs | rhs_attrs
        self.no_dependency_attrs = set(self.all_attrs) - self.dependency_attrs


        print(f"Analyzing {len(self.dependencies)} RFDs")
        print(f"All attributes: {self.all_attrs}")
        print(f"Attributes in both LHS and RHS: {sorted(self.both_attrs)}")
        print(f"Attributes in LHS only: {sorted(self.lhs_attrs)}")
        print(f"Attributes not in any dependency: {sorted(self.no_dependency_attrs)}")

    def _get_top_pairs(self):
        """Get tuple pairs that appear across all 'both' attributes."""
        # Filter to attributes that appear in both LHS and RHS
        if len( self.both_attrs) != 0:
            total_both_attrs = len(self.both_attrs)
            relevant_df = self.attrs_df[self.attrs_df['attribute'].isin(self.both_attrs)]
            #print('Top pairs:\n',relevant_df.head())
        else:
            print("No attribute is present in both LHS and RHS, considering only LHS attributes...")
            total_both_attrs = len(self.lhs_attrs)
            relevant_df = self.attrs_df[self.attrs_df['attribute'].isin(self.lhs_attrs)]

        # Count how many attributes each tuple pair covers
        freq_df = (relevant_df
                   .groupby(['idx1', 'idx2'])
                   .agg(attribute_count=('attribute', 'nunique'))
                   .reset_index())

        freq_df['covers_all_both'] = freq_df['attribute_count'] == total_both_attrs

        return freq_df[freq_df['covers_all_both']]

    def _get_attr_value(self, attr, i1, i2, use_decimal=False, generated_values=None, missing_attrs=None):
        """
        Get attribute value for generation following RFD-aware logic.

        Args:
            attr: Attribute name
            i1, i2: Tuple indices
            use_decimal: If True, generate decimal values to avoid duplicates
            missing_attrs: attribute set which needs to use sys.max to avoid alterate patterns
        """

        if missing_attrs and attr in missing_attrs:
            print(f"  {attr}: Missing in pair coverage, using sys.max strategy")
            return self._get_safe_fallback_value(attr, use_decimal)

        # Try to find the tuple pair data for this attribute
        row = self.attrs_df[(self.attrs_df['attribute'] == attr) &
                            ((self.attrs_df['idx1'] == i1) & (self.attrs_df['idx2'] == i2))]

        if not row.empty:
            # If attr not in any rfd, generate a value random(0, diff)
            if attr in self.no_dependency_attrs:
                print(f'{attr} not in any rfd, generate a value random(0, diff)')
                r = row.iloc[0]
                val1, val2 = r['val1'], r['val2']
                min_val = min(val1, val2)
                diff_val = r['diff']
                if isinstance(min_val, (int, np.integer)):
                    random_val = min_val + random.randint(0, int(diff_val))
                else:
                    random_val = min_val + random.uniform(0, float(diff_val))
                print(f"  {attr}: No dependency, min={min_val}, diff={diff_val}, generating: {random_val:.2f}")
                return round(random_val, 2)
            r = row.iloc[0]
            val1, val2 = r['val1'], r['val2']
            diff = r['diff']

            print(f"  {attr}: val1={val1}, val2={val2}, diff={diff}, threshold={self.threshold}")

            # Check if values are similar (within threshold) or dissimilar
            if diff <= self.threshold:
                # Values are similar - we can generate within their range
                min_val = min(val1, val2)
                max_val = max(val1, val2)

                if diff == 0:
                    # Identical values - must preserve exact value
                    print(f"    → Identical values, using {val1}")
                    return val1
                if diff == 1:
                    try:
                        if generated_values and len(generated_values) > 0:
                            # consider only attributes already generated (exclude current attr)
                            keys = [k for k in generated_values.keys() if k != attr and k in self.all_attrs]
                            if len(keys) > 0:
                                vec_generated = np.array([float(generated_values[k]) for k in keys])
                                vec_i1 = np.array([float(self.imbalance_df_min.at[i1, k]) for k in keys])
                                vec_i2 = np.array([float(self.imbalance_df_min.at[i2, k]) for k in keys])
                                d1 = np.linalg.norm(vec_generated - vec_i1)
                                d2 = np.linalg.norm(vec_generated - vec_i2)
                                chosen = val1 if d1 <= d2 else val2
                                print(f"    → diff==1: euclidean d1={d1:.2f}, d2={d2:.2f}, choosing: {chosen}")
                                return chosen
                    except Exception as e:
                        # fallback in case of unexpected numeric issues
                        print(f"    → Warning computing euclidean distance: {e}")

                        # Fallback: if no generated_values yet or error, choose random
                    chosen = random.randrange(min_val, max_val+1)
                    print(f"    → diff==1 no generated_values yet or error: {chosen}")
                    return chosen
                else:
                    # Similar values - generate within range
                    if use_decimal:
                        generated_val = random.uniform(min_val, max_val)
                        print(f"    → Similar values, generating decimal: {generated_val:.2f}")
                        return round(generated_val, 2)
                    else:
                        if attr in self.integer_attrs: #and isinstance(min_val, (int, np.integer)) and isinstance(max_val, (int, np.integer)):
                            generated_val = random.randint(int(min_val), int(max_val))
                            print(f"    → Integer attribute {attr}, generating integer in range: {generated_val}")
                        else:
                            generated_val = round(random.uniform(min_val, max_val), 2)
                            print(f"    → Float/decimal attribute {attr}, generating decimal: {generated_val}")
                        return generated_val
            else:
                # Values are dissimilar - use safe fallback to avoid unwanted dependencies
                print(f"    → Dissimilar values (diff={diff} > {self.threshold}), using fallback")
                return self._get_safe_fallback_value(attr, use_decimal)
        else:
            # No data for this tuple pair - use fallback
            print(f"  {attr}: No tuple pair data, using fallback")
            return self._get_safe_fallback_value(attr, use_decimal)

    '''
        # quando la strategia è max value della colonna + thr + 1 (incrementale) 
    '''
    '''
    def _get_safe_fallback_value(self, attr, use_decimal=False):
        """
        Generate a safe fallback value that won't trigger unwanted dependencies.
        Uses majority class statistics to decide direction.
        """
        # Statistiche classe minoritaria (quella attuale)
        minority_min = self.imbalance_df_min[attr].min()
        minority_max = self.imbalance_df_min[attr].max()

        # Media classe maggioritaria
        majority_mean = self.majority_stats[attr]['mean']

        print(f"  {attr}: minority_min={minority_min}, minority_max={minority_max}")
        print(f"  {attr}: majority_mean={majority_mean}")

        # Decisione: se media maggioritaria > max minoritaria → scendi sotto min minoritaria
        # altrimenti → sali sopra max minoritaria
        if majority_mean > minority_max:
            # Scendi sotto il minimo minoritario
            safe_base = minority_min - self.threshold - 1
            print(f"  → Majority mean > minority max, going below: {safe_base}")
        else:
            # Sali sopra il massimo minoritario
            safe_base = minority_max + self.threshold + 1
            print(f"  → Majority mean <= minority max, going above: {safe_base}")

        if attr in self.last_safe_value:
            if majority_mean > minority_max:
                # Se stiamo scendendo, continua a scendere
                safe_base = min(safe_base, self.last_safe_value[attr] - self.threshold - 1)
            else:
                # Se stiamo salendo, continua a salire
                safe_base = max(safe_base, self.last_safe_value[attr] + self.threshold + 1)

            # Salva il valore per questo attributo
        self.last_safe_value[attr] = safe_base

        if use_decimal:
            decimal_component = round(random.random(), 2)
            safe_value = safe_base + decimal_component
            print(f"    → Safe fallback with decimal: {safe_value:.2f}")
            return safe_value
        else:
            print(f"    → Safe fallback: {safe_base}")
            return safe_base
    '''


    #'''
    def _get_safe_fallback_value(self, attr, use_decimal=False):
        """
        Generate a safe fallback value that won't trigger unwanted dependencies.

        Args:
            attr: Attribute name
            use_decimal: If True, add decimal component
        """
        # Get all rows for this attribute and find the global maximum
        #attr_rows = self.imbalance_df_min[attr]
        #print(' rows for this attribute and find the global maximum:\n', attr_rows )

        #overall_max = attr_rows.max() + self.threshold + 1
        overall_max = self.last_safe_value - self.threshold - 1  # per non attivare o violare dipendenze per gli attributi i cui pattern sono posti a 0, genero valori fuori range
        print('Overall max:', overall_max)

        self.last_safe_value = overall_max

        safe_base = overall_max

        if use_decimal:
            # Add random decimal component to ensure uniqueness
            decimal_component = round(random.random(), 2)  # 0.0 to 1.0
            safe_value = safe_base + decimal_component
            print(f"    → Safe fallback with decimal: {safe_value:.2f}")
            return safe_value
        else:
            print(f"    → Safe fallback: {safe_base}")
            return safe_base
    #'''

    def _update_distance_matrix(self, new_tuple_values, current_df, prev_matrix=None):
        """
        Update the distance matrix by adding distances to the new tuple.

        Args:
            new_tuple_values: dict of new tuple attributes
            current_df: DataFrame including the new tuple
            prev_matrix: DataFrame of previous distances
        Returns:
            Updated distance matrix DataFrame
        """
        # Start from previous matrix or empty

        #print('New tuple values:\n', new_tuple_values)
        #print('current_df:\n', current_df)


        if prev_matrix is None or prev_matrix.empty:
            updated = pd.DataFrame()
        else:
            updated = prev_matrix

        new_idx = len(current_df) - 1
        #new_name = f"t{int(new_idx)}"

        # Ensure columns for each attr
        for attr in self.all_attrs:
            if attr not in updated.columns:
                updated[attr] = []

        # Compute distances to each existing tuple
        for existing_idx in range(new_idx):
            pair_name = f"t{int(existing_idx)},t{int(new_idx)}"
            distances = {}
            for attr in self.all_attrs:
                new_val = new_tuple_values.get(attr, 0)
                existing_val = current_df.iloc[existing_idx][attr]
                distances[attr] = abs(new_val - existing_val)

            # Add new row
            updated.loc[pair_name] = distances
        #print('Matrice aggiornata:\n',updated)
        return updated


    def _check_violations(self, distance_matrix):
        """
        Check for RFD violations in the updated distance matrix.

        Args:
            distance_matrix: Updated distance matrix

        Returns:
            True if violations found, False otherwise
        """
        violations_found = False
        for lhs, rhs in self.dependencies:
            missing_cols = [col for col in lhs + [rhs] if col not in distance_matrix.columns]
            if missing_cols:
                print(f"    Warning: Missing columns {missing_cols} for dependency {lhs} -> {rhs}")
                continue

            # Trova violazioni: LHS simili ma RHS dissimili
            lhs_similar = (distance_matrix[lhs] <= self.threshold).all(axis=1)
            rhs_dissimilar = (distance_matrix[rhs] > self.threshold)
            violations = lhs_similar & rhs_dissimilar

            if violations.any():
                violation_pairs = distance_matrix.index[violations].tolist()
                print(f"    → Violation in dependency {lhs} -> {rhs}")
                print(f"      Violating pairs: {violation_pairs}")
                violations_found = True
                for pair in violation_pairs[:3]:
                    lhs_diffs = [distance_matrix.loc[pair, attr] for attr in lhs]
                    rhs_diff = distance_matrix.loc[pair, rhs]
                    print(
                        f"        {pair}: LHS diffs {lhs_diffs} ≤ {self.threshold}, RHS diff {rhs_diff} > {self.threshold}")

        return violations_found

    def _is_duplicate_tuple(self, new_tuple_data, current_df):
        """
        Check if the new tuple already exists in the current dataset.

        Args:
            new_tuple_data: Dictionary containing the new tuple's attribute values
            current_df: Current dataset to check against

        Returns:
            True if duplicate found, False otherwise
        """
        # Create a comparison series from the new tuple (excluding 'class' column for comparison)
        comparison_attrs = {attr: new_tuple_data[attr] for attr in self.all_attrs}

        # Check each existing tuple in the dataset
        for idx, existing_row in current_df.iterrows():
            # Compare all attribute values (excluding 'class' column)
            is_duplicate = True
            for attr in self.all_attrs:
                # Handle potential floating point precision issues
                if abs(existing_row[attr] - comparison_attrs[attr]) > 1e-10:
                    is_duplicate = False
                    break

            if is_duplicate:
                print(f"      Duplicate found with existing tuple at index {idx}")
                return True

        return False

    def get_safe_value_ranges(self, attr):
        """
        Calcola i range di valori 'sicuri' per un attributo che non attivino dipendenze indesiderate.

        Args:
            attr: Nome dell'attributo

        Returns:
            List of tuples: [(min_safe, max_safe), ...] range sicuri
        """
        # Prendi tutti i valori esistenti per questo attributo
        existing_values = sorted(self.imbalance_df_min[attr].unique())

        # Trova gli intervalli sicuri (distanza > threshold dai valori esistenti)
        safe_ranges = []

        for i in range(len(existing_values)):
            current_val = existing_values[i]

            # Range prima del valore corrente
            if i == 0:
                # Prima del primo valore
                range_start = current_val - self.threshold - 5  # buffer extra
                range_end = current_val - self.threshold - 1
                if range_start < range_end:
                    safe_ranges.append((range_start, range_end))

            # Range dopo il valore corrente
            if i == len(existing_values) - 1:
                # Dopo l'ultimo valore
                range_start = current_val + self.threshold + 1
                range_end = current_val + self.threshold + 10  # buffer extra
                safe_ranges.append((range_start, range_end))
            else:
                # Tra valori consecutivi
                next_val = existing_values[i + 1]
                range_start = current_val + self.threshold + 1
                range_end = next_val - self.threshold - 1

                if range_start < range_end:
                    safe_ranges.append((range_start, range_end))

        print(f"  {attr} safe ranges: {safe_ranges}")
        return safe_ranges

    def generate_safe_value(self, attr, avoid_similarity=True):
        """
        Genera un valore sicuro per un attributo che non attivi dipendenze.

        Args:
            attr: Nome dell'attributo
            avoid_similarity: Se evitare similarità con valori esistenti

        Returns:
            Valore sicuro per l'attributo
        """
        if avoid_similarity:
            safe_ranges = self.get_safe_value_ranges(attr)

            if safe_ranges:
                # Scegli un range casuale
                chosen_range = random.choice(safe_ranges)
                min_val, max_val = chosen_range

                # Genera valore nel range
                if isinstance(min_val, (int, np.integer)) and isinstance(max_val, (int, np.integer)):
                    safe_val = random.randint(int(min_val), int(max_val))
                else:
                    safe_val = round(random.uniform(min_val, max_val), 2)

                print(f"    Generated safe value for {attr}: {safe_val} in range {chosen_range}")
                return safe_val

        # Fallback: usa il metodo esistente
        return self._get_safe_fallback_value(attr, use_decimal=True)

    def repair_violation_with_fallback(self, row_data, violated_dependency, current_df, distance_matrix):
        """
        Prova prima la riparazione normale, poi usa safe_fallback per tutti gli attributi coinvolti.
        """
        lhs_list, rhs_attr = violated_dependency

        # Tentativo 1-3: Usa la strategia normale di riparazione
        repaired = self.repair_violation(row_data, violated_dependency, current_df, distance_matrix)

        if repaired is not None:
            return repaired

        # Strategia 4: ULTIMA RISORSA - Usa safe_fallback per tutti gli attributi della dipendenza
        print(" Safe fallback per tutti gli attributi della dipendenza")
        row_data_test = row_data.copy()

        # Applica safe_fallback a TUTTI gli attributi coinvolti nella dipendenza
        all_dep_attrs = lhs_list + [rhs_attr]

        for attr in all_dep_attrs:
            old_val = row_data_test[attr]
            safe_val = self._get_safe_fallback_value(attr, use_decimal=True)
            row_data_test[attr] = safe_val
            print(f"      {attr}: {old_val} -> {safe_val} (safe fallback)")

        # Verifica se funziona
        temp_df = pd.concat([current_df, pd.DataFrame([row_data_test])], ignore_index=True)
        temp_matrix = self._update_distance_matrix(row_data_test, temp_df, distance_matrix.copy())

        if not self._check_violations(temp_matrix):
            print(f"      ✓ Riparazione con safe fallback riuscita")
            return row_data_test

        print("    ✗ Anche safe fallback non risolve")
        return None


    def repair_violation(self, row_data, violated_dependency, current_df, distance_matrix):
        """
        Ripara una violazione specifica modificando strategicamente i valori della tupla.

        Args:
            row_data: Dati della tupla che viola
            violated_dependency: (lhs_list, rhs_attr) della dipendenza violata
            current_df: Dataset corrente
            distance_matrix: Matrice delle distanze

        Returns:
            row_data modificato o None se impossibile riparare
        """
        lhs_list, rhs_attr = violated_dependency
        print(f"    Riparazione violazione: {lhs_list} -> {rhs_attr}")

        # Strategia 1: Modifica RHS per renderlo simile (se non crea altre violazioni)
        print("    Tentativo 1: Riparazione RHS")
        original_rhs = row_data[rhs_attr]

        # Trova il valore RHS target dalla tupla più simile negli attributi LHS
        last_pair = distance_matrix.index[-1]  # L'ultima coppia aggiunta
        existing_idx = int(last_pair.split(',')[0][1:])  # Estrai l'indice della tupla esistente

        # Usa il valore RHS della tupla simile
        target_rhs = current_df.iloc[existing_idx][rhs_attr]

        # Prova piccole variazioni attorno al target
        for variation in [0, 0.1, -0.1, 0.5, -0.5, 1, -1]:
            test_rhs = target_rhs + variation
            row_data_test = row_data.copy()
            row_data_test[rhs_attr] = test_rhs

            # Verifica se questa modifica risolve la violazione senza crearne altre
            temp_df = pd.concat([current_df, pd.DataFrame([row_data_test])], ignore_index=True)
            temp_matrix = self._update_distance_matrix(row_data_test, temp_df, distance_matrix.copy())

            if not self._check_violations(temp_matrix):
                print(f"      ✓ RHS riparato: {original_rhs} -> {test_rhs}")
                return row_data_test

        # Strategia 2: Modifica LHS per rompere la similarità
        print("    Tentativo 2: Rottura similarità LHS")

        # Prova a modificare un attributo LHS per volta
        for lhs_attr in lhs_list:
            original_lhs = row_data[lhs_attr]

            # Genera un valore che rompa la similarità
            safe_val = self.generate_safe_value(lhs_attr, avoid_similarity=True)

            row_data_test = row_data.copy()
            row_data_test[lhs_attr] = safe_val

            # Verifica se risolve la violazione
            temp_df = pd.concat([current_df, pd.DataFrame([row_data_test])], ignore_index=True)
            temp_matrix = self._update_distance_matrix(row_data_test, temp_df, distance_matrix.copy())

            if not self._check_violations(temp_matrix):
                print(f"      ✓ LHS riparato: {lhs_attr} {original_lhs} -> {safe_val}")
                return row_data_test
            else:
                print(f"      ✗ Modifica {lhs_attr} non risolve")

        # Strategia 3: Modifica multipla LHS
        print("    Tentativo 3: Modifica multipla LHS")
        row_data_test = row_data.copy()

        for lhs_attr in lhs_list:
            safe_val = self.generate_safe_value(lhs_attr, avoid_similarity=True)
            row_data_test[lhs_attr] = safe_val

        temp_df = pd.concat([current_df, pd.DataFrame([row_data_test])], ignore_index=True)
        temp_matrix = self._update_distance_matrix(row_data_test, temp_df, distance_matrix.copy())

        if not self._check_violations(temp_matrix):
            print(f"      ✓ Modifica multipla LHS riuscita")
            return row_data_test

        print("    ✗ Impossibile riparare la violazione")
        return None

    def identify_violated_dependencies(self, distance_matrix):
        """
        Identifica quali dipendenze sono violate nella matrice delle distanze.

        Args:
            distance_matrix: Matrice delle distanze aggiornata

        Returns:
            List di (dependency, violating_pairs) per le dipendenze violate
        """
        violated_deps = []

        for lhs, rhs in self.dependencies:
            missing_cols = [col for col in lhs + [rhs] if col not in distance_matrix.columns]
            if missing_cols:
                continue

            # Trova violazioni: LHS simili ma RHS dissimili
            lhs_similar = (distance_matrix[lhs] <= self.threshold).all(axis=1)
            rhs_dissimilar = (distance_matrix[rhs] > self.threshold)
            violations = lhs_similar & rhs_dissimilar

            if violations.any():
                violating_pairs = distance_matrix.index[violations].tolist()
                violated_deps.append(((lhs, rhs), violating_pairs))

        return violated_deps


    def _generate_single_tuple(self, i1, i2, current_df ,use_decimal=False, max_repair_attempts=5, missing_attrs=None):
        """
        Generate a single tuple following RFD-aware logic.

        Args:
            i1, i2: Base tuple indices
            current_df: Current dataset
            use_decimal: If True, use decimal values to avoid duplicates
            missing_attrs: attrs not covered by couple (i1,i2)

        Returns:
            Dictionary with new tuple values, or None if failed
        """
        for attempt in range(self.max_iter):
            row_data = {}

            print(f"    Attempt {attempt + 1}: Generating tuple based on t{i1},t{i2}")

            # Generate values for all attributes following RFD-aware logic
            for attr in self.all_attrs:
                row_data[attr] = self._get_attr_value(attr, i1, i2, use_decimal, generated_values=row_data, missing_attrs=missing_attrs)

            # Add class column
            row_data['class'] = 1

            print('Nuova tupla inserita: \n', row_data)

            # Check for duplicate tuples before proceeding with expensive operations
            if self._is_duplicate_tuple(row_data, current_df):
                print(f"    → Duplicate tuple detected with integer values")
                if not use_decimal:
                    print(f"    → Switching to decimal mode for remaining attempts")
                    use_decimal = True
                    continue  # Try again with decimal mode
                else:
                    print(f"    → Duplicate found even in decimal mode, skipping...")
                    continue  # Skip to next attempt


            # Create temporary dataframe with new tuple
            temp_df = pd.concat([current_df, pd.DataFrame([row_data])], ignore_index=True)

            # Update distance matrix and check violations
            self.original_diff_matrix = self._update_distance_matrix(row_data, temp_df, self.original_diff_matrix)

            violated_deps = self.identify_violated_dependencies(self.original_diff_matrix)
            if not violated_deps:
                print(f"    ✓ Valid tuple generated on attempt {attempt + 1}")
                self.original_diff_matrix.to_csv(self.out_diff_path)
                return row_data
            else:
                print(f"    ⚠ Violations detected, attempting repair...")

                # Trying repairing strategy
                repaired_data = row_data.copy()
                repair_successful = True

                for dep_info, violating_pairs in violated_deps:
                    print(f"      Repairing violation: {dep_info}")
                    #repaired_data = self.repair_violation(repaired_data, dep_info, current_df, self.original_diff_matrix)
                    repaired_data = self.repair_violation_with_fallback(
                        repaired_data, dep_info, current_df, self.original_diff_matrix
                    )

                    if repaired_data is None:
                        repair_successful = False
                        break

                if repair_successful and repaired_data is not None:
                    # Verify that the repairing was successful
                    temp_df_repaired = pd.concat([current_df, pd.DataFrame([repaired_data])], ignore_index=True)
                    final_matrix = self._update_distance_matrix(repaired_data, temp_df_repaired,
                                                                self.original_diff_matrix.copy())

                    if not self._check_violations(final_matrix):
                        print(f"    ✓ Tuple successfully repaired on attempt {attempt + 1}")
                        self.original_diff_matrix = final_matrix
                        self.original_diff_matrix.to_csv(self.out_diff_path)
                        return repaired_data
                    else:
                        print(f"    ✗ Repair failed, still has violations")
                else:
                    print(f"    ✗ Could not repair violations on attempt {attempt + 1}")

        print(f"    Failed to generate valid tuple after {self.max_iter} attempts")
        return None


    def get_pairs_for_dependency(self, lhs_list, rhs_attr):
        """
        Trova le coppie di tuple che hanno valori <= threshold per tutti gli attributi
        coinvolti in una specifica dipendenza (LHS + RHS).

        Args:
            lhs_list: Lista degli attributi LHS della dipendenza
            rhs_attr: Attributo RHS della dipendenza

        Returns:
            DataFrame con le coppie valide per questa dipendenza
        """
        # Tutti gli attributi coinvolti nella dipendenza
        all_dependency_attrs = lhs_list + [rhs_attr]

        # Filtra per gli attributi di questa dipendenza
        relevant_df = self.attrs_df[self.attrs_df['attribute'].isin(all_dependency_attrs)]

        if relevant_df.empty:
            print(f"Nessuna coppia trovata per la dipendenza {lhs_list} -> {rhs_attr}")
            return pd.DataFrame()

        # Conta quanti attributi della dipendenza ogni coppia di tuple copre
        freq_df = (relevant_df
                   .groupby(['idx1', 'idx2'])
                   .agg(attribute_count=('attribute', 'nunique'))
                   .reset_index())

        # Mantieni solo le coppie che coprono TUTTI gli attributi della dipendenza
        required_attrs = len(all_dependency_attrs)
        valid_pairs = freq_df[freq_df['attribute_count'] == required_attrs]

        print(f"Dipendenza {lhs_list} -> {rhs_attr}: {len(valid_pairs)} coppie valide")
        return valid_pairs

    def sort_dependencies_by_complexity(self):
        """
        Ordina le dipendenze per complessità (numero di attributi coinvolti) in ordine decrescente.
        Le dipendenze con più attributi vengono prioritizzate.

        Returns:
            Lista di dipendenze ordinate per complessità
        """
        dependency_complexity = []

        for lhs_list, rhs_attr in self.dependencies:
            total_attrs = len(lhs_list) + 1  # +1 per RHS
            dependency_complexity.append((total_attrs, lhs_list, rhs_attr))

        # Ordina per numero di attributi (decrescente)
        dependency_complexity.sort(key=lambda x: x[0], reverse=True)

        sorted_deps = [(lhs, rhs) for _, lhs, rhs in dependency_complexity]

        print("Dipendenze ordinate per complessità:")
        for i, (lhs, rhs) in enumerate(sorted_deps):
            total_attrs = len(lhs) + 1
            print(f"  {i + 1}. {lhs} -> {rhs} ({total_attrs} attributi)")

        return sorted_deps

    def generate_tuple_for_dependency(self, i1, i2, lhs_list, rhs_attr, current_df, use_decimal=False):
        """
        Genera una nuova tupla partendo da una tupla base e modificando solo gli attributi
        coinvolti in una specifica dipendenza.

        Args:
            i1, i2: Indici delle tuple base
            lhs_list: Attributi LHS della dipendenza
            rhs_attr: Attributo RHS della dipendenza
            current_df: Dataset corrente
            use_decimal: Se usare valori decimali

        Returns:
            Dictionary con i valori della nuova tupla o None se fallisce
        """
        # Attributi coinvolti nella dipendenza
        dependency_attrs = set(lhs_list + [rhs_attr])

        for attempt in range(self.max_iter):
            print(f"    Tentativo {attempt + 1}: Generazione per dipendenza {lhs_list} -> {rhs_attr}")

            # Parti dalla prima tupla come base
            base_tuple = self.imbalance_df_min.iloc[i1].copy()
            row_data = base_tuple.to_dict()

            # Rimuovi tuple_id se presente
            if 'tuple_id' in row_data:
                del row_data['tuple_id']

            # Modifica solo gli attributi coinvolti nella dipendenza
            generated_values = {}
            for attr in dependency_attrs:
                if attr in self.all_attrs:  # Assicurati che l'attributo esista
                    new_val = self._get_attr_value(attr, i1, i2, use_decimal, generated_values=generated_values)
                    row_data[attr] = new_val
                    generated_values[attr] = new_val
                    print(f"      {attr}: {base_tuple[attr]} -> {new_val}")

            # Mantieni la classe come 1
            row_data['class'] = 1

            print(f'Nuova tupla (dipendenza {lhs_list} -> {rhs_attr}): modificati {len(dependency_attrs)} attributi')

            # Verifica duplicati
            if self._is_duplicate_tuple(row_data, current_df):
                print(f"    → Tupla duplicata rilevata")
                if not use_decimal:
                    print(f"    → Passaggio a modalità decimale")
                    use_decimal = True
                    continue
                else:
                    print(f"    → Duplicato anche in modalità decimale, riprova...")
                    continue

            # Crea dataframe temporaneo e verifica violazioni
            temp_df = pd.concat([current_df, pd.DataFrame([row_data])], ignore_index=True)
            temp_distance_matrix = self._update_distance_matrix(row_data, temp_df, self.original_diff_matrix.copy())

            violated_deps = self.identify_violated_dependencies(temp_distance_matrix)

            if not violated_deps:
                print(f"    ✓ Tupla valida generata al tentativo {attempt + 1}")
                self.original_diff_matrix = temp_distance_matrix
                self.original_diff_matrix.to_csv(self.out_diff_path)
                return row_data
            else:
                print(f"    ⚠ Violazioni rilevate, tentativo di riparazione...")

                # Prova a riparare
                repaired_data = row_data.copy()
                repair_successful = True

                for dep_info, violating_pairs in violated_deps:
                    repaired_data = self.repair_violation(repaired_data, dep_info, current_df, temp_distance_matrix)
                    if repaired_data is None:
                        repair_successful = False
                        break

                if repair_successful and repaired_data is not None:
                    # Verifica riparazione
                    temp_df_repaired = pd.concat([current_df, pd.DataFrame([repaired_data])], ignore_index=True)
                    final_matrix = self._update_distance_matrix(repaired_data, temp_df_repaired,
                                                                self.original_diff_matrix.copy())

                    if not self._check_violations(final_matrix):
                        print(f"    ✓ Tupla riparata con successo al tentativo {attempt + 1}")
                        self.original_diff_matrix = final_matrix
                        self.original_diff_matrix.to_csv(self.out_diff_path)
                        return repaired_data
                    else:
                        print(f"    ✗ Riparazione fallita")
                else:
                    print(f"    ✗ Impossibile riparare le violazioni")

        print(f"    Impossibile generare tupla valida dopo {self.max_iter} tentativi")
        return None

    def augment_dataset_by_dependency(self):
        """
        Metodo di augmentazione alternativo che lavora dipendenza per dipendenza.

        Returns:
            Dataset aumentato
        """
        oversampling_quantity = self.oversampling_quantity
        print(f"Generazione di {oversampling_quantity} nuovi campioni (metodo per dipendenza)")

        # Ordina le dipendenze per complessità
        sorted_dependencies = self.sort_dependencies_by_complexity()
        print('SORTED DEPENDENCIES:\n', sorted_dependencies)

        current_df = self.imbalance_df_min.copy()
        new_rows = []
        generated_count = 0

        # Itera sulle dipendenze in ordine di complessità
        for dep_idx, (lhs_list, rhs_attr) in enumerate(sorted_dependencies):
            if generated_count >= oversampling_quantity:
                break

            print(f"\n=== Dipendenza {dep_idx + 1}/{len(sorted_dependencies)}: {lhs_list} -> {rhs_attr} ===")

            # Trova le coppie valide per questa dipendenza
            valid_pairs = self.get_pairs_for_dependency(lhs_list, rhs_attr)

            if valid_pairs.empty:
                print(f"Nessuna coppia valida per questa dipendenza, continua...")
                continue

            # Calcola quante tuple generare per questa dipendenza
            remaining_needed = oversampling_quantity - generated_count
            pairs_count = len(valid_pairs)

            # Distribuisci equamente tra le dipendenze rimanenti
            remaining_deps = len(sorted_dependencies) - dep_idx
            target_for_this_dep = min(remaining_needed // remaining_deps + 1, pairs_count * 2)

            print(f"Target per questa dipendenza: {target_for_this_dep}")

            # Genera tuple per ogni coppia valida
            tuples_generated_this_dep = 0
            for _, pair in valid_pairs.iterrows():
                if generated_count >= oversampling_quantity or tuples_generated_this_dep >= target_for_this_dep:
                    break

                i1, i2 = pair['idx1'], pair['idx2']
                print(f"\nGenerazione per coppia ({i1}, {i2})")

                new_tuple = self.generate_tuple_for_dependency(i1, i2, lhs_list, rhs_attr, current_df)

                if new_tuple is not None:
                    new_rows.append(new_tuple)
                    current_df = pd.concat([current_df, pd.DataFrame([new_tuple])], ignore_index=True)
                    generated_count += 1
                    tuples_generated_this_dep += 1
                    print(f"    Generato campione {generated_count}/{oversampling_quantity}")

            print(f"Dipendenza completata: {tuples_generated_this_dep} tuple generate")

        # Crea il dataset finale
        if new_rows:
            new_df = pd.DataFrame(new_rows, columns=self.all_attrs + ['class'])

            # Taglia alla quantità esatta richiesta
            if len(new_df) > oversampling_quantity:
                new_df = new_df.sample(n=oversampling_quantity, random_state=42).reset_index(drop=True)

            new_df.to_csv(f'augmentation_results/{self.base}_new_tuples_by_dependency.csv', index=False)

            # Combina con il dataset originale
            augmented_df = pd.concat([self.imbalance_df_min, new_df], ignore_index=True)
            print(f"\nGenerazione completata: {len(new_df)} nuovi campioni")
            print(f"Forma dataset finale: {augmented_df.shape}")

            return new_df
        else:
            print("Nessuna tupla valida generata!")
            return self.imbalance_df



    def augment_dataset(self):
        """
        Main augmentation method.

        Returns:
            Augmented dataset
        """

        # Calculate required oversampling
        #minority_count = len(self.imbalance_df[self.imbalance_df['class'] == 1])
        #majority_count = len(self.imbalance_df[self.imbalance_df['class'] == 0])
        #oversampling_quantity = majority_count - minority_count
        oversampling_quantity=self.oversampling_quantity

        print(f"Need to generate {oversampling_quantity} new samples")
        top_pairs = self._get_top_pairs()
        if len(top_pairs) == 0:
            print("No complete top pairs found, using ALL available pairs from attrs_df")
            # Estrai tutte le coppie uniche da attrs_df
            all_pairs = self.attrs_df[['idx1', 'idx2']].drop_duplicates().reset_index(drop=True)
            print(f"Found {len(all_pairs)} total pairs in attrs_df")

            if len(all_pairs) == 0:
                print("No top-pairs available at all, using dependency method!")
                return self.augment_dataset_by_dependency()

            top_pairs = all_pairs
        else:
            print(f"Found {len(top_pairs)} suitable top-pairs")

        #top_pairs = self._get_top_pairs_flexible(min_coverage_ratio=0.7)

        '''
        # Get tuple pairs where all the values for both LHS and RHS attributes are <= thr or where all the LHS attributes are <= thr
        if len(top_pairs)==0:
            print("No suitable tuple pairs found, generating tuples by dependency!")
            self.augment_dataset_by_dependency()
        '''

        #print(f"Found {len(top_pairs)} suitable tuple pairs")
        # Calculate oversampling factor
        oversampling_factor = max(1, (oversampling_quantity // len(top_pairs)) + 1)
        print(f"Oversampling factor: {oversampling_factor}")

        # Start with original dataset which is the minority dataset
        current_df = self.imbalance_df_min.copy()
        new_rows = []
        generated_count = 0

        # Generate new tuples
        for _, pair in top_pairs.iterrows():
            if generated_count >= oversampling_quantity:
                break

            i1, i2 = pair['idx1'], pair['idx2']
            print(f"\nGenerating tuples for pair ({i1}, {i2})")

            # Identifica attributi mancanti per questa coppia
            covered_attrs = set(self.attrs_df[
                                    (self.attrs_df['idx1'] == i1) & (self.attrs_df['idx2'] == i2)
                                    ]['attribute'].unique())

            #missing_attrs = self.both_attrs - covered_attrs
            missing_attrs = set(self.all_attrs) - covered_attrs

            if missing_attrs:
                print(f"  Pair ({i1},{i2}) missing attrs: {missing_attrs} -> using sys.max")

            for iteration in range(oversampling_factor):
                if generated_count >= oversampling_quantity:
                    break

                print(f"  Iteration {iteration + 1}/{oversampling_factor}")
                #new_tuple = self._generate_single_tuple(i1, i2, current_df)
                new_tuple = self._generate_single_tuple(i1, i2, current_df,missing_attrs=missing_attrs)

                if new_tuple is not None:
                    new_rows.append(new_tuple)
                    # Add to current dataset for next distance calculations
                    current_df = pd.concat([current_df, pd.DataFrame([new_tuple])],
                                           ignore_index=True)
                    generated_count += 1
                    print(f"    Generated tuple {generated_count}/{oversampling_quantity}")

        # Create final augmented dataset
        if new_rows:
            new_df = pd.DataFrame(new_rows, columns=self.all_attrs + ['class'])

            # Trim to exact required quantity
            if len(new_df) > oversampling_quantity:
                new_df = new_df.sample(n=oversampling_quantity, random_state=42).reset_index(drop=True)
            #basename_min = os.path.basename(updated_df_path).split('.')[0]
            sorted_cols = sorted(new_df.columns,
                                 key=lambda c: (0, int(re.search(r'\d+', c).group())) if re.search(r'Attr(\d+)$', c)
                                 else (2, 0) if c == 'class' else (1, c))
            new_df = new_df[sorted_cols]
            new_df.to_csv(f'augmentation_results/{self.base}_new_tuples_{self.threshold}.csv', index=False)
            # Combine with original dataset
            augmented_df = pd.concat([self.imbalance_df_min, new_df], ignore_index=True)

            print(f"\nSuccessfully generated {len(new_df)} new samples")
            print(f"Final dataset shape: {augmented_df.shape}")

            return new_df
        else:
            print("No valid tuples could be generated!")
            return self.imbalance_df
