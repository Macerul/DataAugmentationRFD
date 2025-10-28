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
        self.last_safe_value = sys.maxsize - self.threshold - 1     # sys.max (sys.max - thr - 1)

        # Load datasets
        self.imbalance_dataset_path = imbalance_dataset_path # path to the imbalance datasets
        self.base = os.path.basename(self.imbalance_dataset_path).split('.')[0]
        self.imbalance_df = pd.read_csv(imbalance_dataset_path)
        ###### IDENTIFY IMB CLASS ######
        counts = self.imbalance_df['class'].value_counts()
        self.min_class = counts.idxmin()
        self.dataset_min = self.imbalance_df[self.imbalance_df['class']== self.min_class]
        self.out_min_path = os.path.join(self.imbalance_dir, f'{self.base}_min.csv')
        self.dataset_min.to_csv(self.out_min_path, index=False)

        self.imbalance_df_min =  pd.read_csv(self.out_min_path)
        print(f'Reading imbalance dataset...:\n{self.imbalance_df_min.head()}')

        self.integer_attrs = set()
        for attr in self.imbalance_df_min.columns:
            if (self.imbalance_df_min[attr].dtype in ['int64', 'int32'] or
                    np.issubdtype(self.imbalance_df_min[attr].dtype, np.integer)):
                self.integer_attrs.add(attr)
                print(f"{attr} is int")


        self.out_diff_path = os.path.join(self.output_diff_dir, f'pw_diff_mx_{self.base}_min.csv')
        #self.attrs_df = pd.read_csv(attr_diff_path)
        print("Computing difference matrix...")
        self.diff_df = self._compute_diff_matrix()

        self.original_diff_matrix = pd.read_csv(self.out_diff_path, index_col='tuple_pair')
        print("Initial difference matrix:\n", self.original_diff_matrix)

        print("Filtering difference pairs...")
        self.attrs_df = self._filter_diff_pairs()

        self.dependencies = self._parse_rfds(rfd_file_path)
        self._analyze_attributes()


    def order_attributes(self):
        attrs = self.dataset_min.columns[:-1]
        print('Attributes: \n', attrs)

        ordered = sorted(
            attrs,
            key=lambda x: (
                1 if self.check_bool_attr(x) else 0,
                int(x.replace('Attr', ''))
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
        dependencies = []

        with open(rfd_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '->' in line:
                    lhs, rhs = line.split('->')
                    lhs_list = [a.split('@')[0] for a in re.split('[, ]+', lhs) if a and '@' in a]
                    rhs_attr = re.match(r"(\w+)@", rhs.strip()).group(1)
                    dependencies.append((lhs_list, rhs_attr))

        if self.selected_rfds is not None:
            filtered_deps = []
            for lhs, rhs in dependencies:
                dep_str = f"{','.join(lhs)} -> {rhs}"
                if dep_str in self.selected_rfds:
                    filtered_deps.append((lhs, rhs))
            dependencies = filtered_deps
        print('RFDcs:\n', dependencies)
        return dependencies

    def _compute_diff_matrix(self) -> pd.DataFrame:
        self.imbalance_df_min.reset_index(inplace=True)
        self.imbalance_df_min.rename(columns={'index': 'tuple_id'}, inplace=True)
        attribute_columns = [col for col in self.imbalance_df_min.columns if col not in ['tuple_id']]
        diff_list = []
        for (_, row_i), (_, row_j) in combinations(self.imbalance_df_min.iterrows(), 2):
            diff_values = np.abs(row_i[attribute_columns] - row_j[attribute_columns])

            diff_entry = {
                'tuple_pair': f"t{int(row_i['tuple_id'])},t{int(row_j['tuple_id'])}"
            }
            for col in attribute_columns:
                diff_entry[f"{col}"] = diff_values[col]

            diff_list.append(diff_entry)

        diff_df = pd.DataFrame(diff_list)

        diff_df.to_csv(self.out_diff_path, index=False)
        print(f"Saved initial difference matrix to {self.out_diff_path}")
        return diff_df

    # IVD
    def _filter_diff_pairs(self) -> pd.DataFrame:
        records = []
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
        lhs_attrs = set()
        rhs_attrs = set()

        for lhs_list, rhs_attr in self.dependencies:
            lhs_attrs.update(lhs_list)
            rhs_attrs.add(rhs_attr)

        self.lhs_attrs = lhs_attrs
        self.rhs_attrs = rhs_attrs
        self.both_attrs = lhs_attrs & rhs_attrs
        self.all_attrs = self.order_attributes()
        self.dependency_attrs = lhs_attrs | rhs_attrs
        self.no_dependency_attrs = set(self.all_attrs) - self.dependency_attrs


        print(f"Analyzing {len(self.dependencies)} RFDs")
        print(f"All attributes: {self.all_attrs}")
        print(f"Attributes in both LHS and RHS: {sorted(self.both_attrs)}")
        print(f"Attributes in LHS: {sorted(self.lhs_attrs)}")
        print(f"Attributes not in any dependency: {sorted(self.no_dependency_attrs)}")

    def _get_top_pairs(self):

        if len( self.both_attrs) != 0:
            total_both_attrs = len(self.both_attrs)
            relevant_df = self.attrs_df[self.attrs_df['attribute'].isin(self.both_attrs)]
            #print('Top pairs:\n',relevant_df.head())
        else:
            print("No attribute is present in both LHS and RHS, considering only LHS attributes...")
            total_both_attrs = len(self.lhs_attrs)
            relevant_df = self.attrs_df[self.attrs_df['attribute'].isin(self.lhs_attrs)]

        freq_df = (relevant_df
                   .groupby(['idx1', 'idx2'])
                   .agg(attribute_count=('attribute', 'nunique'))
                   .reset_index())

        freq_df['covers_all_both'] = freq_df['attribute_count'] == total_both_attrs

        return freq_df[freq_df['covers_all_both']]

    def _get_attr_value(self, attr, i1, i2, use_decimal=False, generated_values=None, missing_attrs=None):

        if missing_attrs and attr in missing_attrs:
            print(f"  {attr}: Missing, using sys.max strategy")
            return self._get_safe_fallback_value(attr, use_decimal)

        row = self.attrs_df[(self.attrs_df['attribute'] == attr) &
                            ((self.attrs_df['idx1'] == i1) & (self.attrs_df['idx2'] == i2))]
        if not row.empty:

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
            # range-based logic
            if diff <= self.threshold:
                min_val = min(val1, val2)
                max_val = max(val1, val2)

                if diff == 0:
                    print(f"Identical values, using {val1}")
                    return val1
                if diff == 1:
                    try:
                        if generated_values and len(generated_values) > 0:
                            keys = [k for k in generated_values.keys() if k != attr and k in self.all_attrs]
                            if len(keys) > 0:
                                vec_generated = np.array([float(generated_values[k]) for k in keys])
                                vec_i1 = np.array([float(self.imbalance_df_min.at[i1, k]) for k in keys])
                                vec_i2 = np.array([float(self.imbalance_df_min.at[i2, k]) for k in keys])
                                d1 = np.linalg.norm(vec_generated - vec_i1)
                                d2 = np.linalg.norm(vec_generated - vec_i2)
                                chosen = val1 if d1 <= d2 else val2
                                print(f"diff==1: euclidean d1={d1:.2f}, d2={d2:.2f}, choosing: {chosen}")
                                return chosen
                    except Exception as e:
                        print(f"Warning computing euclidean distance: {e}")

                    chosen = random.randrange(min_val, max_val+1)
                    print(f"diff==1 no generated_values yet or error: {chosen}")
                    return chosen
                else:
                    if use_decimal:
                        generated_val = random.uniform(min_val, max_val)
                        print(f"Similar values, generating decimal: {generated_val:.2f}")
                        return round(generated_val, 2)
                    else:
                        if attr in self.integer_attrs:
                            generated_val = random.randint(int(min_val), int(max_val))
                            print(f"Integer attribute {attr}, generating integer in range: {generated_val}")
                        else:
                            generated_val = round(random.uniform(min_val, max_val), 2)
                            print(f"Float/decimal attribute {attr}, generating decimal: {generated_val}")
                        return generated_val
            else:
                print(f"Dissimilar values (diff={diff} > {self.threshold}), using max")
                return self._get_safe_fallback_value(attr, use_decimal)
        else:
            print(f"  {attr}: No tuple pair data, using max")
            return self._get_safe_fallback_value(attr, use_decimal)


    def _get_safe_fallback_value(self, attr, use_decimal=False):

        overall_max = self.last_safe_value - self.threshold - 1  # max strategy
        print('Overall max:', overall_max)
        self.last_safe_value = overall_max
        safe_base = overall_max

        if use_decimal:
            decimal_component = round(random.random(), 2)
            safe_value = safe_base + decimal_component
            print(f"Safe max with decimal: {safe_value:.2f}")
            return safe_value
        else:
            print(f"Safe max: {safe_base}")
            return safe_base

    def _update_distance_matrix(self, new_tuple_values, current_df, prev_matrix=None):

        if prev_matrix is None or prev_matrix.empty:
            updated = pd.DataFrame()
        else:
            updated = prev_matrix

        new_idx = len(current_df) - 1

        for attr in self.all_attrs:
            if attr not in updated.columns:
                updated[attr] = []

        for existing_idx in range(new_idx):
            pair_name = f"t{int(existing_idx)},t{int(new_idx)}"
            distances = {}
            for attr in self.all_attrs:
                new_val = new_tuple_values.get(attr, 0)
                existing_val = current_df.iloc[existing_idx][attr]
                distances[attr] = abs(new_val - existing_val)

            updated.loc[pair_name] = distances
        #print('Updated matrix:\n',updated)
        return updated

    def _check_violations(self, distance_matrix):

        violations_found = False
        for lhs, rhs in self.dependencies:
            missing_cols = [col for col in lhs + [rhs] if col not in distance_matrix.columns]
            if missing_cols:
                print(f"Missing columns {missing_cols} for dependency {lhs} -> {rhs}")
                continue

            lhs_similar = (distance_matrix[lhs] <= self.threshold).all(axis=1)
            rhs_dissimilar = (distance_matrix[rhs] > self.threshold)
            violations = lhs_similar & rhs_dissimilar

            if violations.any():
                violation_pairs = distance_matrix.index[violations].tolist()
                print(f"Violation in dependency {lhs} -> {rhs}")
                print(f"Violating pairs: {violation_pairs}")
                violations_found = True
                for pair in violation_pairs[:3]:
                    lhs_diffs = [distance_matrix.loc[pair, attr] for attr in lhs]
                    rhs_diff = distance_matrix.loc[pair, rhs]
                    print(
                        f"{pair}: LHS diffs {lhs_diffs} ≤ {self.threshold}, RHS diff {rhs_diff} > {self.threshold}")

        return violations_found

    def _is_duplicate_tuple(self, new_tuple_data, current_df):
        comparison_attrs = {attr: new_tuple_data[attr] for attr in self.all_attrs}
        for idx, existing_row in current_df.iterrows():
            is_duplicate = True
            for attr in self.all_attrs:
                if abs(existing_row[attr] - comparison_attrs[attr]) > 1e-10:
                    is_duplicate = False
                    break
            if is_duplicate:
                print(f"Duplicate found with existing tuple at index {idx}")
                return True

        return False

    # Safe intervals
    def get_safe_value_ranges(self, attr):
        existing_values = sorted(self.imbalance_df_min[attr].unique())

        safe_ranges = []

        for i in range(len(existing_values)):
            current_val = existing_values[i]

            if i == 0:
                # Before first value
                range_start = current_val - self.threshold - 5
                range_end = current_val - self.threshold - 1
                if range_start < range_end:
                    safe_ranges.append((range_start, range_end))

            # After last value
            if i == len(existing_values) - 1:
                # Dopo l'ultimo valore
                range_start = current_val + self.threshold + 1
                range_end = current_val + self.threshold + 10  # buffer extra
                safe_ranges.append((range_start, range_end))
            else:
                # Intermediate intervals
                next_val = existing_values[i + 1]
                range_start = current_val + self.threshold + 1
                range_end = next_val - self.threshold - 1

                if range_start < range_end:
                    safe_ranges.append((range_start, range_end))

        print(f"  {attr} safe interval: {safe_ranges}")
        return safe_ranges

    def generate_safe_value(self, attr, avoid_similarity=True):

        if avoid_similarity:
            safe_ranges = self.get_safe_value_ranges(attr)

            if safe_ranges:

                chosen_range = random.choice(safe_ranges)
                min_val, max_val = chosen_range

                if isinstance(min_val, (int, np.integer)) and isinstance(max_val, (int, np.integer)):
                    safe_val = random.randint(int(min_val), int(max_val))
                else:
                    safe_val = round(random.uniform(min_val, max_val), 2)

                print(f"Generated safe value for {attr}: {safe_val} in range {chosen_range}")
                return safe_val

        # max
        return self._get_safe_fallback_value(attr, use_decimal=True)

    def repair_violation_with_fallback(self, row_data, violated_dependency, current_df, distance_matrix):
        lhs_list, rhs_attr = violated_dependency

        repaired = self.repair_violation(row_data, violated_dependency, current_df, distance_matrix)

        if repaired is not None:
            return repaired

        print(" Max for LHS and RHS")
        row_data_test = row_data.copy()

        all_dep_attrs = lhs_list + [rhs_attr]

        for attr in all_dep_attrs:
            old_val = row_data_test[attr]
            safe_val = self._get_safe_fallback_value(attr, use_decimal=True)
            row_data_test[attr] = safe_val
            print(f"{attr}: {old_val} -> {safe_val} (max)")

        temp_df = pd.concat([current_df, pd.DataFrame([row_data_test])], ignore_index=True)
        temp_matrix = self._update_distance_matrix(row_data_test, temp_df, distance_matrix.copy())

        if not self._check_violations(temp_matrix):
            print(f"Successful repair")
            return row_data_test

        print("Repair failed")
        return None


    def repair_violation(self, row_data, violated_dependency, current_df, distance_matrix):

        lhs_list, rhs_attr = violated_dependency
        print(f"Repairing: {lhs_list} -> {rhs_attr}")
        original_rhs = row_data[rhs_attr]

        last_pair = distance_matrix.index[-1]
        existing_idx = int(last_pair.split(',')[0][1:])

        target_rhs = current_df.iloc[existing_idx][rhs_attr]

        for variation in [0, 0.1, -0.1, 0.5, -0.5, 1, -1]:
            test_rhs = target_rhs + variation
            row_data_test = row_data.copy()
            row_data_test[rhs_attr] = test_rhs

            temp_df = pd.concat([current_df, pd.DataFrame([row_data_test])], ignore_index=True)
            temp_matrix = self._update_distance_matrix(row_data_test, temp_df, distance_matrix.copy())

            if not self._check_violations(temp_matrix):
                #print(f"RHS repaired: {original_rhs} -> {test_rhs}")
                return row_data_test

        for lhs_attr in lhs_list:
            original_lhs = row_data[lhs_attr]

            safe_val = self.generate_safe_value(lhs_attr, avoid_similarity=True)

            row_data_test = row_data.copy()
            row_data_test[lhs_attr] = safe_val

            temp_df = pd.concat([current_df, pd.DataFrame([row_data_test])], ignore_index=True)
            temp_matrix = self._update_distance_matrix(row_data_test, temp_df, distance_matrix.copy())

            if not self._check_violations(temp_matrix):
                print(f"Successful LHS repair: {lhs_attr} {original_lhs} -> {safe_val}")
                return row_data_test
            else:
                print(f"Repairing LHS failed {lhs_attr}")

        row_data_test = row_data.copy()

        for lhs_attr in lhs_list:
            safe_val = self.generate_safe_value(lhs_attr, avoid_similarity=True)
            row_data_test[lhs_attr] = safe_val

        temp_df = pd.concat([current_df, pd.DataFrame([row_data_test])], ignore_index=True)
        temp_matrix = self._update_distance_matrix(row_data_test, temp_df, distance_matrix.copy())

        if not self._check_violations(temp_matrix):
            return row_data_test
        return None

    def identify_violated_dependencies(self, distance_matrix):

        violated_deps = []

        for lhs, rhs in self.dependencies:
            missing_cols = [col for col in lhs + [rhs] if col not in distance_matrix.columns]
            if missing_cols:
                continue

            lhs_similar = (distance_matrix[lhs] <= self.threshold).all(axis=1)
            rhs_dissimilar = (distance_matrix[rhs] > self.threshold)
            violations = lhs_similar & rhs_dissimilar

            if violations.any():
                violating_pairs = distance_matrix.index[violations].tolist()
                violated_deps.append(((lhs, rhs), violating_pairs))

        return violated_deps


    def _generate_single_tuple(self, i1, i2, current_df ,use_decimal=False, max_repair_attempts=5, missing_attrs=None):

        for attempt in range(self.max_iter):
            row_data = {}

            print(f"Attempt {attempt + 1}: Generating tuple based on t{i1},t{i2}")

            # Generate values for all attributes following RFD-aware logic
            for attr in self.all_attrs:
                row_data[attr] = self._get_attr_value(attr, i1, i2, use_decimal, generated_values=row_data, missing_attrs=missing_attrs)

            row_data['class'] = self.min_class

            print('New tuple added: \n', row_data)

            if self._is_duplicate_tuple(row_data, current_df):
                print(f"Duplicate tuple detected with integer values")
                if not use_decimal:
                    print(f"Switching to decimal")
                    use_decimal = True
                    continue
                else:
                    print(f"Duplicate found, skipping...")
                    continue  # skip to next attempt

            temp_df = pd.concat([current_df, pd.DataFrame([row_data])], ignore_index=True)
            self.original_diff_matrix = self._update_distance_matrix(row_data, temp_df, self.original_diff_matrix)

            violated_deps = self.identify_violated_dependencies(self.original_diff_matrix)
            if not violated_deps:
                print(f"Valid tuple generated on attempt {attempt + 1}")
                self.original_diff_matrix.to_csv(self.out_diff_path)
                return row_data
            else:
                print(f"Violations detected, attempting repair...")

                repaired_data = row_data.copy()
                repair_successful = True

                for dep_info, violating_pairs in violated_deps:
                    print(f"Repairing violation: {dep_info}")
                    repaired_data = self.repair_violation_with_fallback(
                        repaired_data, dep_info, current_df, self.original_diff_matrix
                    )

                    if repaired_data is None:
                        repair_successful = False
                        break

                if repair_successful and repaired_data is not None:
                    temp_df_repaired = pd.concat([current_df, pd.DataFrame([repaired_data])], ignore_index=True)
                    final_matrix = self._update_distance_matrix(repaired_data, temp_df_repaired,
                                                                self.original_diff_matrix.copy())

                    if not self._check_violations(final_matrix):
                        print(f"Tuple successfully repaired on attempt {attempt + 1}")
                        self.original_diff_matrix = final_matrix
                        self.original_diff_matrix.to_csv(self.out_diff_path)
                        return repaired_data
                    else:
                        print(f"Repair failed, still has violations")
                else:
                    print(f"Could not repair violations on attempt {attempt + 1}")

        print(f"Failed to generate valid tuple after {self.max_iter} attempts")
        return None

    def augment_dataset(self):

        oversampling_quantity=self.oversampling_quantity

        print(f"Need to generate {oversampling_quantity} new samples")
        top_pairs = self._get_top_pairs()
        if len(top_pairs) == 0:
            print("No complete top pairs found, using ALL available pairs from IVD")
            all_pairs = self.attrs_df[['idx1', 'idx2']].drop_duplicates().reset_index(drop=True)
            print(f"Found {len(all_pairs)} total pairs in IVD")

            top_pairs = all_pairs
        else:
            print(f"Found {len(top_pairs)} suitable top-pairs")

        oversampling_factor = max(1, (oversampling_quantity // len(top_pairs)) + 1)
        print(f"Oversampling factor: {oversampling_factor}")

        current_df = self.imbalance_df_min.copy()
        new_rows = []
        generated_count = 0

        for _, pair in top_pairs.iterrows():
            if generated_count >= oversampling_quantity:
                break

            i1, i2 = pair['idx1'], pair['idx2']
            print(f"\nGenerating tuples for pair ({i1}, {i2})")

            covered_attrs = set(self.attrs_df[
                                    (self.attrs_df['idx1'] == i1) & (self.attrs_df['idx2'] == i2)
                                    ]['attribute'].unique())

            missing_attrs = set(self.all_attrs) - covered_attrs

            if missing_attrs:
                print(f"Pair ({i1},{i2}) missing attrs: {missing_attrs} -> using sys.max")

            for iteration in range(oversampling_factor):
                if generated_count >= oversampling_quantity:
                    break

                print(f"Iteration {iteration + 1}/{oversampling_factor}")
                new_tuple = self._generate_single_tuple(i1, i2, current_df,missing_attrs=missing_attrs)

                if new_tuple is not None:
                    new_rows.append(new_tuple)

                    current_df = pd.concat([current_df, pd.DataFrame([new_tuple])],
                                           ignore_index=True)
                    generated_count += 1
                    print(f"Generated tuple {generated_count}/{oversampling_quantity}")

        if new_rows:
            new_df = pd.DataFrame(new_rows, columns=self.all_attrs + ['class'])

            if len(new_df) > oversampling_quantity:
                new_df = new_df.sample(n=oversampling_quantity, random_state=42).reset_index(drop=True)
            sorted_cols = sorted(new_df.columns,
                                 key=lambda c: (0, int(re.search(r'\d+', c).group())) if re.search(r'Attr(\d+)$', c)
                                 else (2, 0) if c == 'class' else (1, c))
            new_df = new_df[sorted_cols]
            new_df.to_csv(f'augmentation_results/{self.base}_new_tuples_{self.threshold}.csv', index=False)
            augmented_df = pd.concat([self.imbalance_df_min, new_df], ignore_index=True)

            print(f"\nSuccessfully generated {len(new_df)} new samples")
            print(f"Final dataset shape: {augmented_df.shape}")

            return new_df
        else:
            print("No valid tuples could be generated!")
            return self.imbalance_df
