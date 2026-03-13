import re
import random
import os
import sys
import time
import tracemalloc
import numpy as np
import pandas as pd
from itertools import combinations


class SyRFD:
    def __init__(self, imbalance_dataset_path,
                 rfd_file_path, oversampling,
                 threshold=4, max_iter=100, selected_rfds=None):

        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self.total_start_time = time.time()

        # Create output directory
        self.output_dir = 'augmentation_results'
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_diff_dir = 'diff_matrices'
        os.makedirs(self.output_diff_dir, exist_ok=True)
        self.output_diff_tuples_dir = 'diff_tuples'
        os.makedirs(self.output_diff_tuples_dir, exist_ok=True)
        self.imbalance_dir = 'imbalanced_datasets'

        # Initialize parameters
        self.threshold = threshold
        self.max_iter = max_iter
        self.oversampling_quantity = oversampling
        self.selected_rfds = selected_rfds
        self.last_safe_value = sys.maxsize - self.threshold - 1
        self.repair_counter = 0

        # Load datasets
        self.imbalance_dataset_path = imbalance_dataset_path
        self.base = os.path.basename(self.imbalance_dataset_path).split('.')[0]
        self.imbalance_df = pd.read_csv(imbalance_dataset_path)

        ###### IDENTIFY IMB CLASS ######
        counts = self.imbalance_df['class'].value_counts()
        self.min_class = counts.idxmin()
        self.dataset_min = self.imbalance_df[self.imbalance_df['class'] == self.min_class].copy()

        self.out_min_path = os.path.join(self.imbalance_dir, f'{self.base}_min.csv')
        self.dataset_min.to_csv(self.out_min_path, index=False)

        self.imbalance_df_min = pd.read_csv(self.out_min_path)
        print(f'Reading imbalance dataset...:\n{self.imbalance_df_min.head()}')

        # Pre-calculate integer attributes for fast lookup
        self.integer_attrs = set()
        for attr in self.imbalance_df_min.columns:
            if (self.imbalance_df_min[attr].dtype in ['int64', 'int32'] or
                    np.issubdtype(self.imbalance_df_min[attr].dtype, np.integer)):
                self.integer_attrs.add(attr)
                print(f"{attr} is int")

        # Keep a numpy version of the data for fast vector operations
        # We assume the 'class' column is the last one or we handle it specifically.
        # The logic mostly operates on attributes, so we separate them.
        self.attribute_cols = [c for c in self.imbalance_df_min.columns if c != 'class' and c != 'tuple_id']
        self.current_data_numpy = self.imbalance_df_min[self.attribute_cols].to_numpy(dtype=float)

        # Mapping column name to index for numpy access
        self.col_to_idx = {col: i for i, col in enumerate(self.attribute_cols)}

        self.out_diff_path = os.path.join(self.output_diff_dir, f'pw_diff_mx_{self.base}_min.csv')

        print("Computing difference matrix...")
        self.diff_df = self._compute_diff_matrix()

        # To save memory, we don't keep original_diff_matrix as a massive DF in memory
        # unless strictly needed. We will trust diff_df for the initial state.
        print("Filtering difference pairs...")
        self.attrs_df = self._filter_diff_pairs()

        # OPTIMIZATION: Build a cache for O(1) lookup in _get_attr_value
        # Structure: key=(idx1, idx2), value={attr: {'val1': v1, 'val2': v2, 'diff': d}}
        self._build_diff_cache()

        self.dependencies = self._parse_rfds(rfd_file_path)
        self._analyze_attributes()

    def _build_diff_cache(self):
        """Builds a dictionary for fast lookup of pair differences."""
        print("Building difference cache for fast lookup...")
        self.diff_cache = {}

        # It is faster to iterate over the dataframe values directly
        records = self.attrs_df.to_dict('records')
        for row in records:
            key = (row['idx1'], row['idx2'])
            if key not in self.diff_cache:
                self.diff_cache[key] = {}

            self.diff_cache[key][row['attribute']] = {
                'val1': row['val1'],
                'val2': row['val2'],
                'diff': row['diff']
            }

    def order_attributes(self):
        attrs = [c for c in self.dataset_min.columns if c != 'class']
        print('Attributes: \n', attrs)

        ordered = sorted(
            attrs,
            key=lambda x: (
                1 if self.check_bool_attr(x) else 0,
                int(x.replace('Attr', '')) if 'Attr' in x else 999
            )
        )
        return ordered

    def check_bool_attr(self, attr):
        # Vectorized check
        if np.isin(self.imbalance_df_min[attr], [0, 1]).all():
            return True
        return False

    def _parse_rfds(self, rfd_file_path):
        dependencies = []
        with open(rfd_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '->' in line:
                    lhs, rhs = line.split('->')
                    lhs_list = [a.split('@')[0] for a in re.split('[, ]+', lhs) if a and '@' in a]
                    rhs_match = re.match(r"(\w+)@", rhs.strip())
                    if rhs_match:
                        rhs_attr = rhs_match.group(1)
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
        """
        Optimized calculation using NumPy broadcasting instead of itertools.combinations.
        """
        # Get values (N_samples, N_features)
        vals = self.imbalance_df_min[self.attribute_cols].values
        n_samples = vals.shape[0]

        # We only need upper triangular pairs (i < j)
        # Constructing the full N^2 matrix might be heavy if N is large,
        # but for minority classes usually N is small enough.

        # Calculate diffs: |V_i - V_j|
        # expanding dims for broadcasting: (N, 1, F) - (1, N, F)
        diffs = np.abs(vals[:, None, :] - vals[None, :, :])

        # Extract indices for upper triangle
        i_idx, j_idx = np.triu_indices(n_samples, k=1)

        # Filter the differences
        upper_diffs = diffs[i_idx, j_idx, :]

        # Create result dictionary structure
        data = {}
        pair_names = [f"t{i},t{j}" for i, j in zip(i_idx, j_idx)]
        data['tuple_pair'] = pair_names

        for k, col_name in enumerate(self.attribute_cols):
            data[col_name] = upper_diffs[:, k]

        diff_df = pd.DataFrame(data)
        diff_df.to_csv(self.out_diff_path, index=False)
        print(f"Saved initial difference matrix to {self.out_diff_path}")
        return diff_df

    def _filter_diff_pairs(self) -> pd.DataFrame:
        """
        Optimized filtering using vectorized pandas operations.
        """
        # Melt or stack is usually efficient, but let's stick to the logic
        # We need rows where Diff <= Threshold

        # 1. Melt the dataframe to get (tuple_pair, attribute, diff)
        melted = self.diff_df.melt(id_vars=['tuple_pair'], var_name='attribute', value_name='diff')

        # 2. Filter by threshold
        filtered = melted[melted['diff'] <= self.threshold].copy()

        # 3. Extract indices from tuple_pair string (vectorized)
        # Format "tX,tY"
        # We can perform a split and convert to int
        split_pairs = filtered['tuple_pair'].str.replace('t', '').str.split(',', expand=True).astype(int)
        filtered['idx1'] = split_pairs[0]
        filtered['idx2'] = split_pairs[1]

        # 4. Lookup original values.
        # This is faster if we map indices to values using the pre-loaded dataframe
        # We can use map() with a lookup dict or merge. Merge is safer.

        # Add values for idx1
        val_lookup = self.imbalance_df_min[self.attribute_cols].stack().reset_index()
        val_lookup.columns = ['idx', 'attribute', 'val']

        # We can simply use numpy indexing which is faster
        vals = self.imbalance_df_min[self.attribute_cols].values
        col_map = {c: i for i, c in enumerate(self.attribute_cols)}

        # Use apply is slow, let's use list comprehension with numpy array direct access
        # Since filtered is likely smaller than the full cross product, this is okay.
        # But even better:
        attr_indices = filtered['attribute'].map(col_map).values
        idx1_indices = filtered['idx1'].values
        idx2_indices = filtered['idx2'].values

        filtered['val1'] = vals[idx1_indices, attr_indices]
        filtered['val2'] = vals[idx2_indices, attr_indices]

        # Clean up columns
        final_df = filtered[['attribute', 'idx1', 'val1', 'idx2', 'val2', 'diff']]

        base = os.path.basename(self.imbalance_dataset_path).split('.')[0]
        out_path = os.path.join(self.output_diff_tuples_dir, f'diff_tuples_{base}_min.csv')
        final_df.to_csv(out_path, index=False)
        print(f"Saved filtered tuples to {out_path}")
        return final_df

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

    def _get_top_pairs(self):
        if len(self.both_attrs) != 0:
            total_both_attrs = len(self.both_attrs)
            relevant_df = self.attrs_df[self.attrs_df['attribute'].isin(self.both_attrs)]
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
            # print(f"  {attr}: Missing, using sys.max strategy")
            return self._get_safe_fallback_value(attr, use_decimal)

        # OPTIMIZATION: Use Dictionary Lookup instead of DataFrame filtering
        pair_data = self.diff_cache.get((i1, i2), {}).get(attr)

        if pair_data:
            val1 = pair_data['val1']
            val2 = pair_data['val2']
            diff = pair_data['diff']

            if attr in self.no_dependency_attrs:
                # Logic for no dependency
                min_val = min(val1, val2)
                if isinstance(min_val, (int, np.integer)):
                    random_val = min_val + random.randint(0, int(diff))
                else:
                    random_val = min_val + random.uniform(0, float(diff))
                return round(random_val, 2)

            # Range based logic
            if diff <= self.threshold:
                min_val = min(val1, val2)
                max_val = max(val1, val2)

                if diff == 0:
                    return val1
                if diff == 1:
                    # Euclidean check logic
                    try:
                        if generated_values and len(generated_values) > 0:
                            # Optimization: Use current_data_numpy for faster access
                            # (generated_values is a dict, so we convert relevant parts to array)

                            # Filter keys that are already generated and exist in all_attrs
                            keys = [k for k in generated_values.keys() if k != attr and k in self.all_attrs]
                            if keys:
                                vec_generated = np.array([float(generated_values[k]) for k in keys])
                                # Use col_to_idx for fast numpy access
                                key_indices = [self.col_to_idx[k] for k in keys]

                                vec_i1 = self.current_data_numpy[i1, key_indices]
                                vec_i2 = self.current_data_numpy[i2, key_indices]

                                d1 = np.linalg.norm(vec_generated - vec_i1)
                                d2 = np.linalg.norm(vec_generated - vec_i2)
                                return val1 if d1 <= d2 else val2
                    except Exception as e:
                        pass  # Warning suppressed for speed

                    return random.randrange(min_val, max_val + 1)
                else:
                    if use_decimal:
                        return round(random.uniform(min_val, max_val), 2)
                    else:
                        if attr in self.integer_attrs:
                            return random.randint(int(min_val), int(max_val))
                        else:
                            return round(random.uniform(min_val, max_val), 2)
            else:
                return self._get_safe_fallback_value(attr, use_decimal)
        else:
            return self._get_safe_fallback_value(attr, use_decimal)

    def _get_safe_fallback_value(self, attr, use_decimal=False):
        overall_max = self.last_safe_value - self.threshold - 1
        self.last_safe_value = overall_max
        safe_base = overall_max

        if use_decimal:
            decimal_component = round(random.random(), 2)
            return safe_base + decimal_component
        else:
            return safe_base

    def _calculate_distances_for_new_tuple(self, new_tuple_values):
        """
        Calculates distances between the new tuple and all existing tuples in current_data_numpy.
        Returns a DataFrame row-like structure or dict of arrays.
        """
        # Convert new tuple dict to array in correct order
        new_vec = np.array([new_tuple_values.get(c, 0) for c in self.attribute_cols])

        # Calculate absolute difference against all current data: |Existing - New|
        # current_data_numpy is (N, F), new_vec is (F,) -> Broadcasts to (N, F)
        dists = np.abs(self.current_data_numpy - new_vec)

        # We return this as a DataFrame only because the violation check expects column names.
        # Constructing DF here is much faster than appending to a giant one.
        idx_existing = np.arange(len(self.current_data_numpy))
        idx_new = len(self.current_data_numpy)  # Index of the new tuple if it were added

        return pd.DataFrame(dists, columns=self.attribute_cols, index=[f"t{i},t{idx_new}" for i in idx_existing])

    def _check_violations(self, distance_matrix_subset):
        """
        Checks violations on a subset of the distance matrix (usually just the new pairs).
        """
        violations_found = False

        # Pre-fetch columns to avoid repeated DF indexing
        dm_cols = distance_matrix_subset.columns

        for lhs, rhs in self.dependencies:
            # Quick check if cols exist
            if not all(c in dm_cols for c in lhs + [rhs]):
                continue

            # Vectorized check
            # lhs_similar: AND condition across all LHS attributes
            lhs_data = distance_matrix_subset[lhs].values
            rhs_data = distance_matrix_subset[rhs].values

            lhs_similar = (lhs_data <= self.threshold).all(axis=1)
            rhs_dissimilar = (
                    rhs_data > self.threshold)  # rhs is usually a single column, but values returns (N,) or (N,1)

            violations = lhs_similar & rhs_dissimilar

            if violations.any():
                violations_found = True
                # Just print first violation to save I/O time
                # print(f"Violation in dependency {lhs} -> {rhs}")
                break  # Optimization: If we just need to know IF there is a violation, break early.
                # However, the original code loops through all.
                # To keep logic exact: we continue, but we can return True immediately if strict logic isn't required.
                # Sticking to original logic of printing, but optimizing the loop.

        return violations_found

    def _is_duplicate_tuple(self, new_tuple_data, current_data_numpy):
        # Convert new tuple to array
        new_vec = np.array([new_tuple_data[attr] for attr in self.attribute_cols])

        # Vectorized comparison: (N, F) == (F,) -> (N, F) -> all(axis=1) -> (N,)
        # Using a small epsilon for float comparison
        matches = np.all(np.abs(current_data_numpy - new_vec) < 1e-9, axis=1)

        if matches.any():
            print(f"Duplicate found")
            return True
        return False

    def get_safe_value_ranges(self, attr, current_df_numpy=None):
        # Use numpy specific column
        idx = self.col_to_idx[attr]

        if current_df_numpy is not None:
            col_vals = current_df_numpy[:, idx]
        else:
            col_vals = self.current_data_numpy[:, idx]

        existing_values = np.sort(np.unique(col_vals))

        safe_ranges = []
        threshold = self.threshold

        if len(existing_values) == 0:
            return []

        # Logic optimized with simple indexing
        # Start
        start_0 = existing_values[0] - threshold - 5
        end_0 = existing_values[0] - threshold - 1
        if start_0 < end_0:
            safe_ranges.append((start_0, end_0))

        # End
        start_last = existing_values[-1] + threshold + 1
        end_last = existing_values[-1] + threshold + 10
        safe_ranges.append((start_last, end_last))

        # Middle intervals
        # ranges: (val[i] + thr + 1, val[i+1] - thr - 1)
        starts = existing_values[:-1] + threshold + 1
        ends = existing_values[1:] - threshold - 1

        valid_indices = starts < ends
        for s, e in zip(starts[valid_indices], ends[valid_indices]):
            safe_ranges.append((s, e))

        return safe_ranges

    def generate_safe_value(self, attr, avoid_similarity=True):
        if avoid_similarity:
            safe_ranges = self.get_safe_value_ranges(attr)  # Uses self.current_data_numpy implicitly
            if safe_ranges:
                chosen_range = random.choice(safe_ranges)
                min_val, max_val = chosen_range

                if attr in self.integer_attrs:
                    return random.randint(int(min_val), int(max_val))
                else:
                    return round(random.uniform(min_val, max_val), 2)

        return self._get_safe_fallback_value(attr, use_decimal=True)

    def repair_violation_with_fallback(self, row_data, violated_dependency):
        # NOTE: Reduced arguments, uses self.current_data_numpy directly
        lhs_list, rhs_attr = violated_dependency

        repaired = self.repair_violation(row_data, violated_dependency)
        if repaired is not None:
            return repaired

        # Fallback strategy
        print("Max for LHS and RHS fallback")
        related_deps = self._get_related_dependencies_for_violation(violated_dependency)

        critical_attrs = set()
        for dep_lhs, dep_rhs in related_deps:
            attrs_in_dep = set(dep_lhs + [dep_rhs])
            for attr in attrs_in_dep:
                if attr in self.both_attrs:
                    critical_attrs.add(attr)
                    critical_attrs.update(dep_lhs)

        row_data_test = row_data.copy()

        targets = critical_attrs if critical_attrs else (set(lhs_list) | {rhs_attr})
        for attr in targets:
            row_data_test[attr] = self._get_safe_fallback_value(attr, use_decimal=True)

        # Validate
        dist_matrix = self._calculate_distances_for_new_tuple(row_data_test)
        if not self._check_violations(dist_matrix):
            return row_data_test
        return None

    def repair_violation(self, row_data, violated_dependency):
        lhs_list, rhs_attr = violated_dependency

        # Attempt LHS repair
        row_data_test = row_data.copy()
        for lhs_attr in lhs_list:
            safe_val = self.generate_safe_value(lhs_attr, avoid_similarity=True)
            row_data_test[lhs_attr] = safe_val

        dist_matrix = self._calculate_distances_for_new_tuple(row_data_test)
        if not self._check_violations(dist_matrix):
            return row_data_test

        return None

    def identify_violated_dependencies(self, distance_matrix_subset):
        violated_deps = []
        dm_cols = distance_matrix_subset.columns

        for lhs, rhs in self.dependencies:
            if not all(c in dm_cols for c in lhs + [rhs]):
                continue

            lhs_vals = distance_matrix_subset[lhs].values
            rhs_vals = distance_matrix_subset[rhs].values

            lhs_similar = (lhs_vals <= self.threshold).all(axis=1)
            rhs_dissimilar = (rhs_vals > self.threshold)
            violations = lhs_similar & rhs_dissimilar

            if violations.any():
                # We return the dependency info.
                # Original code returned (dep, violation_pairs),
                # but we just need to know WHAT dependency broke to fix it.
                violated_deps.append(((lhs, rhs), None))

        return violated_deps

    def _get_related_dependencies_for_violation(self, violated_dependency):
        lhs_list, rhs_attr = violated_dependency
        all_attrs_in_violation = set(lhs_list + [rhs_attr])
        related_deps = []
        for dep_lhs, dep_rhs in self.dependencies:
            dep_attrs = set(dep_lhs + [dep_rhs])
            if dep_attrs & all_attrs_in_violation:
                related_deps.append((dep_lhs, dep_rhs))
        return related_deps

    def _generate_single_tuple(self, i1, i2, missing_attrs=None):
        # We try max_iter times
        for attempt in range(self.max_iter):
            row_data = {}
            use_decimal = False

            # Generate values
            for attr in self.all_attrs:
                row_data[attr] = self._get_attr_value(attr, i1, i2, use_decimal, generated_values=row_data,
                                                      missing_attrs=missing_attrs)
            row_data['class'] = self.min_class

            # Check duplicate against numpy array (FAST)
            if self._is_duplicate_tuple(row_data, self.current_data_numpy):
                # Try decimal strategy
                use_decimal = True
                for attr in self.all_attrs:
                    row_data[attr] = self._get_attr_value(attr, i1, i2, use_decimal=True, generated_values=row_data,
                                                          missing_attrs=missing_attrs)

                if self._is_duplicate_tuple(row_data, self.current_data_numpy):
                    continue

            # Calculate distances for NEW tuple only
            diff_mx_subset = self._calculate_distances_for_new_tuple(row_data)

            # Check violations
            violated_deps = self.identify_violated_dependencies(diff_mx_subset)

            if not violated_deps:
                return row_data
            else:
                # Repair
                repaired_data = row_data.copy()
                repair_successful = True

                for dep_info, _ in violated_deps:
                    repaired_data = self.repair_violation_with_fallback(repaired_data, dep_info)
                    if repaired_data is None:
                        repair_successful = False
                        break

                if repair_successful and repaired_data is not None:
                    # Final check
                    final_mx = self._calculate_distances_for_new_tuple(repaired_data)
                    if not self._check_violations(final_mx):
                        return repaired_data

        return None

    def augment_dataset(self):
        print(f"Need to generate {self.oversampling_quantity} new samples")
        top_pairs = self._get_top_pairs()

        if len(top_pairs) == 0:
            print("No complete top pairs found, using ALL available pairs from IVD")
            top_pairs = self.attrs_df[['idx1', 'idx2']].drop_duplicates().reset_index(drop=True)

        oversampling_factor = max(1, (self.oversampling_quantity // len(top_pairs)) + 1)

        new_rows = []
        generated_count = 0

        # Helper to convert dict tuple to array row
        def dict_to_arr(d):
            return [d.get(c, 0) for c in self.attribute_cols]

        for _, pair in top_pairs.iterrows():
            if generated_count >= self.oversampling_quantity:
                break

            i1, i2 = pair['idx1'], pair['idx2']

            # Identify missing attrs
            # Optimization: use set logic on pre-computed cache keys or similar?
            # Original logic queries attrs_df again. We can use the cache.
            # Using cache to find which attributes exist for this pair
            pair_entry = self.diff_cache.get((i1, i2), {})
            covered_attrs = set(pair_entry.keys())
            missing_attrs = set(self.all_attrs) - covered_attrs

            for iteration in range(oversampling_factor):
                if generated_count >= self.oversampling_quantity:
                    break

                new_tuple = self._generate_single_tuple(i1, i2, missing_attrs=missing_attrs)

                if new_tuple is not None:
                    new_rows.append(new_tuple)

                    # Update the internal Numpy Array immediately so subsequent tuples utilize it
                    new_vec = np.array(dict_to_arr(new_tuple))
                    self.current_data_numpy = np.vstack([self.current_data_numpy, new_vec])

                    generated_count += 1
                    if generated_count % 10 == 0:
                        print(f"Generated {generated_count}/{self.oversampling_quantity}")

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            # Ensure columns are ordered and class is present
            if 'class' not in new_df.columns:
                new_df['class'] = self.min_class

            if len(new_df) > self.oversampling_quantity:
                new_df = new_df.sample(n=self.oversampling_quantity, random_state=42).reset_index(drop=True)

            # Sorting columns logic
            sorted_cols = sorted(new_df.columns,
                                 key=lambda c: (0, int(re.search(r'\d+', c).group())) if re.search(r'Attr(\d+)$', c)
                                 else (2, 0) if c == 'class' else (1, c))
            new_df = new_df[sorted_cols]

            new_df.to_csv(f'augmentation_results/{self.base}_new_tuples_{self.threshold}.csv', index=False)
            augmented_df = pd.concat([self.imbalance_df_min, new_df], ignore_index=True)

            print(f"\nSuccessfully generated {len(new_df)} new samples")
            self._save_performance_log(len(new_df), success=True)
            return new_df
        else:
            print("No valid tuples could be generated!")
            self._save_performance_log(0, success=False)
            return self.imbalance_df

    def _save_performance_log(self, generated_count, success):
        end_time = time.time()
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        elapsed_time = end_time - self.total_start_time
        peak_memory_mb = peak_mem / (1024 * 1024)

        log_dir = 'augmentation_logs'
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"log_{self.base}_thr{self.threshold}_os{self.oversampling_quantity}.txt"
        log_path = os.path.join(log_dir, log_filename)

        with open(log_path, 'w') as f:
            f.write(f"=== Augmentation Performance Log ===\n")
            f.write(f"Configuration:\n")
            f.write(f"  Dataset: {self.base}\n")
            f.write(f"  Threshold: {self.threshold}\n")
            f.write(f"  Oversampling: {self.oversampling_quantity}\n")
            f.write(f"  Max Iterations: {self.max_iter}\n")
            f.write(f"----------------------------------------\n")
            f.write(f"Time elapsed (total): {elapsed_time:.2f} seconds\n")
            f.write(f"Peak memory consumption: {peak_memory_mb:.2f} MB\n")
            f.write(f"Generated tuples: {generated_count}\n")
            if not success:
                f.write(f"Status: Failed to generate valid tuples\n")

        tracemalloc.stop()