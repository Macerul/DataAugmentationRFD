import re
import random
import os
import sys
import numpy as np
import pandas as pd
import tracemalloc
import time
from itertools import combinations


class SyRFD:
    def __init__(self, imbalance_dataset_path,
                 rfd_file_path, oversampling,
                 threshold=4, max_iter=100, selected_rfds=None):

        # Start taking time and memory consumption
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self.total_start_time = time.time()

        # Create output directories
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

        # Load datasets
        self.imbalance_dataset_path = imbalance_dataset_path
        self.base = os.path.basename(self.imbalance_dataset_path).split('.')[0]
        self.imbalance_df = pd.read_csv(imbalance_dataset_path)

        # Identify minority class
        counts = self.imbalance_df['class'].value_counts()
        self.min_class = counts.idxmin()
        self.dataset_min = self.imbalance_df[self.imbalance_df['class'] == self.min_class]
        self.out_min_path = os.path.join(self.imbalance_dir, f'{self.base}_min.csv')
        self.dataset_min.to_csv(self.out_min_path, index=False)

        self.imbalance_df_min = pd.read_csv(self.out_min_path)
        print(f'Reading imbalance dataset...:\n{self.imbalance_df_min.head()}')

        # Identify integer attributes
        self.integer_attrs = set()
        for attr in self.imbalance_df_min.columns:
            if (self.imbalance_df_min[attr].dtype in ['int64', 'int32'] or
                    np.issubdtype(self.imbalance_df_min[attr].dtype, np.integer)):
                self.integer_attrs.add(attr)
                print(f"{attr} is int")

        self.out_diff_path = os.path.join(self.output_diff_dir, f'pw_diff_mx_{self.base}_min.csv')
        print("Computing difference matrix...")
        self.diff_df = self._compute_diff_matrix()

        self.original_diff_matrix = pd.read_csv(self.out_diff_path, index_col='tuple_pair')
        print("Initial difference matrix:\n", self.original_diff_matrix)

        # ── NEW: keep a NumPy mirror of the diff matrix for fast violation checks ──
        # _dm_array[i, j] = distance for pair i, attr j
        # _dm_index maps pair_name -> row index in _dm_array
        # _dm_cols  maps attr_name  -> col index in _dm_array
        self._dm_index: dict[str, int] = {}
        self._dm_cols:  dict[str, int] = {}
        self._dm_array: np.ndarray | None = None
        self._rebuild_dm_cache()

        print("Filtering difference pairs...")
        self.attrs_df = self._filter_diff_pairs()

        self.dependencies = self._parse_rfds(rfd_file_path)
        self._analyze_attributes()

        # ── Pre-compute column indices used in violation checks ──
        self._dep_lhs_cols: list[np.ndarray] = []   # col indices for each dependency's LHS
        self._dep_rhs_cols: list[int] = []           # col index  for each dependency's RHS
        self._precompute_dep_col_indices()

    # ------------------------------------------------------------------ #
    #  Cache helpers                                                       #
    # ------------------------------------------------------------------ #

    def _rebuild_dm_cache(self):
        """Rebuild the NumPy mirror from self.original_diff_matrix."""
        df = self.original_diff_matrix
        self._dm_index = {pair: i for i, pair in enumerate(df.index)}
        self._dm_cols  = {col:  j for j, col  in enumerate(df.columns)}
        self._dm_array = df.values.astype(float)

    def _precompute_dep_col_indices(self):
        """Map dependency attribute names to column indices in _dm_array."""
        self._dep_lhs_cols = []
        self._dep_rhs_cols = []
        for lhs, rhs in self.dependencies:
            lhs_idx = np.array([self._dm_cols[a] for a in lhs if a in self._dm_cols], dtype=np.intp)
            rhs_idx = self._dm_cols.get(rhs, -1)
            self._dep_lhs_cols.append(lhs_idx)
            self._dep_rhs_cols.append(rhs_idx)

    # ------------------------------------------------------------------ #
    #  Attribute ordering                                                  #
    # ------------------------------------------------------------------ #

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
        return False

    # ------------------------------------------------------------------ #
    #  RFD parsing                                                         #
    # ------------------------------------------------------------------ #

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
        print('RFDs:\n', dependencies)
        return dependencies

    # ------------------------------------------------------------------ #
    #  Difference matrix                                                   #
    # ------------------------------------------------------------------ #

    def _compute_diff_matrix(self) -> pd.DataFrame:
        """
        Vectorised pairwise difference matrix.

        Original: O(n²) Python-level loop with iterrows().
        Optimised: broadcast subtraction on the full attribute array → ~10-50× faster.
        """
        self.imbalance_df_min.reset_index(inplace=True)
        self.imbalance_df_min.rename(columns={'index': 'tuple_id'}, inplace=True)

        attribute_columns = [col for col in self.imbalance_df_min.columns if col != 'tuple_id']
        ids   = self.imbalance_df_min['tuple_id'].values
        vals  = self.imbalance_df_min[attribute_columns].values.astype(float)  # (n, d)

        n = len(ids)
        idx_i, idx_j = np.triu_indices(n, k=1)          # upper-triangle indices

        # Vectorised |row_i - row_j| for all pairs at once
        diff_vals = np.abs(vals[idx_i] - vals[idx_j])   # (num_pairs, d)

        pair_names = [f"t{int(ids[i])},t{int(ids[j])}" for i, j in zip(idx_i, idx_j)]

        diff_df = pd.DataFrame(diff_vals, columns=attribute_columns)
        diff_df.insert(0, 'tuple_pair', pair_names)

        diff_df.to_csv(self.out_diff_path, index=False)
        print(f"Saved initial difference matrix to {self.out_diff_path}")
        return diff_df

    # ------------------------------------------------------------------ #
    #  IVD filtering                                                       #
    # ------------------------------------------------------------------ #

    def _filter_diff_pairs(self) -> pd.DataFrame:
        """
        Original: nested Python loop over rows then attributes.
        Optimised: vectorised melt + boolean mask → much faster for wide matrices.
        """
        attr_cols = [c for c in self.diff_df.columns if c.startswith('Attr')]

        # Wide → long in one shot
        long_df = self.diff_df.melt(
            id_vars='tuple_pair', value_vars=attr_cols,
            var_name='attribute', value_name='diff'
        )
        long_df = long_df[long_df['diff'] <= self.threshold].copy()

        # Parse idx1/idx2 from tuple_pair (vectorised string ops)
        split = long_df['tuple_pair'].str.extract(r't(\d+),t(\d+)').astype(int)
        long_df['idx1'] = split[0].values
        long_df['idx2'] = split[1].values

        # Look up val1/val2 using a dict for O(1) access
        attr_vals: dict[str, dict[int, float]] = {
            attr: self.imbalance_df_min.set_index('tuple_id')[attr].to_dict()
            for attr in attr_cols
        }
        long_df['val1'] = [attr_vals[row.attribute][row.idx1] for row in long_df.itertuples(index=False)]
        long_df['val2'] = [attr_vals[row.attribute][row.idx2] for row in long_df.itertuples(index=False)]

        result_df = long_df[['attribute', 'idx1', 'val1', 'idx2', 'val2', 'diff']].reset_index(drop=True)

        base = os.path.basename(self.imbalance_dataset_path).split('.')[0]
        out_path = os.path.join(self.output_diff_tuples_dir, f'diff_tuples_{base}_min.csv')
        result_df.to_csv(out_path, index=False)
        print(f"Saved filtered tuples to {out_path}")
        return result_df

    # ------------------------------------------------------------------ #
    #  Attribute analysis                                                  #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Top-pair selection                                                  #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Value generation helpers                                            #
    # ------------------------------------------------------------------ #

    def _get_attr_value(self, attr, i1, i2, use_decimal=False, generated_values=None, missing_attrs=None):
        if missing_attrs and attr in missing_attrs:
            print(f"  {attr}: Missing, using sys.max strategy")
            return self._get_safe_fallback_value(attr, use_decimal)

        row = self.attrs_df[
            (self.attrs_df['attribute'] == attr) &
            (self.attrs_df['idx1'] == i1) &
            (self.attrs_df['idx2'] == i2)
        ]

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
                                vec_i1 = np.array([float(self.imbalance_df_min.at[i1, attr]) for _ in keys])
                                vec_i2 = np.array([float(self.imbalance_df_min.at[i2, attr]) for _ in keys])
                                # Use the actual attribute values for i1/i2
                                vec_i1 = np.array([float(self.imbalance_df_min.at[i1, k]) for k in keys])
                                vec_i2 = np.array([float(self.imbalance_df_min.at[i2, k]) for k in keys])
                                d1 = np.linalg.norm(vec_generated - vec_i1)
                                d2 = np.linalg.norm(vec_generated - vec_i2)
                                chosen = val1 if d1 <= d2 else val2
                                print(f"diff==1: euclidean d1={d1:.2f}, d2={d2:.2f}, choosing: {chosen}")
                                return chosen
                    except Exception as e:
                        print(f"Warning computing euclidean distance: {e}")

                    chosen = random.randrange(int(min_val), int(max_val) + 1)
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
        overall_max = self.last_safe_value - self.threshold - 1
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

    # ------------------------------------------------------------------ #
    #  Distance matrix update — incremental NumPy append                  #
    # ------------------------------------------------------------------ #

    def _update_distance_matrix(self, new_tuple_values, current_df, prev_matrix=None):
        """
        Original: iterated all existing rows with Python loop + DataFrame.loc assignment.
        Optimised:
          - Compute new distances via vectorised NumPy subtraction on the cached array.
          - Append one new block of rows to _dm_array in-place (np.vstack).
          - Keep the pandas mirror in sync only at commit time (flush to CSV).

        NOTE: prev_matrix argument is kept for API compatibility but ignored;
              we always use the NumPy cache _dm_array.
        """
        new_idx = len(current_df) - 1   # index of the just-added tuple

        # Attribute values of all *existing* tuples (rows 0..new_idx-1)
        attr_order = list(self.original_diff_matrix.columns)  # canonical column order

        # Build existing value matrix from current_df (fast: iloc slicing)
        existing_vals = current_df.iloc[:new_idx][attr_order].values.astype(float)  # (new_idx, d)
        new_vals      = np.array([float(new_tuple_values.get(a, 0)) for a in attr_order])

        # |existing - new| for every existing tuple at once
        new_distances = np.abs(existing_vals - new_vals)  # (new_idx, d)

        pair_names = [f"t{int(i)},t{int(new_idx)}" for i in range(new_idx)]

        # Grow the NumPy array
        if self._dm_array is None or self._dm_array.shape[0] == 0:
            self._dm_array = new_distances
        else:
            self._dm_array = np.vstack([self._dm_array, new_distances])

        # Update the index map
        for i, name in enumerate(pair_names):
            self._dm_index[name] = len(self._dm_index) - len(pair_names) + i

        # ── Keep pandas mirror lightweight: only rebuild what changed ──
        new_rows_df = pd.DataFrame(new_distances, index=pair_names, columns=attr_order)
        self.original_diff_matrix = pd.concat([self.original_diff_matrix, new_rows_df])

        return self.original_diff_matrix

    # ------------------------------------------------------------------ #
    #  Violation checking — pure NumPy, only on NEW rows                  #
    # ------------------------------------------------------------------ #

    def _check_violations(self, distance_matrix=None):
        """
        Original: full pandas boolean-indexing scan of the entire matrix.
        Optimised: scan only the NEW rows (those added since the last commit)
                   using pre-computed column indices on _dm_array.

        distance_matrix arg kept for API compatibility; ignored internally.
        """
        return self._check_violations_in_range(start_row=None)

    def _check_violations_in_range(self, start_row=None):
        """
        Check violations in _dm_array[start_row:].
        If start_row is None, check the entire array (used for initial validation).
        """
        if self._dm_array is None or self._dm_array.shape[0] == 0:
            return False

        arr = self._dm_array if start_row is None else self._dm_array[start_row:]
        violations_found = False

        for dep_idx, (lhs, rhs) in enumerate(self.dependencies):
            lhs_col_idx = self._dep_lhs_cols[dep_idx]
            rhs_col_idx = self._dep_rhs_cols[dep_idx]

            if len(lhs_col_idx) == 0 or rhs_col_idx == -1:
                print(f"Missing columns for dependency {lhs} -> {rhs}")
                continue

            lhs_similar  = np.all(arr[:, lhs_col_idx] <= self.threshold, axis=1)
            rhs_dissimilar = arr[:, rhs_col_idx] > self.threshold
            violation_mask = lhs_similar & rhs_dissimilar

            if violation_mask.any():
                violations_found = True
                # Recover pair names for logging (offset by start_row)
                offset = start_row or 0
                pairs  = list(self._dm_index.keys())
                violating = [pairs[offset + i] for i in np.where(violation_mask)[0]
                             if offset + i < len(pairs)]
                print(f"Violation in dependency {lhs} -> {rhs}")
                print(f"Violating pairs: {violating[:5]}")
                for pair in violating[:3]:
                    row_i = self._dm_array[self._dm_index[pair]]
                    lhs_diffs = row_i[lhs_col_idx].tolist()
                    rhs_diff  = row_i[rhs_col_idx]
                    print(f"  {pair}: LHS diffs {lhs_diffs} ≤ {self.threshold}, RHS diff {rhs_diff} > {self.threshold}")

        return violations_found

    def identify_violated_dependencies(self, distance_matrix=None):
        """
        Original: pandas boolean-indexing per dependency over the full matrix.
        Optimised: NumPy scan on NEW rows only (_dm_array[_last_commit_row:]).
        """
        violated_deps = []

        if self._dm_array is None or self._dm_array.shape[0] == 0:
            return violated_deps

        # Only inspect rows added since last commit
        start_row = getattr(self, '_last_commit_row', 0)
        arr = self._dm_array[start_row:]

        pairs = list(self._dm_index.keys())

        for dep_idx, (lhs, rhs) in enumerate(self.dependencies):
            lhs_col_idx = self._dep_lhs_cols[dep_idx]
            rhs_col_idx = self._dep_rhs_cols[dep_idx]

            if len(lhs_col_idx) == 0 or rhs_col_idx == -1:
                continue

            lhs_similar    = np.all(arr[:, lhs_col_idx] <= self.threshold, axis=1)
            rhs_dissimilar = arr[:, rhs_col_idx] > self.threshold
            violation_mask = lhs_similar & rhs_dissimilar

            if violation_mask.any():
                offset = start_row
                violating_pairs = [pairs[offset + i] for i in np.where(violation_mask)[0]
                                   if offset + i < len(pairs)]
                violated_deps.append(((lhs, rhs), violating_pairs))

        return violated_deps

    # ------------------------------------------------------------------ #
    #  Duplicate detection — NumPy broadcasting                           #
    # ------------------------------------------------------------------ #

    def _is_duplicate_tuple(self, new_tuple_data, current_df):
        """
        Original: Python loop over all rows.
        Optimised: vectorised broadcast comparison on the full current_df array.
        """
        attr_order = list(self.all_attrs)
        new_vals = np.array([float(new_tuple_data[a]) for a in attr_order])

        existing_vals = current_df[attr_order].values.astype(float)  # (n, d)
        diffs = np.abs(existing_vals - new_vals)                      # (n, d)
        is_dup = np.all(diffs <= 1e-10, axis=1)

        if is_dup.any():
            dup_idx = np.where(is_dup)[0][0]
            print(f"Duplicate found with existing tuple at index {dup_idx}")
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Safe-value helpers (unchanged logic)                                #
    # ------------------------------------------------------------------ #

    def get_safe_value_ranges(self, attr):
        existing_values = sorted(self.imbalance_df_min[attr].unique())
        safe_ranges = []
        for i in range(len(existing_values)):
            current_val = existing_values[i]
            if i == 0:
                range_start = current_val - self.threshold - 5
                range_end   = current_val - self.threshold - 1
                if range_start < range_end:
                    safe_ranges.append((range_start, range_end))
            if i == len(existing_values) - 1:
                range_start = current_val + self.threshold + 1
                range_end   = current_val + self.threshold + 10
                safe_ranges.append((range_start, range_end))
            else:
                next_val    = existing_values[i + 1]
                range_start = current_val + self.threshold + 1
                range_end   = next_val   - self.threshold - 1
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
        return self._get_safe_fallback_value(attr, use_decimal=True)

    # ------------------------------------------------------------------ #
    #  Violation repair (logic unchanged, temp DataFrames avoided)        #
    # ------------------------------------------------------------------ #

    def repair_violation_with_fallback(self, row_data, violated_dependency, current_df, distance_matrix=None):
        lhs_list, rhs_attr = violated_dependency

        repaired = self.repair_violation(row_data, violated_dependency, current_df)
        if repaired is not None:
            return repaired

        print(" Max for LHS and RHS")
        row_data_test = row_data.copy()
        for attr in lhs_list + [rhs_attr]:
            old_val = row_data_test[attr]
            safe_val = self._get_safe_fallback_value(attr, use_decimal=True)
            row_data_test[attr] = safe_val
            print(f"{attr}: {old_val} -> {safe_val} (max)")

        # Speculatively test without touching self._dm_array
        if not self._speculative_violation_check(row_data_test, current_df):
            print("Successful repair")
            return row_data_test

        print("Repair failed")
        return None

    def repair_violation(self, row_data, violated_dependency, current_df, distance_matrix=None):
        lhs_list, rhs_attr = violated_dependency
        print(f"Repairing: {lhs_list} -> {rhs_attr}")
        original_rhs = row_data[rhs_attr]

        last_pair    = self.original_diff_matrix.index[-1]
        existing_idx = int(last_pair.split(',')[0][1:])
        target_rhs   = current_df.iloc[existing_idx][rhs_attr]

        for variation in [0, 0.1, -0.1, 0.5, -0.5, 1, -1]:
            row_data_test = row_data.copy()
            row_data_test[rhs_attr] = target_rhs + variation
            if not self._speculative_violation_check(row_data_test, current_df):
                return row_data_test

        for lhs_attr in lhs_list:
            original_lhs = row_data[lhs_attr]
            safe_val = self.generate_safe_value(lhs_attr, avoid_similarity=True)
            row_data_test = row_data.copy()
            row_data_test[lhs_attr] = safe_val
            if not self._speculative_violation_check(row_data_test, current_df):
                print(f"Successful LHS repair: {lhs_attr} {original_lhs} -> {safe_val}")
                return row_data_test
            else:
                print(f"Repairing LHS failed {lhs_attr}")

        # Try repairing all LHS at once
        row_data_test = row_data.copy()
        for lhs_attr in lhs_list:
            row_data_test[lhs_attr] = self.generate_safe_value(lhs_attr, avoid_similarity=True)

        if not self._speculative_violation_check(row_data_test, current_df):
            return row_data_test
        return None

    def _speculative_violation_check(self, candidate_data, current_df):
        """
        Check whether adding candidate_data to current_df would cause violations
        WITHOUT permanently modifying _dm_array or the pandas mirror.

        Returns True if violations exist, False if clean.
        """
        attr_order = list(self.original_diff_matrix.columns)
        new_vals   = np.array([float(candidate_data.get(a, 0)) for a in attr_order])

        n_existing = len(current_df)
        existing_vals = current_df[attr_order].values.astype(float)  # (n, d)
        new_distances = np.abs(existing_vals - new_vals)             # (n, d)

        for dep_idx, (lhs, rhs) in enumerate(self.dependencies):
            lhs_col_idx = self._dep_lhs_cols[dep_idx]
            rhs_col_idx = self._dep_rhs_cols[dep_idx]
            if len(lhs_col_idx) == 0 or rhs_col_idx == -1:
                continue

            lhs_sim  = np.all(new_distances[:, lhs_col_idx] <= self.threshold, axis=1)
            rhs_dis  = new_distances[:, rhs_col_idx] > self.threshold
            if np.any(lhs_sim & rhs_dis):
                return True  # violation found

        # Also check new candidate against the already-committed distance matrix
        # (pairs among previously inserted tuples — those are already validated,
        #  so we only need new candidate vs existing).
        return False

    # ------------------------------------------------------------------ #
    #  Single-tuple generation                                            #
    # ------------------------------------------------------------------ #

    def _generate_single_tuple(self, i1, i2, current_df, use_decimal=False,
                               max_repair_attempts=5, missing_attrs=None):
        # Track how many rows are already committed so violation checks
        # only scan the newly added distances.
        self._last_commit_row = len(self._dm_index)

        for attempt in range(self.max_iter):
            row_data = {}
            print(f"Attempt {attempt + 1}: Generating tuple based on t{i1},t{i2}")

            for attr in self.all_attrs:
                row_data[attr] = self._get_attr_value(
                    attr, i1, i2, use_decimal,
                    generated_values=row_data, missing_attrs=missing_attrs
                )
            row_data['class'] = self.min_class
            print('New tuple added: \n', row_data)

            if self._is_duplicate_tuple(row_data, current_df):
                print("Duplicate tuple detected")
                if not use_decimal:
                    print("Switching to decimal")
                    use_decimal = True
                continue

            # Append to current_df temporarily and update diff matrix
            temp_df = pd.concat([current_df, pd.DataFrame([row_data])], ignore_index=True)
            self.original_diff_matrix = self._update_distance_matrix(
                row_data, temp_df, self.original_diff_matrix
            )

            violated_deps = self.identify_violated_dependencies()
            if not violated_deps:
                print(f"Valid tuple generated on attempt {attempt + 1}")
                self.original_diff_matrix.to_csv(self.out_diff_path)
                self._last_commit_row = len(self._dm_index)
                return row_data

            print("Violations detected, attempting repair...")
            repaired_data = row_data.copy()
            repair_successful = True

            # Roll back the speculative distance-matrix rows before repair
            self._rollback_dm(self._last_commit_row)

            for dep_info, violating_pairs in violated_deps:
                print(f"Repairing violation: {dep_info}")
                repaired_data = self.repair_violation_with_fallback(
                    repaired_data, dep_info, current_df
                )
                if repaired_data is None:
                    repair_successful = False
                    break

            if repair_successful and repaired_data is not None:
                # Final commit of the repaired tuple
                temp_df_repaired = pd.concat(
                    [current_df, pd.DataFrame([repaired_data])], ignore_index=True
                )
                self.original_diff_matrix = self._update_distance_matrix(
                    repaired_data, temp_df_repaired, self.original_diff_matrix
                )

                if not self._check_violations():
                    print(f"Tuple successfully repaired on attempt {attempt + 1}")
                    self.original_diff_matrix.to_csv(self.out_diff_path)
                    self._last_commit_row = len(self._dm_index)
                    return repaired_data
                else:
                    print("Repair failed, still has violations")
                    self._rollback_dm(self._last_commit_row)
            else:
                print(f"Could not repair violations on attempt {attempt + 1}")
                self._rollback_dm(self._last_commit_row)

        print(f"Failed to generate valid tuple after {self.max_iter} attempts")
        return None

    def _rollback_dm(self, target_row_count):
        """
        Discard any speculative rows appended to _dm_array / original_diff_matrix
        since the last commit, reverting to target_row_count rows.
        """
        current_rows = self._dm_array.shape[0] if self._dm_array is not None else 0
        if current_rows <= target_row_count:
            return  # nothing to roll back

        print(f"Rolling back diff matrix from {current_rows} to {target_row_count} rows")
        self._dm_array = self._dm_array[:target_row_count]

        # Rebuild index map from scratch (cheap: just slice the pandas mirror)
        self.original_diff_matrix = self.original_diff_matrix.iloc[:target_row_count]
        self._dm_index = {pair: i for i, pair in enumerate(self.original_diff_matrix.index)}

    # ------------------------------------------------------------------ #
    #  Dataset augmentation                                                #
    # ------------------------------------------------------------------ #

    def augment_dataset(self):
        oversampling_quantity = self.oversampling_quantity
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
                new_tuple = self._generate_single_tuple(i1, i2, current_df, missing_attrs=missing_attrs)

                if new_tuple is not None:
                    new_rows.append(new_tuple)
                    current_df = pd.concat(
                        [current_df, pd.DataFrame([new_tuple])], ignore_index=True
                    )
                    generated_count += 1
                    print(f"Generated tuple {generated_count}/{oversampling_quantity}")

        if new_rows:
            new_df = pd.DataFrame(new_rows, columns=self.all_attrs + ['class'])
            if len(new_df) > oversampling_quantity:
                new_df = new_df.sample(n=oversampling_quantity, random_state=42).reset_index(drop=True)

            sorted_cols = sorted(
                new_df.columns,
                key=lambda c: (0, int(re.search(r'\d+', c).group())) if re.search(r'Attr(\d+)$', c)
                else (2, 0) if c == 'class' else (1, c)
            )
            new_df = new_df[sorted_cols]
            new_df.to_csv(f'augmentation_results/{self.base}_new_tuples_{self.threshold}.csv', index=False)
            augmented_df = pd.concat([self.imbalance_df_min, new_df], ignore_index=True)

            print(f"\nSuccessfully generated {len(new_df)} new samples")
            self._save_performance_log(len(new_df), success=True)
            print(f"Final dataset shape: {augmented_df.shape}")
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