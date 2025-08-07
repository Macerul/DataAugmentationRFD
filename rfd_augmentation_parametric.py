import re
import random
import os
import ast
import numpy as np
import pandas as pd
from itertools import combinations


class RFDAwareAugmenter:
    def __init__(self, imbalance_dataset_path,
                  rfd_file_path,  oversampling,
                  threshold=4, max_iter=100, selected_rfds=None):
        """
        Initialize the RFD-aware data augmenter.

        Args:
            imbalance_dataset_path: Path to the imbalanced dataset
            attr_diff_path: Path to the attribute differences file
            rfd_file_path: Path to the RFD file
            diff_matrix_path: Path to the distance matrix file
            threshold: Similarity threshold for RFDs
            max_iter: Maximum iterations to try generating a valid tuple
            selected_rfds: List of specific RFDs to respect (None = all RFDs)
        """

        # Initialize output directory
        self.output_dir = 'augmentation_results'
        os.makedirs(self.output_dir, exist_ok=True)

        self.output_diff_dir = 'diff_matrices'
        os.makedirs(self.output_diff_dir, exist_ok=True)

        self.output_diff_tuples_dir = 'diff_tuples'
        os.makedirs( self.output_diff_tuples_dir, exist_ok=True)

        self.imbalance_dir = 'imbalanced_datasets'


        self.threshold = threshold
        self.max_iter = max_iter
        self.oversampling_quantity = oversampling
        self.selected_rfds = selected_rfds



        # Load datasets
        self.imbalance_dataset_path = imbalance_dataset_path
        self.base = os.path.basename(self.imbalance_dataset_path).split('.')[0]
        self.imbalance_df = pd.read_csv(imbalance_dataset_path)
        self.dataset_min = self.imbalance_df[self.imbalance_df['class']==1]
        self.out_min_path = os.path.join(self.imbalance_dir, f'{self.base}_min.csv')
        self.dataset_min.to_csv(self.out_min_path, index=False)


        self.imbalance_df_min =  pd.read_csv(self.out_min_path)
        print(f'LETTURA DATASET INIZIALE:\n{self.imbalance_df_min.head()}')



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
        self.all_attrs = sorted(self.attrs_df['attribute'].unique(),
                                key=lambda x: int(x.replace('Attr', '')))
        # Attributes not found in any dependency
        self.dependency_attrs = lhs_attrs | rhs_attrs
        self.no_dependency_attrs = set(self.all_attrs) - self.dependency_attrs


        print(f"Analyzing {len(self.dependencies)} RFDs")
        print(f"All attributes: {self.all_attrs}")
        print(f"Attributes in both LHS and RHS: {sorted(self.both_attrs)}")
        print(f"Attributes not in any dependency: {sorted(self.no_dependency_attrs)}")

    def _get_top_pairs(self):
        """Get tuple pairs that appear across all 'both' attributes."""
        # Filter to attributes that appear in both LHS and RHS
        if len( self.both_attrs) != 0:
            total_both_attrs = len(self.both_attrs)
            relevant_df = self.attrs_df[self.attrs_df['attribute'].isin(self.both_attrs)]
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

    def _get_attr_value(self, attr, i1, i2, use_decimal=False):
        """
        Get attribute value for generation following RFD-aware logic.

        Args:
            attr: Attribute name
            i1, i2: Tuple indices
            use_decimal: If True, generate decimal values to avoid duplicates
        """

        # Try to find the tuple pair data for this attribute
        row = self.attrs_df[(self.attrs_df['attribute'] == attr) &
                            ((self.attrs_df['idx1'] == i1) & (self.attrs_df['idx2'] == i2))]

        if not row.empty:
            # If attr not in any rfd, generate a value random(0, diff)
            if attr in self.no_dependency_attrs:
                r = row.iloc[0]
                val1, val2 = r['val1'], r['val2']
                min_val = min(val1, val2)
                diff_val = abs(val1 - val2)
                if isinstance(min_val, (int, np.integer)):
                    random_val = min_val + random.randint(0, int(diff_val))
                else:
                    random_val = min_val + random.uniform(0, float(diff_val))
                print(f"  {attr}: No dependency, min={min_val}, diff={diff_val}, generating: {random_val:.2f}")
                return round(random_val, 2)
            r = row.iloc[0]
            val1, val2 = r['val1'], r['val2']
            diff = abs(val1 - val2)

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
                else:
                    # Similar values - generate within range
                    if use_decimal:
                        generated_val = random.uniform(min_val, max_val)
                        print(f"    → Similar values, generating decimal: {generated_val:.2f}")
                        return round(generated_val, 2)
                    else:
                        if isinstance(min_val, (int, np.integer)) and isinstance(max_val, (int, np.integer)):
                            generated_val = random.randint(int(min_val), int(max_val))
                        else:
                            generated_val = round(random.uniform(min_val, max_val), 2)
                        print(f"    → Similar values, generating: {generated_val}")
                        return generated_val
            else:
                # Values are dissimilar - use safe fallback to avoid unwanted dependencies
                print(f"    → Dissimilar values (diff={diff} > {self.threshold}), using fallback")
                return self._get_safe_fallback_value(attr, use_decimal)
        else:
            # No data for this tuple pair - use fallback
            print(f"  {attr}: No tuple pair data, using fallback")
            return self._get_safe_fallback_value(attr, use_decimal)

    def _get_safe_fallback_value(self, attr, use_decimal=False):
        """
        Generate a safe fallback value that won't trigger unwanted dependencies.

        Args:
            attr: Attribute name
            use_decimal: If True, add decimal component
        """
        # Get all rows for this attribute and find the global maximum
        attr_rows = self.imbalance_df_min[attr]
        print(' rows for this attribute and find the global maximum:\n', attr_rows )

        overall_max = attr_rows.max()
        print('Overall max:', overall_max)

        # Generate safe value: max + threshold + 1 (+ decimal if requested)
        safe_base = overall_max + self.threshold + 1

        if use_decimal:
            # Add random decimal component to ensure uniqueness
            decimal_component = round(random.random(), 2)  # 0.0 to 1.0
            safe_value = safe_base + decimal_component
            print(f"    → Safe fallback with decimal: {safe_value:.2f}")
            return safe_value
        else:
            print(f"    → Safe fallback: {safe_base}")
            return safe_base

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

        print('New tuple values:\n', new_tuple_values)
        print('current_df:\n', current_df)


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
        print('Matrice aggiornata:\n',updated)
        return updated


    def _check_violations(self, distance_matrix):
        """
        Check for RFD violations in the updated distance matrix.

        Args:
            distance_matrix: Updated distance matrix

        Returns:
            True if violations found, False otherwise
        """
        for lhs, rhs in self.dependencies:
            # Check if all required columns exist
            missing_cols = [col for col in lhs + [rhs] if col not in distance_matrix.columns]
            if missing_cols:
                continue

            # Find violations: LHS diffs <= threshold and RHS diff > threshold
            mask = (distance_matrix[lhs] <= self.threshold).all(axis=1) & \
                   (distance_matrix[rhs] > self.threshold)

            if mask.any():
                return True

        return False

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


    def _generate_single_tuple(self, i1, i2, current_df ,use_decimal=False):
        """
        Generate a single tuple following RFD-aware logic.

        Args:
            i1, i2: Base tuple indices
            current_df: Current dataset
            use_decimal: If True, use decimal values to avoid duplicates

        Returns:
            Dictionary with new tuple values, or None if failed
        """
        for attempt in range(self.max_iter):
            row_data = {}

            print(f"    Attempt {attempt + 1}: Generating tuple based on t{i1},t{i2}")

            # Generate values for all attributes following RFD-aware logic
            for attr in self.all_attrs:
                row_data[attr] = self._get_attr_value(attr, i1, i2, use_decimal)

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


            if not self._check_violations(self.original_diff_matrix):
                print(f"    ✓ Valid tuple generated on attempt {attempt + 1}")
                self.original_diff_matrix.to_csv(self.out_diff_path)
                #temp_df.to_csv(updated_df_path, index=False)
                return row_data
            else:
                print(f"    ✗ Violation detected on attempt {attempt + 1}, retrying...")

        print(f"    Failed to generate valid tuple after {self.max_iter} attempts")
        return None

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

        # Get top tuple pairs
        top_pairs = self._get_top_pairs()
        print(f"Found {len(top_pairs)} suitable tuple pairs")

        if len(top_pairs) == 0:
            print("No suitable tuple pairs found!")
            return self.imbalance_df

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

            for iteration in range(oversampling_factor):
                if generated_count >= oversampling_quantity:
                    break

                print(f"  Iteration {iteration + 1}/{oversampling_factor}")
                new_tuple = self._generate_single_tuple(i1, i2, current_df)

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
            new_df.to_csv(f'augmentation_results/{self.base}_new_tuples.csv', index=False)
            # Combine with original dataset
            augmented_df = pd.concat([self.imbalance_df_min, new_df], ignore_index=True)

            print(f"\nSuccessfully generated {len(new_df)} new samples")
            print(f"Final dataset shape: {augmented_df.shape}")

            return new_df
        else:
            print("No valid tuples could be generated!")
            return self.imbalance_df

'''
# Usage example
def run_augmentation():
    """Example of how to use the RFDAwareAugmenter."""

    # Define file paths
    IMBALANCE_DATASET_PATH = 'imbalanced_datasets/wisconsin.csv'
    IMBALANCE_DATASET_PATH_MIN = 'imbalanced_datasets/wisconsin_min.csv'
    ATTR_PATH_DIFF = 'diff_tuples/diff_tuples_wisconsin_min.csv'
    RFD_FILE = 'discovered_rfds/discovered_rfds_processed/RFD12_E0.0_wisconsin_min.txt'
    DIFF_MATRIX_PATH = 'diff_matrices/pw_diff_mx_wisconsin_min.csv'


    # Optional: specify which RFDs to respect
    selected_rfds = [
        'Attr0 -> Attr1',
        'Attr1 -> Attr0',
        'Attr7 -> Attr4'
        # Add more as needed
    ]

    # Create augmenter instance
    augmenter = RFDAwareAugmenter(
        imbalance_dataset_path=IMBALANCE_DATASET_PATH,
        #imbalance_dataset_path_min = IMBALANCE_DATASET_PATH_MIN,
        #attr_diff_path=ATTR_PATH_DIFF,
        rfd_file_path=RFD_FILE,
        #diff_matrix_path=DIFF_MATRIX_PATH,
        threshold=12,  # RFD similarity threshold
        max_iter=5,  # Maximum attempts per tuple generation
        selected_rfds=None  # Use None for all RFDs, or specify list
    )

    # Run augmentation
    augmented_dataset = augmenter.augment_dataset()

    # Save results
    output_path = os.path.join(augmenter.output_dir, 'augmented_dataset.csv')
    augmented_dataset.to_csv(output_path, index=False)
    print(f"Augmented dataset saved to: {output_path}")

    return augmented_dataset
'''


# Uncomment to run
#augmented_data = run_augmentation()