"""
----------------
Processes all RFD files in discovered_rfds/ and writes standardised versions
to discovered_rfds_processed/, keeping the same filename.

For empty-LHS entries ( -> AttrX@t ), the LHS attributes are taken from the
actual dataset CSV found in imbalanced_datasets/ whose name matches the
Dataset field in the RFD file (e.g. "pima_min.csv" -> imbalanced_datasets/pima_min.csv).
Only columns that exist in the CSV (excluding 'class') and are different from
the RHS attribute are used as LHS.

RFDs are sorted by number of LHS attributes (ascending).
"""

import os
import re
import glob
import pandas as pd

INPUT_DIR   = "../discovered_rfds"
OUTPUT_DIR  = "../discovered_rfds/discovered_rfds_processed"
DATASET_DIR = "../imbalanced_datasets/min"


# ─────────────────────────────────────────────────────────────
#  Dataset column loading
# ─────────────────────────────────────────────────────────────

def get_dataset_attrs(dataset_filename: str, threshold: str) -> list[str]:
    """
    Load the CSV from imbalanced_datasets/<dataset_filename> and return
    its attribute column names (excluding 'class') formatted as Attr@threshold,
    e.g. ["Attr0@2.0", "Attr1@2.0", ...].

    Falls back to an empty list if the file cannot be found.
    """
    csv_path = os.path.join(DATASET_DIR, dataset_filename)
    if not os.path.isfile(csv_path):
        print(f"  [WARN] Dataset not found: {csv_path} — empty-LHS entries will be skipped")
        return []

    df = pd.read_csv(csv_path, nrows=0)   # only header needed
    attrs = [
        f"{col}@{threshold}"
        for col in df.columns
        if col.lower() != "class"
    ]
    print(f"  Dataset columns ({len(attrs)}): {attrs}")
    return attrs


# ─────────────────────────────────────────────────────────────
#  RFD file parsing
# ─────────────────────────────────────────────────────────────

def parse_rfd_file(filepath: str) -> tuple:
    """
    Returns:
        dataset_name   : str  (value after "Dataset:")
        threshold      : str  (value after "Comparison:")
        normal_rfds    : list of (lhs_tokens, rhs_token)
        empty_lhs_rfds : list of rhs tokens with empty LHS
    """
    dataset_name   = ""
    threshold      = ""
    normal_rfds    = []
    empty_lhs_rfds = []
    in_rfd_section = False

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line.startswith("Dataset:"):
                dataset_name = line.split(":", 1)[1].strip()
                continue

            if line.startswith("Comparison:"):
                threshold = line.split(":", 1)[1].strip()
                continue

            if line == "RFDs:":
                in_rfd_section = True
                continue

            if in_rfd_section and re.match(
                r"^(Time limit|Out of memory|Memory consumption|RFDs count)", line
            ):
                in_rfd_section = False
                continue

            if not in_rfd_section or "->" not in line:
                continue

            lhs_part, rhs_part = line.split("->", 1)
            token_re = r"(?:Attr\w+|class)@[\d.]+"
            lhs_tokens = re.findall(token_re, lhs_part)
            rhs_tokens = re.findall(token_re, rhs_part)

            if not rhs_tokens:
                continue

            rhs = rhs_tokens[0]

            if lhs_tokens:
                normal_rfds.append((lhs_tokens, rhs))
            else:
                empty_lhs_rfds.append(rhs)

    return dataset_name, threshold, normal_rfds, empty_lhs_rfds


# ─────────────────────────────────────────────────────────────
#  Formatting
# ─────────────────────────────────────────────────────────────

def format_line(lhs_tokens: list[str], rhs: str) -> str:
    return ",".join(lhs_tokens) + ", -> " + rhs


def process_file(filepath: str, output_dir: str) -> None:
    print(f"\nProcessing: {filepath}")

    dataset_name, threshold, normal_rfds, empty_lhs_rfds = parse_rfd_file(filepath)
    print(f"  Dataset: {dataset_name}  |  Threshold: {threshold}")
    print(f"  Normal RFDs: {len(normal_rfds)}  |  Empty-LHS: {len(empty_lhs_rfds)}")

    # Load real dataset columns only when needed
    dataset_attrs = []
    if empty_lhs_rfds:
        dataset_attrs = get_dataset_attrs(dataset_name, threshold)

    rfd_lines = []

    # Normal RFDs
    for lhs_tokens, rhs in normal_rfds:
        rfd_lines.append((len(lhs_tokens), format_line(lhs_tokens, rhs)))

    # Expanded empty-LHS: one line per dataset attribute ≠ rhs
    for rhs in empty_lhs_rfds:
        for attr in dataset_attrs:
            if attr != rhs:
                rfd_lines.append((1, format_line([attr], rhs)))

    # Sort by LHS length ascending
    rfd_lines.sort(key=lambda x: x[0])

    output_lines = ["****** DISCOVERED RFDs *******"] + [line for _, line in rfd_lines]

    filename = os.path.basename(filepath)
    base = os.path.splitext(filename)[0]

    # Separo da destra: ..._<rfd>_<epsilon>_<flag>
    try:
        left, rfd_value, epsilon, _ = base.rsplit("_", 3)
    except ValueError:
        raise ValueError(f"Formato filename non valido: {filename}")

    # Rimuovo eventuale .csv dal dataset
    dataset_name = left.replace(".csv", "")

    # Converto RFD (es. 2.0 -> 2)
    try:
        rfd_int = int(float(rfd_value))
    except ValueError:
        raise ValueError(f"Valore RFD non valido in: {filename}")

    new_filename = f"RFD{rfd_int}_E{epsilon}_{dataset_name}.txt"
    out_path = os.path.join(output_dir, new_filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")

    print(f"  -> Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.log")))
    if not files:
        print(f"No .log files found in '{INPUT_DIR}/'")
        return

    print(f"Found {len(files)} file(s) in '{INPUT_DIR}/'")
    for filepath in files:
        try:
            process_file(filepath, OUTPUT_DIR)
        except Exception as e:
            print(f"  [ERROR] {filepath}: {e}")

    print(f"\nDone. Results in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()