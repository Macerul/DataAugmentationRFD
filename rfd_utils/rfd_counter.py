import os
import re

base_dir = "../discovered_rfds/discovered_rfds_processed"

rfd_pattern = re.compile(r"Attr\d+@\d+\.?\d*,\s*->\s*Attr\d+@\d+\.?\d*")

rfd_counts = {}

for filename in os.listdir(base_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(base_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        count = sum(1 for line in lines if rfd_pattern.search(line))
        rfd_counts[filename] = count

print("RFD per file:\n")
for fname, count in sorted(rfd_counts.items()):
    print(f"{fname}:{count} RFDcs")
