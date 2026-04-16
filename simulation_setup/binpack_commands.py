#!/usr/bin/env python3
import re
import sys

# Read commands from the file
with open("commands_to_run.txt") as f:
    commands = [line.strip() for line in f if line.strip()]

# Function to extract the core count following "-np"
def extract_np(cmd):
    m = re.search(r'-np\s+(\d+)', cmd)
    if m:
        return int(m.group(1))
    else:
        print("Could not extract core count from:", cmd, file=sys.stderr)
        return 0

# Build a list of tuples: (command, cores)
cmds = [(cmd, extract_np(cmd)) for cmd in commands]

# Sort commands descending by required cores.
# (This brings the 4- and 3-core commands to the front so that pairing of 1's and 3's is easier.)
cmds.sort(key=lambda x: x[1], reverse=True)

bins = []   # Each bin is a dict with keys: "commands" and "sum"
bin_capacity = 32

# Greedy bin packing: for each command, try to put it in the first bin where it fits.
for cmd, cores in cmds:
    placed = False
    for b in bins:
        if b["sum"] + cores <= bin_capacity:
            b["commands"].append(cmd)
            b["sum"] += cores
            placed = True
            break
    if not placed:
        bins.append({"commands": [cmd], "sum": cores})

# Check that each bin sums to 32.
# (Given a total of 640 cores, we expect exactly 20 bins.)
for idx, b in enumerate(bins):
    if b["sum"] != bin_capacity:
        print(f"Warning: Bin {idx} has a total of {b['sum']} cores (expected {bin_capacity}).", file=sys.stderr)

# Write each bin to a separate file; we use the name "job_X.txt"
for i, b in enumerate(bins):
    out_name = f"job_{i}.txt"
    with open(out_name, "w") as fout:
        for line in b["commands"]:
            fout.write(line + "\n")

