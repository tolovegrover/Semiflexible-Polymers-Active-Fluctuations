import numpy as np
import glob
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# --- 1. Global LaTeX & Publication Aesthetics ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.5,
})

# --- Configuration ---
folders = sys.argv[1:] if len(sys.argv) > 1 else ["/home/love/apfs/root/kappa128/position"]
kappa = 32
endn = 64
N_segments = endn - 1 # 63
b_len = 2.0**(1.0/6.0)
L = float(N_segments) * b_len
activities = [0, 8, 16, 32, 48, 64, 72, 80, 88, 96, 104]
target_taus = [0.1, 1, 4, 7, 10, 13, 19]
NUM_MODES = 63 # Calculate up to N-1 (max) modes

# --- Data Processing Functions ---
def iter_timesteps(file_path):
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            if line.startswith("ITEM: TIMESTEP"):
                try:
                    t = int(f.readline().strip())
                    f.readline(); N = int(f.readline().strip())
                    f.readline(); f.readline(); f.readline(); f.readline()
                    header = f.readline().split()[2:] # ITEM: ATOMS id xu yu ...
                    content = [f.readline() for _ in range(N)]
                    cols = [l.split() for l in content]
                    try:
                        idx_id = header.index("id")
                        idx_xu = header.index("xu")
                        idx_yu = header.index("yu")
                    except ValueError:
                        idx_id = header.index("id")
                        idx_xu = header.index("x")
                        idx_yu = header.index("y")
                        
                    rows = []
                    for c in cols:
                        rows.append((int(c[idx_id]), float(c[idx_xu]), float(c[idx_yu])))
                    
                    # Sort by atom ID
                    rows.sort(key=lambda r: r[0])
                    coords = np.array([(x, y) for _, x, y in rows])
                    yield t, coords
                except Exception as e:
                    continue

def process_file_bond_fluctuations(args):
    file_path, tau, act = args
    # Changing to include the 0th mode
    q = np.arange(0, NUM_MODES + 1)[:, None] 
    j = np.arange(N_segments)[None, :]       
    DCT_matrix = np.cos(np.pi * q * (j + 0.5) / N_segments) * (2.0 / N_segments)
    
    try:
        tmax = 0
        with open(file_path, 'r') as f: content = f.read()
        lines = content.splitlines()
        timesteps = [int(l) for i, l in enumerate(lines) if "ITEM: TIMESTEP" in lines[i-1]] 
        if timesteps: tmax = timesteps[-1]
        
        # Taking last 30% of the trajectory
        tstart = int(0.7 * tmax)
    except:
        tstart = 0
        
    all_bond_lengths = []
    
    for t_step, coords in iter_timesteps(file_path):
        if t_step < tstart: continue
        
        # Calculate instantaneous bond lengths
        # b_n(t) = || r_{n+1}(t) - r_n(t) ||
        bond_vectors = np.diff(coords, axis=0)
        bond_lengths = np.linalg.norm(bond_vectors, axis=1)
        
        all_bond_lengths.append(bond_lengths)
        
    if not all_bond_lengths:
        return (file_path, tau, act, None)
        
    # Convert list of arrays to 2D numpy array: shape = (n_timesteps, N_segments)
    b_n_t = np.array(all_bond_lengths)
    
    # Calculate text temporal average of the bond lengths
    # <b_n> = mean over time t
    b_n_mean = np.mean(b_n_t, axis=0)
    
    # Calculate the fluctuations
    # delta b_n(t) = b_n(t) - <b_n>
    delta_b_n_t = b_n_t - b_n_mean
    
    # Calculate spatial DCT modes of the fluctuations
    # B_p(t) = sum_{n} delta b_n(t) * DCT_matrix
    # DCT matrix shape (NUM_MODES, N_segments)
    # delta_b_n_t shape (n_timesteps, N_segments)
    
    # Transform to mode amplitudes: B_p(t) has shape (n_timesteps, NUM_MODES)
    # Using Einstein summation: t is time, p is mode, n is segment
    # B_{t,p} = sum_n delta_b_{t,n} * DCT_{p,n} 
    # B = delta_b @ DCT^T
    B_p_t = delta_b_n_t @ DCT_matrix.T
    
    # Calculate the time-averaged variance of the mode amplitudes: <B_p^2>
    var_B_p = np.mean(B_p_t**2, axis=0)
    
    return (file_path, tau, act, var_B_p)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_res = os.path.join(script_dir, "bond_fluctuation_variances.txt")
    output_pdf = os.path.join(script_dir, "Bond_Fluctuation_Modes.pdf")
    
    if os.path.exists(output_res):
        print(f"Found existing {output_res}. Skipping.")
        return

    tasks = []
    for tau in target_taus:
        for act in activities:
            if tau == 0.1: tau_str = "0.1"
            else: tau_str = str(int(tau))
            
            for folder in folders:
                pat = f"{folder}/position.64_128_{act}_0.1_0.1_{tau_str}_*.dump"
                for f in glob.glob(pat):
                    tasks.append((f, tau, act))
                
    print(f"Found {len(tasks)} files to process.")
    
    if not tasks:
        print("No dump files found matching the criteria. Exiting.")
        return
        
    with Pool() as p:
        results = []
        for i, res in enumerate(p.imap_unordered(process_file_bond_fluctuations, tasks), 1):
            results.append(res)
            print(f"\r{i}/{len(tasks)} files processed...", end='', flush=True)
        print()
        
    aggregated = {}
    for res in results:
        file_path, tau, act, var_B_p = res
        if var_B_p is None: continue
        key = (tau, act)
        if key not in aggregated:
            aggregated[key] = {'count': 0, 'var_B_p': np.zeros_like(var_B_p)}
        aggregated[key]['var_B_p'] += var_B_p
        aggregated[key]['count'] += 1

    # Save text
    with open(output_res, 'w') as f:
        f.write("# Bond length fluctuation mode variances (taking last 30% of steps)\n")
        f.write("# Tau Act p Variance\n")
        for (tau, act), data in sorted(aggregated.items()):
            count = data['count']
            avg_var = data['var_B_p'] / count
            for i, var in enumerate(avg_var):
                p = i # Since we start from mode 0 array
                f.write(f"{tau} {act} {p} {var}\n")
                
    print(f"Saved mode variances to {output_res}")

if __name__ == "__main__":
    main()
