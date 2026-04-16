import numpy as np
import glob
from multiprocessing import Pool
import os

import sys

import sys

# --- Configuration ---
folders = sys.argv[1:] if len(sys.argv) > 1 else ["/home/love/apfs/root/kappa128/position", "/home/love/apfs/root/kappa128_2/position"]
kappa = 128
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
                    
                    rows.sort(key=lambda r: r[0])
                    coords = np.array([(x, y) for _, x, y in rows])
                    yield t, coords
                except Exception as e:
                    continue

def process_file_tangent_modes(args):
    file_path, tau, act = args
    q = np.arange(1, NUM_MODES + 1)[:, None] 
    j = np.arange(N_segments)[None, :]       
    DCT_matrix = np.cos(np.pi * q * (j + 0.5) / N_segments) * (2.0 / N_segments)
    
    try:
        tmax = 0
        with open(file_path, 'r') as f: content = f.read()
        lines = content.splitlines()
        timesteps = [int(l) for i, l in enumerate(lines) if "ITEM: TIMESTEP" in lines[i-1]] 
        if timesteps: tmax = timesteps[-1]
        tstart = int(0.3 * tmax)
    except:
        tstart = 0
        
    traj_a0 = []
    traj_an = [] 
    
    for t_step, coords in iter_timesteps(file_path):
        if t_step < tstart: continue
        tangents = np.diff(coords, axis=0) 
        a0 = np.mean(tangents, axis=0) 
        traj_a0.append(a0)
        an = DCT_matrix @ tangents 
        traj_an.append(an)
        
    if not traj_a0:
        return (file_path, tau, act, None, None)
        
    traj_a0 = np.array(traj_a0) 
    traj_an = np.array(traj_an) 
    
    var_a0_sq = np.mean(np.sum(traj_a0**2, axis=1))
    var_an_sq = np.mean(np.sum(traj_an**2, axis=2), axis=0)
    return (file_path, tau, act, var_a0_sq, var_an_sq)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_res = os.path.join(script_dir, "tangent_mode_variances.txt")
    
    if os.path.exists(output_res):
        print(f"Found existing {output_res}. Skipping raw dump processing.")
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
    
    with Pool() as p:
        results = []
        for i, res in enumerate(p.imap_unordered(process_file_tangent_modes, tasks), 1):
            results.append(res)
            print(f"\rMapped {i}/{len(tasks)} files...", end='', flush=True)
        print()
        
    aggregated = {}
    for res in results:
        file_path, tau, act, var_a0_sq, var_an_sq = res
        if var_a0_sq is None: continue
        key = (tau, act)
        if key not in aggregated:
            aggregated[key] = {'count': 0, 'var_a0_sq': 0.0, 'var_an_sq': np.zeros_like(var_an_sq)}
        aggregated[key]['var_a0_sq'] += var_a0_sq
        aggregated[key]['var_an_sq'] += var_an_sq
        aggregated[key]['count'] += 1

    with open(output_res, 'w') as f:
        f.write("# Tau Act n Variance\n")
        f.write("# (n=0 represents var_a0_sq)\n")
        for (tau, act), data in sorted(aggregated.items()):
            count = data['count']
            avg_a0 = data['var_a0_sq'] / count
            avg_an = data['var_an_sq'] / count
            f.write(f"{tau} {act} 0 {avg_a0}\n")
            for i, var in enumerate(avg_an):
                n = i + 1
                f.write(f"{tau} {act} {n} {var}\n")
                
    print(f"Saved mode variances to {output_res}")

if __name__ == "__main__":
    main()
