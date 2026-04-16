import os
import glob
import numpy as np

# --- Configuration ---
rg_folders = ["/home/love/apfs/root/kappa128/rg", "/home/love/apfs/root/kappa128_2/rg"]
activities = [0, 8, 16, 32, 48, 64, 72, 80, 88, 96, 104]
target_taus = [0.1, 1, 4, 7, 10, 13, 19]

def process_file_rg(file_path):
    times = []
    rgs = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Consider last 20% of the trace
        n_total = len(lines)
        start_idx = int(0.8 * n_total)
        
        if start_idx >= n_total: return None
        
        steady_lines = lines[start_idx:]
        
        for line in steady_lines:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    rgs.append(float(parts[1]))
                except ValueError:
                    pass
                    
        if not rgs: return None
        
        return np.mean(rgs)
        
    except Exception as e:
        print(f"Failed processing {file_path}: {e}")
        return None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_res = os.path.join(script_dir, "rg_averaged_from_folder.txt")
    
    results = {}
    
    for tau in target_taus:
        for act in activities:
            if tau == 0.1: tau_str = "0.1"
            else: tau_str = str(int(tau))
            
            files = []
            for folder in rg_folders:
                pat = os.path.join(folder, f"64_128_{act}_0.1_0.1_{tau_str}_*.rg")
                files.extend(glob.glob(pat))
            
            ensemble_rgs = []
            for f in files:
                avg_rg = process_file_rg(f)
                if avg_rg is not None:
                    ensemble_rgs.append(avg_rg)
                    
            if ensemble_rgs:
                results[(tau, act)] = np.mean(ensemble_rgs)
                print(f"Tau={tau}, f_a={act}: <Rg> = {results[(tau, act)]:.4f} (over {len(ensemble_rgs)} trajectories)")
                
    with open(output_res, 'w') as f:
        f.write("# Ensemble Averaged Rg data from last 20% of runs\n")
        f.write("# Tau Activity Average_Rg\n")
        for tau in target_taus:
            for act in activities:
                if (tau, act) in results:
                    f.write(f"{tau} {act} {results[(tau, act)]:.6f}\n")
                    
    print(f"\nSaved newly processed Rg to {output_res}")

if __name__ == "__main__":
    main()
