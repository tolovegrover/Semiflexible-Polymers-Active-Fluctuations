import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys

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
data_folders = sys.argv[1:] if len(sys.argv) > 1 else ["/home/love/apfs/root/kappa32/bondangle"]
activities = [0, 8, 16, 32, 48, 64, 72, 80, 88, 96, 104]
target_taus = [0.1, 1, 4, 7, 10, 13, 19]

def process_file_bondlength(file_path):
    blengths = []
    
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
            # ts avblength avcostheta
            if len(parts) >= 2:
                try:
                    blengths.append(float(parts[1]))
                except ValueError:
                    pass
                    
        if not blengths: return None
        
        return np.mean(blengths)
        
    except Exception as e:
        print(f"Failed processing {file_path}: {e}")
        return None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_res = os.path.join(script_dir, "bondlength_averaged.txt")
    
    if os.path.exists(output_res):
        print(f"Found existing {output_res}. Skipping.")
        return

    results = {}
    
    for tau in target_taus:
        for act in activities:
            if tau == 0.1: tau_str = "0.1"
            else: tau_str = str(int(tau))
            
            files = []
            for folder in data_folders:
                pat = os.path.join(folder, f"64_32_{act}_0.1_0.1_{tau_str}_*.bondangle")
                files.extend(glob.glob(pat))
            
            with Pool() as p:
                avg_blengths = p.map(process_file_bondlength, files)
            
            ensemble_blengths = [b for b in avg_blengths if b is not None]
                    
            if ensemble_blengths:
                results[(tau, act)] = np.mean(ensemble_blengths)
                print(f"Tau={tau}, f_a={act}: <b_len> = {results[(tau, act)]:.4f} (over {len(ensemble_blengths)} trajectories)")
                
    with open(output_res, 'w') as f:
        f.write("# Ensemble Averaged Bond Length data from last 20% of runs\n")
        f.write("# Tau Activity Average_BondLength\n")
        for tau in target_taus:
            for act in activities:
                if (tau, act) in results:
                    f.write(f"{tau} {act} {results[(tau, act)]:.6f}\n")
                    
    print(f"\nSaved newly processed Bond Lengths to {output_res}")

    # Compile results into a dataframe for easier plotting
    plot_data = []
    taus = sorted(list(set(k[0] for k in results.keys())))
    if 0.1 in taus:
        taus.remove(0.1)
    for tau in taus:
        acts = sorted(list(set(k[1] for k in results.keys() if k[0] == tau)))
        for act in acts:
            plot_data.append({'Tau': tau, 'Act': act, 'b_len': results[(tau, act)]})
            
    df_plot = pd.DataFrame(plot_data)

    # --- Create Scatter Plot ---
    colors = ["#000000", "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    for i, tau in enumerate(taus):
        sub_df = df_plot[df_plot['Tau'] == tau].sort_values(by='Act')
        if sub_df.empty: continue
        
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        tau_label = f"${tau:g}$"
        
        ax.plot(sub_df['Act'], sub_df['b_len'], marker=m, linestyle='-', color=c, 
                markeredgecolor='k', markersize=7, label=tau_label)

    # Place the main legend for generic tau colors inside the plot
    leg1 = ax.legend(loc='upper left', frameon=False, title="Persistence Time $\\tau$", ncol=3, columnspacing=1.0)
    leg1.get_title().set_fontweight('bold')
    
    ax.set_xlabel('Activity $f_a$')
    ax.set_ylabel('Average Bond Length $\\langle b \\rangle$')
    
    # Clean up top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', top=False, right=False)
    
    plt.tight_layout()
    output_pdf = os.path.join(script_dir, "BondLength_vs_Activity.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved professional plot to {output_pdf}")

if __name__ == "__main__":
    main()
