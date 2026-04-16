import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Global Style Settings ---
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.5,
})

def read_rg_file(path):
    data = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.split()
                if not parts: continue
                try:
                    # In both files we wrote: tau act rg
                    tau = float(parts[0]) if path.endswith('rg_averaged_from_folder.txt') else float(parts[1])
                    act = float(parts[1]) if path.endswith('rg_averaged_from_folder.txt') else float(parts[2])
                    rg = float(parts[2]) if path.endswith('rg_averaged_from_folder.txt') else float(parts[3])
                    # Note: rg_results.txt format: something tau act rg. Thus indices [1], [2], [3]
                    # Note: rg_averaged_from_folder.txt format: tau act rg. Thus indices [0], [1], [2]
                    data[(tau, act)] = rg
                except Exception:
                    pass
    return data

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    file_old = os.path.join(parent_dir, "rg_results.txt")
    file_new = os.path.join(script_dir, "rg_averaged_from_folder.txt")
    
    data_old = read_rg_file(file_old)
    
    # Custom read for the new file format
    data_new = {}
    if os.path.exists(file_new):
        with open(file_new, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.split()
                if not parts: continue
                try:
                    tau = float(parts[0])
                    act = float(parts[1])
                    rg = float(parts[2])
                    data_new[(tau, act)] = rg
                except:
                    pass
        
    taus = sorted(list(set(k[0] for k in data_old.keys() | data_new.keys())))
    if 0.1 in taus: taus.remove(0.1)

    colors = ["#000000", "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for i, tau in enumerate(taus):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        tau_label = f"$\\tau={tau:g}$"
        
        # Plot old data
        acts_old = sorted([k[1] for k in data_old.keys() if k[0] == tau])
        rgs_old = [data_old[(tau, a)] for a in acts_old]
        if rgs_old:
            axes[0].plot(acts_old, rgs_old, linestyle='-', marker=m, color=c, 
                         markeredgecolor='k', markersize=7, label=tau_label)

        # Plot new data
        acts_new = sorted([k[1] for k in data_new.keys() if k[0] == tau])
        rgs_new = [data_new[(tau, a)] for a in acts_new]
        if rgs_new:
            axes[1].plot(acts_new, rgs_new, linestyle='-', marker=m, color=c, 
                         markeredgecolor='k', markersize=7, label=tau_label)
                         
    axes[0].set_title("Previous: rg_results.txt")
    axes[1].set_title("New: rg_averaged.txt")
    
    for ax in axes:
        ax.set_xlabel('Activity $f_a$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', top=False, right=False)

    axes[0].set_ylabel('Radius of Gyration $R_g$')
    axes[1].legend(loc='lower right', frameon=False, ncol=2)
    
    plt.tight_layout()
    output_pdf = os.path.join(script_dir, "rg_side_by_side_comparison.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Saved side-by-side plot to {output_pdf}")

if __name__ == "__main__":
    main()
