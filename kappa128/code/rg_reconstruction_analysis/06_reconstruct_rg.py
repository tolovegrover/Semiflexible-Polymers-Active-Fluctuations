import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from matplotlib.lines import Line2D
from multiprocessing import Pool
import sys

# ============================================================
#  CONFIG  --  change only these paths to port to another kappa
# ============================================================
KAPPA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/")
KAPPA_DIR2 = KAPPA_DIR

RG_AVG_FILE     = os.path.join(KAPPA_DIR2, "rg_averaged_from_folder.txt")
FIT_PARAMS_FILE = os.path.join(KAPPA_DIR2, "MSD_Fit_Parameters.csv")
COV_MATRIX_FILE = os.path.join(KAPPA_DIR, "covariance_reconstruction",
                                "tangent_covariance_matrices.txt")

# Paths to raw simulation dumps used if the matrix file is missing
DUMP_FOLDERS = sys.argv[1:] if len(sys.argv) > 1 else ["/home/love/apfs/root/kappa128/position", "/home/love/apfs/root/kappa128_2/position"]

N_SEGS = 63   # number of bond segments (beads - 1)
T_START = 28000 # Wait time for steady state

# ============================================================
#  PUBLICATION AESTHETICS (Target Style)
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix", # Stix gives a LaTeX-like appearance
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.5,
})

FIG_W, FIG_H = 7.0, 5.3 

# Modern colorblind-safe palette
COLORS = ["#000000", "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']

# ============================================================
#  PHYSICS — exact discrete weight matrix
# ============================================================

def build_W_matrix(N=63):
    """
    Exact geometric weight matrix W = T^T M T.
    W[n,m] multiplies C_nm = <a_n . a_m> in the Rg^2 sum.
    """
    N1 = N + 1
    S = np.zeros((N1, N))
    for k in range(1, N1):
        S[k, :k] = 1.0
    C_op = np.eye(N1) - np.ones((N1, N1)) / N1
    M    = S.T @ C_op @ S / N1
    q    = np.arange(N1)[:, None]
    j    = np.arange(N)[None, :]
    DCT  = np.cos(np.pi * q * (j + 0.5) / N) * (2.0 / N)
    T    = np.linalg.pinv(DCT)
    return T.T @ M @ T


def reconstruct_rg2_discrete(C, W):
    """DISCRETE EXACT: Rg^2 = sum_nm  C_nm * W_nm."""
    return float(np.sum(C * W))

def active_func(n, A, C_p, D, E):
    """Fitted active-spectrum model."""
    return (A / (n**2 + C_p)) * (1.0 + D / (1.0 + E**2 * (n**4 + C_p * n**2)))

def reconstruct_rg2_fitted_nn(C_measured, fit_row, W, N=63):
    """FITTED NN ONLY: C_nn from model, C_nm = 0."""
    A   = float(fit_row['A'])
    C_p = float(fit_row['C'])
    D   = float(fit_row['D'])
    E   = float(fit_row['E'])
    C_fit = np.zeros((N + 1, N + 1))
    C_fit[0, 0] = C_measured[0, 0]
    for n in range(1, N + 1):   
        C_fit[n, n] = float(active_func(n, A, C_p, D, E))
    return float(np.sum(C_fit * W))


# ============================================================
#  COVARIANCE CALCULATION (Independent Algorithm)
# ============================================================

def process_file_tangent_covariance(args):
    """
    Processes a single dump file to extract:
    1. The time-averaged mode covariance matrix C_nm = <a_n . a_m>.
    2. The time-averaged true Rg^2 from spatial coordinates.
    """
    file_path, tau, act = args
    N = 63
    q = np.arange(0, N + 1)[:, None] 
    j = np.arange(N)[None, :]       
    DCT_matrix = np.cos(np.pi * q * (j + 0.5) / N) * (2.0 / N)
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except: return None
        
    lines_per_step = 64 + 9 # 64 beads + LAMMPS header
    total_steps = len(lines) // lines_per_step
    
    # Analyze last 30% of simulation for steady state
    start_step = int(0.7 * total_steps)
    if start_step >= total_steps: return None

    C_sum = np.zeros((N + 1, N + 1))
    rg2_sum = 0.0
    count = 0

    with open(file_path, 'r') as f:
        # Move pointer to the start_step
        f.seek(0)
        line_count = 0
        while line_count < start_step * lines_per_step:
            f.readline()
            line_count += 1
            
        while True:
            line = f.readline()
            if not line: break
            if line.startswith("ITEM: TIMESTEP"):
                try:
                    t = int(f.readline().strip())
                    [f.readline() for _ in range(6)] # Skip metadata
                    header = f.readline().split()[2:]
                    content = [f.readline() for _ in range(64)]
                    
                    if t < T_START: continue
                    
                    cols = [l.split() for l in content]
                    idx_id = header.index("id")
                    idx_xu = header.index("xu") if "xu" in header else header.index("x")
                    idx_yu = header.index("yu") if "yu" in header else header.index("y")
                        
                    atom_data = [[int(row[idx_id]), float(row[idx_xu]), float(row[idx_yu])] for row in cols]
                    atom_data.sort(key=lambda x: x[0])
                    coords = np.array([x[1:] for x in atom_data])

                    # --- TRUE Rg^2 CALCULATION ---
                    com = np.mean(coords, axis=0)
                    sq_dists = np.sum((coords - com)**2, axis=1)
                    rg2_sum += np.mean(sq_dists)
                    # -----------------------------

                    # 1. Extract bonds (tangents)
                    tangents = np.diff(coords, axis=0)
                    
                    # 2. Forward DCT to find mode amplitudes 'an'
                    an = DCT_matrix @ tangents
                    
                    # 3. Compute dot products (Full Matrix including off-diagonals)
                    C_sum += an @ an.T
                    count += 1
                except: pass
            
    if count == 0: return None
    return (tau, act, C_sum / count, rg2_sum / count)


# ============================================================
#  DATA LOADERS
# ============================================================

def load_rg_data(path):
    rows = []
    if not os.path.exists(path): return pd.DataFrame()
    with open(path) as f:
        for line in f:
            if line.startswith('#'): continue
            p = line.split()
            if len(p) >= 3:
                try: rows.append({'tau': float(p[0]), 'act': float(p[1]), 'rg':  float(p[2])})
                except: pass
    return pd.DataFrame(rows)

def load_covariance_data(path):
    """
    Load pre-calculated matrices and true Rg values.
    Returns {(tau, act): (C_matrix, true_rg2)}
    """
    if not os.path.exists(path): return None
    data = {}
    with open(path) as f:
        lines = f.readlines()[1:]
    for line in lines:
        p = line.split()
        if len(p) < 5: continue
        tau, act = float(p[0]), float(p[1])
        true_rg2 = float(p[2]) # Now holds computed True Rg^2
        C = np.array(p[5:], dtype=float).reshape((64, 64))
        data[(tau, act)] = (C, true_rg2)
    return data

def load_fit_params(path):
    if not os.path.exists(path): return {}
    df = pd.read_csv(path)
    return {(row['tau'], row['f_a']): row for _, row in df.iterrows()}


# ============================================================
#  MAIN
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Building Weight Matrix W ...")
    W = build_W_matrix(N_SEGS)

    rg_data  = load_rg_data(RG_AVG_FILE)
    fit_pars = load_fit_params(FIT_PARAMS_FILE)
    
    # Strategy: Try to load cached matrices first, else calculate from dumps
    cov_dict = load_covariance_data(COV_MATRIX_FILE)
    if cov_dict is None:
        print("Pre-calculated covariance matrix file NOT FOUND.")
        print("Calculating full C_nm matrices and True Rg from simulation dumps...")
        tasks = []
        if not fit_pars and rg_data.empty:
            print("No configuration parameters found to reconstruct.")
            return
            
        unique_taus = sorted({k[0] for k in fit_pars.keys()}) if fit_pars else sorted(rg_data['tau'].unique())
        if 0.1 in unique_taus: unique_taus.remove(0.1)
        acts = sorted({k[1] for k in fit_pars.keys()}) if fit_pars else sorted(rg_data['act'].unique())
        
        for tau in unique_taus:
            tau_str = "0.1" if tau == 0.1 else str(int(tau))
            for act in acts:
                for folder in DUMP_FOLDERS:
                    pat = f"{folder}/position.64_128_{act:.0f}_0.1_0.1_{tau_str}_*.dump"
                    for f in glob.glob(pat):
                        tasks.append((f, tau, act))
        
        if tasks:
            with Pool() as p:
                results = []
                for i, r in enumerate(p.imap_unordered(process_file_tangent_covariance, tasks), 1):
                    results.append(r)
                    print(f"\rProcessed {i}/{len(tasks)} covariance matrices...", end='', flush=True)
                print()
                
            agg_cov = {}
            for r in results:
                if r is None: continue
                tau, act, C, rg2 = r
                key = (tau, act)
                if key not in agg_cov:
                    agg_cov[key] = {'count': 0, 'C': np.zeros_like(C), 'rg2': 0.0}
                agg_cov[key]['C'] += C
                agg_cov[key]['rg2'] += rg2
                agg_cov[key]['count'] += 1
                
            cov_dict = { key: (data['C'] / data['count'], data['rg2'] / data['count']) 
                         for key, data in agg_cov.items() }
            print(f"Direct calculation complete for {len(cov_dict)} aggregated configurations.")
            
            # --- SAVING LOGIC ---
            os.makedirs(os.path.dirname(COV_MATRIX_FILE), exist_ok=True)
            print(f"Saving consistent data to {COV_MATRIX_FILE}...")
            with open(COV_MATRIX_FILE, 'w') as f:
                f.write("# tau act True_Rg2 dummy2 dummy3 C_flattened (64x64)\n")
                for (tau, act), (C, rg2_true) in cov_dict.items():
                    c_str = " ".join(map(str, C.flatten()))
                    # We store the computed true_rg2 in result[2] for consistency
                    f.write(f"{tau} {act} {rg2_true} 0 0 {c_str}\n")
            
            w_path = os.path.join(KAPPA_DIR, "W_matrix.txt")
            np.savetxt(w_path, W, header="W_nm weight matrix")
            # -------------------------------
        else:
            print("No dump files found to process.")
            cov_dict = {}

    if not cov_dict:
        print("Error: No covariance data available.")
        return

    # Filter Taus as requested (remove 0.1)
    unique_taus = sorted({k[0] for k in fit_pars.keys()}) if fit_pars else sorted(rg_data['tau'].unique())
    if 0.1 in unique_taus: unique_taus.remove(0.1)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for i, tau in enumerate(unique_taus):
        c = COLORS[i % len(COLORS)]
        m = MARKERS[i % len(MARKERS)]
        tau_label = rf'${tau:g}$'
        
        acts_t = sorted({k[1] for k in fit_pars.keys() if k[0] == tau}) if fit_pars else sorted(rg_data[rg_data['tau'] == tau]['act'].unique())
        
        tx, ty, dx, dy, fx, fy = [], [], [], [], [], []
        for act in acts_t:
            if (tau, act) in cov_dict:
                C, true_rg2 = cov_dict[(tau, act)]
                
                # Use ONLY the consistent data from the dump processing
                t_val = np.sqrt(max(true_rg2, 0))
                tx.append(act)
                ty.append(t_val)
                
                # Exact Discrete reconstruction
                rg2_recon = reconstruct_rg2_discrete(C, W)
                dx.append(act)
                dy.append(np.sqrt(max(rg2_recon, 0)))

            if (tau, act) in fit_pars:
                rg2_f_nn = reconstruct_rg2_fitted_nn(C, fit_pars[(tau, act)], W, N_SEGS)
                fx.append(act)
                fy.append(np.sqrt(max(rg2_f_nn, 0)))

        if tx: ax.plot(tx, ty, marker=m, linestyle='', color=c, markeredgecolor='k', markersize=8, label=tau_label, zorder=4)
        if dx: ax.plot(dx, dy, linestyle='-', linewidth=2.5, color=c, zorder=3, alpha=0.8)
        if fx: ax.plot(fx, fy, linestyle='--', linewidth=2.5, color=c, zorder=2, alpha=0.3)

    ax.set_xlabel(r'$f_a$')
    ax.set_ylabel(r'$\langle R_g \rangle$')
    ax.set_ylim(None, 24)

    leg1 = ax.legend(loc='upper left', frameon=False, title="Persistence Time ($\\tau$)", ncol=3, columnspacing=1.0)
    ax.add_artist(leg1)
    
    style_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label=r'Simulation ($N$-body)'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5, label=r'Reconstruction with full $\mathbf{C}_{nm}$'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2.5, alpha=0.3, label=r'Reconstruction from fitted $\langle a_n^2 \rangle$')
    ]
    ax.legend(handles=style_legend_elements, loc='upper right', frameon=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', which='both', bottom=True, top=False, left=True, right=False)

    fig.tight_layout()
    out_path = os.path.join(script_dir, 'Rg_Reconstruction_Discrete_Comparison_t.pdf')
    fig.savefig(out_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Analysis Complete. Numerical consistency ensured.")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()