"""
plot_rg_reconstruction_combined_style.py
========================================
Calculations: Exact discrete geometry and purely fitted parameter reconstructions.
Aesthetics: Modern colorblind-safe palette, Stix fonts, dual legends.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.lines import Line2D

# ============================================================
#  CONFIG  --  change only these paths to port to another kappa
# ============================================================
KAPPA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

RG_AVG_FILE     = os.path.join(KAPPA_DIR, "rg_averaged_from_folder.txt")
COV_MATRIX_FILE = os.path.join(KAPPA_DIR, "covariance_reconstruction",
                                "tangent_covariance_matrices.txt")
FIT_PARAMS_FILE = os.path.join(KAPPA_DIR, "MSD_Fit_Parameters.csv")

N_SEGS = 63   # number of bond segments (beads - 1)

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

FIG_W, FIG_H = 7.0, 5.3  # Adjusted slightly for dual-legend breathing room

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


def active_func(n, A, C_p, D, E):
    """
    Fitted active-spectrum model.
    """
    return (A / (n**2 + C_p)) * (1.0 + D / (1.0 + E**2 * (n**4 + C_p * n**2)))


def reconstruct_rg2_discrete(C, W):
    """DISCRETE EXACT: Rg^2 = sum_nm  C_nm * W_nm."""
    return float(np.sum(C * W))


def reconstruct_rg2_fitted_pure(fit_row, W, C00_measured, N=63):
    """
    FITTED PURE:
    - Assumes zero cross-correlation (off-diagonals = 0).
    - C_nn (n >= 1) generated strictly from the fitted active_func.
    - C_00 is provided as the boundary baseline.
    """
    A   = float(fit_row['A'])
    C_p = float(fit_row['C'])
    D   = float(fit_row['D'])
    E   = float(fit_row['E'])

    # Start with a completely empty (decoupled) matrix
    C_pure = np.zeros((N + 1, N + 1))
    
    # 1. Slot in the base mode (the fit doesn't cover n=0)
    C_pure[0, 0] = C00_measured
    
    # 2. Generate bending modes strictly from the theoretical fit
    for n in range(1, N + 1):   
        C_pure[n, n] = float(active_func(n, A, C_p, D, E))
        
    # Multiply by the exact discrete geometric weights
    return float(np.sum(C_pure * W))


# ============================================================
#  DATA LOADERS
# ============================================================

def load_rg_data(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'): continue
            p = line.split()
            if len(p) >= 3:
                try:
                    rows.append({'tau': float(p[0]),
                                 'act': float(p[1]),
                                 'rg':  float(p[2])})
                except ValueError:
                    pass
    return pd.DataFrame(rows)

def load_covariance_matrices(path):
    data = {}
    with open(path) as f:
        lines = f.readlines()[1:]
    for line in lines:
        p = line.split()
        if len(p) < 5: continue
        tau, act = float(p[0]), float(p[1])
        true_rg2 = float(p[2])
        C = np.array(p[5:], dtype=float).reshape((64, 64))
        data[(tau, act)] = (true_rg2, C)
    return data

def load_fit_params(path):
    df = pd.read_csv(path)
    return {(row['tau'], row['f_a']): row for _, row in df.iterrows()}


# ============================================================
#  MAIN
# ============================================================

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print("Building W matrix ...")
    W = build_W_matrix(N_SEGS)

    rg_data  = load_rg_data(RG_AVG_FILE)
    cov_data = load_covariance_matrices(COV_MATRIX_FILE)
    fit_pars = load_fit_params(FIT_PARAMS_FILE)

    unique_taus = sorted(rg_data['tau'].unique())
    n_taus = len(unique_taus)

    def collect_series(tau, acts):
        true_x, true_y = [], []
        disc_x, disc_y = [], []
        fit_x,  fit_y  = [], []
        for act in sorted(acts):
            row = rg_data[(rg_data['tau'] == tau) & (rg_data['act'] == act)]
            if not row.empty:
                true_x.append(act)
                true_y.append(float(row['rg'].values[0]))
            
            # 1. Full Matrix Reconstruction
            if (tau, act) in cov_data:
                _, C = cov_data[(tau, act)]
                disc_x.append(act)
                disc_y.append(np.sqrt(max(reconstruct_rg2_discrete(C, W), 0)))
                
            # 2. Pure Fitted Reconstruction
            if (tau, act) in fit_pars and (tau, act) in cov_data:
                _, C = cov_data[(tau, act)]
                # Pass only the fit row, the W matrix, and the C00 baseline component
                rg2_f = reconstruct_rg2_fitted_pure(fit_pars[(tau, act)], W, C[0, 0], N_SEGS)
                fit_x.append(act)
                fit_y.append(np.sqrt(max(rg2_f, 0)))
                
        return (true_x, true_y), (disc_x, disc_y), (fit_x, fit_y)

    # -------------------------------------------------------
    # FIGURE PLOTTING
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for i, tau in enumerate(unique_taus):
        c = COLORS[i % len(COLORS)]
        m = MARKERS[i % len(MARKERS)]
        tau_label = rf'${tau:g}$'
        
        acts_t = sorted(rg_data[rg_data['tau'] == tau]['act'].unique())
        (tx, ty), (dx, dy), (fx, fy) = collect_series(tau, acts_t)

        # 1. True data — Scattered points, distinct black edge
        if tx:
            ax.plot(tx, ty, marker=m, linestyle='', color=c, 
                    markeredgecolor='k', markersize=8, label=tau_label, zorder=4)

        # 2. Discrete exact (Tangent equivalent) — Dashed line
        if dx:
            ax.plot(dx, dy, linestyle='--', linewidth=2.5, color=c, zorder=3, alpha=0.8)

        # 3. Fitted pure (Fitted Theory equivalent) — Solid line
        if fx:
            ax.plot(fx, fy, linestyle='-', linewidth=2.5, color=c, zorder=2)

    ax.set_xlabel(r'$f_a$')
    ax.set_ylabel(r'$\langle R_g \rangle$')
    ax.set_ylim(None, 25)

    # --- Dual Legends ---
    # Legend 1: Parameter Legend (Tau) - Tied to the scatter labels automatically
    leg1 = ax.legend(loc='upper left', frameon=False, title="Persistence Time ($\\tau$)", ncol=3, columnspacing=1.0)
    ax.add_artist(leg1)
    
    # Legend 2: Methodology Legend (Styles)
    style_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='Simulated'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2.5, label='Discrete Exact'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5, label='Fitted Theory (Pure)')
    ]
    leg2 = ax.legend(handles=style_legend_elements, loc='upper right',frameon=True)

    # Clean up axes and add a subtle grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', which='both', bottom=True, top=False, left=True, right=False)
    ax.grid(False) # Set to True if you want a subtle background grid

    fig.tight_layout()

    out_path = os.path.join(out_dir, 'Rg_Reconstruction_AllTaus_Professional.pdf')
    fig.savefig(out_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    print("Done.")

if __name__ == "__main__":
    main()