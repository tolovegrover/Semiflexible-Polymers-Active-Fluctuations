"""
plot_rg_reconstruction_comparison.py
=====================================
Single publication-quality figure: all taus on one plot, x = activity.

Three series per tau (same color, different line/marker style):
  1. True R_g       -- filled circles, no line (simulation data)
  2. Discrete exact -- solid line + square markers  (full C_nm × W_nm)
  3. Fitted full    -- dashed line + triangle markers
                       (C_nn replaced by active_func fit, off-diagonals kept)

PORTABILITY: change the four CONFIG paths below for kappa32 or any system.
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
BOND_LENGTH_FILE = os.path.join(KAPPA_DIR, "bondlength_averaged.txt")

N_SEGS = 63   # number of bond segments (beads - 1)

# ============================================================
#  PUBLICATION AESTHETICS (TWO-COLUMN FORMAT)
# ============================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif":  ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size":       12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth":  1.0,
    "lines.linewidth": 1.5,
})

# Standard double-column width (7.1 inches). Adjusted height for a crisp aspect ratio.
FIG_W, FIG_H = 7,5.3  

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


def reconstruct_rg2_fitted_full(C_measured, fit_row, W, N=63):
    """
    FITTED FULL:
    Keep C_00 and off-diagonals, replace C_nn (n>=1) with fitted active_func.
    """
    A   = float(fit_row['A'])
    C_p = float(fit_row['C'])
    D   = float(fit_row['D'])
    E   = float(fit_row['E'])

    C_hybrid = C_measured.copy()
    for n in range(1, N + 1):   
        C_hybrid[n, n] = float(active_func(n, A, C_p, D, E))
    return float(np.sum(C_hybrid * W))

def reconstruct_rg2_fitted_nn(C_measured, fit_row, W, b_ratio=1.0, N=63):
    """
    FITTED NN ONLY:
    C_nn replaced by active_func fit, off-diagonals are 0.
    """
    A   = float(fit_row['A'])
    C_p = float(fit_row['C'])
    D   = float(fit_row['D'])
    E   = float(fit_row['E'])

    C_fit = np.zeros((N + 1, N + 1))
    
    for n in range(1, N + 1):   
        C_fit[n, n] = float(active_func(n, A, C_p, D, E)) * (b_ratio**2)
        
    C_fit[0, 0] = C_measured[0, 0]
    
    return float(np.sum(C_fit * W))


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

def load_bond_lengths(path):
    data = {}
    if not os.path.exists(path): return data
    with open(path) as f:
        for line in f:
            if line.startswith('#'): continue
            p = line.split()
            if len(p) >= 3:
                try:
                    data[(float(p[0]), float(p[1]))] = float(p[2])
                except ValueError:
                    pass
    return data


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
    bond_data = load_bond_lengths(BOND_LENGTH_FILE)

    unique_taus = sorted(rg_data['tau'].unique())
    cmap   = plt.get_cmap('jet')
    n_taus = len(unique_taus)

    def collect_series(tau, acts):
        b_thermal = bond_data.get((tau, 0))

        true_x, true_y = [], []
        disc_x, disc_y = [], []
        fit_x,  fit_y  = [], []
        fit_nn_x, fit_nn_y = [], []
        fit_bond_x, fit_bond_y = [], []
        for act in sorted(acts):
            row = rg_data[(rg_data['tau'] == tau) & (rg_data['act'] == act)]
            if not row.empty:
                true_x.append(act)
                true_y.append(float(row['rg'].values[0]))
            if (tau, act) in cov_data:
                _, C = cov_data[(tau, act)]
                disc_x.append(act)
                disc_y.append(np.sqrt(max(reconstruct_rg2_discrete(C, W), 0)))
            if (tau, act) in fit_pars and (tau, act) in cov_data:
                _, C = cov_data[(tau, act)]
                rg2_f = reconstruct_rg2_fitted_full(C, fit_pars[(tau, act)], W, N_SEGS)
                fit_x.append(act)
                fit_y.append(np.sqrt(max(rg2_f, 0)))
                
                # 1. Standard fitted curve without artificial bond scaling
                rg2_f_nn = reconstruct_rg2_fitted_nn(C, fit_pars[(tau, act)], W, b_ratio=1.0, N=N_SEGS)
                fit_nn_x.append(act)
                rg_f_nn = np.sqrt(max(rg2_f_nn, 0))
                fit_nn_y.append(rg_f_nn)
                
                # 2. Bond-scaled hypothesis curve (only scaling n>=1 modes)
                if b_thermal is not None and (tau, act) in bond_data:
                    b_act = bond_data[(tau, act)]
                    b_ratio = b_act / b_thermal
                    rg2_f_nn_scaled = reconstruct_rg2_fitted_nn(C, fit_pars[(tau, act)], W, b_ratio=b_ratio, N=N_SEGS)
                    fit_bond_x.append(act)
                    fit_bond_y.append(np.sqrt(max(rg2_f_nn_scaled, 0)))
                
        return (true_x, true_y), (disc_x, disc_y), (fit_x, fit_y), (fit_nn_x, fit_nn_y), (fit_bond_x, fit_bond_y)

    # -------------------------------------------------------
    # TWO-COLUMN FIGURE: all taus, x = activity
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for i, tau in enumerate(unique_taus):
        color  = cmap(i / max(1, n_taus - 1))
        acts_t = sorted(rg_data[rg_data['tau'] == tau]['act'].unique())
        (tx, ty), (dx, dy), (fx, fy), (fnn_x, fnn_y), (fbond_x, fbond_y) = collect_series(tau, acts_t)

        # Series 1: True data — filled circles, NO line, distinct black edge
        ax.plot(tx, ty, 'o', color=color, markersize=8, linewidth=0,
                markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)

        # Series 2: Discrete exact — solid line + square markers
        if dx:
            ax.plot(dx, dy, '-s', color=color, markersize=8,
                    linewidth=2.5, markeredgecolor='black', markeredgewidth=0.5, alpha=0.8)

        # Series 3: Fitted full — dashed line + triangle markers
        if fx:
            ax.plot(fx, fy, '--^', color=color, markersize=8,
                    linewidth=2.5, markeredgecolor='black', markeredgewidth=0.5, alpha=0.8)
                    
        # Series 4: Fitted NN only - dashed line + no markers, light color (alpha=0.3)
        if fnn_x:
            ax.plot(fnn_x, fnn_y, '--', color=color, linewidth=2.5, alpha=0.3)

        # Series 5: Thermal Rg scaled by bond stretch - dotted line + star markers
        if fbond_x:
            ax.plot(fbond_x, fbond_y, ':', color=color, markersize=8,
                    linewidth=2.5, marker='*', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)

    ax.set_xlabel(r'$f_a$')
    ax.set_ylabel(r'$\langle R_g \rangle$')
    ax.set_ylim(None, 22)

    # 1. Methodology Legend (Upper Left)
    # style_handles = [
    #     Line2D([0], [0], marker='o', color='k', lw=0, markersize=10, 
    #            markeredgecolor='black', markeredgewidth=0.5, label=r'Simulation ($N$-body)'),
    #     Line2D([0], [0], marker='s', color='k', lw=2.5, ls='-', markersize=10, 
    #            markeredgecolor='black', markeredgewidth=0.5, label=r'Reconstruction with full $\mathbf{C}_{nm}$'),
    #     Line2D([0], [0], marker='^', color='k', lw=2.5, ls='--', markersize=10, 
    #            markeredgecolor='black', markeredgewidth=0.5, label=r'Reconstruction from fitted $\langle a_n^2 \rangle$'),
    #     Line2D([0], [0], marker='*', color='k', lw=2.5, ls=':', markersize=10, 
    #            markeredgecolor='black', markeredgewidth=0.5, label=r'$R_{g,\mathrm{fit}}(f_a) \times \langle b(f_a) \rangle / \langle b(0) \rangle$')
    # ]
    # leg_style = ax.legend(handles=style_handles, loc='upper right',
    #                       frameon=False, fontsize=12)

    # 2. Parameter Legend (Upper Right, split into 2 columns)
    tau_handles = [Line2D([0], [0], color=cmap(i / max(1, n_taus - 1)), lw=2.5,
                          label=rf'$\tau={tau:g}$')
                   for i, tau in enumerate(unique_taus)]
    ax.legend(handles=tau_handles, loc='upper center',
              ncol=int(n_taus/2 +1),  # <--- Set columns equal to the number of items
              frameon=False, fontsize=12)
    #ax.add_artist(leg_style) # Re-add the methodology legend

    # Clean up axes and add a subtle grid for readability
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', which='both', bottom=True, top=False, left=True, right=False)
    ax.grid(True, linestyle=':', alpha=0)

    fig.tight_layout()

    out_path = os.path.join(out_dir, 'Rg_Reconstruction_AllTaus_BondStretch.pdf')
    fig.savefig(out_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    print("Done.")

if __name__ == "__main__":
    main()