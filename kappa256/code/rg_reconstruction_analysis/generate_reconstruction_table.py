import numpy as np
import os
import pandas as pd

# ============================================================
#  CONFIG
# ============================================================
KAPPA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

RG_AVG_FILE     = os.path.join(KAPPA_DIR, "rg_averaged_from_folder.txt")
COV_MATRIX_FILE = os.path.join(KAPPA_DIR, "covariance_reconstruction", "tangent_covariance_matrices.txt")

N_SEGS = 63

def build_W_matrix(N=63):
    """Exact geometric weight matrix W = T^T M T."""
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
    return float(np.sum(C * W))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "verify_reconstruction.txt")
    
    # 1. Load Data
    rg_data = []
    if os.path.exists(RG_AVG_FILE):
        with open(RG_AVG_FILE) as f:
            for line in f:
                if line.startswith('#'): continue
                p = line.split()
                if len(p) >= 3:
                    rg_data.append({'tau': float(p[0]), 'act': float(p[1]), 'rg': float(p[2])})
    df_rg = pd.DataFrame(rg_data)
    
    cov_data = {}
    if os.path.exists(COV_MATRIX_FILE):
        with open(COV_MATRIX_FILE) as f:
            lines = f.readlines()[1:]
        for line in lines:
            p = line.split()
            if len(p) < 5: continue
            tau, act = float(p[0]), float(p[1])
            C = np.array(p[5:], dtype=float).reshape((N_SEGS+1, N_SEGS+1))
            cov_data[(tau, act)] = C
            
    W = build_W_matrix(N_SEGS)
    
    # 2. Process and Write
    unique_taus = sorted(df_rg['tau'].unique())
    
    with open(out_path, 'w') as fout:
        header = f"{'Tau':>5} | {'Act':>5} | {'Sim Rg':>10} | {'Discrete Rg':>12} | {'Difference':>10}\n"
        fout.write(header)
        fout.write("-" * len(header) + "\n")
        
        for tau in unique_taus:
            sub_rg = df_rg[df_rg['tau'] == tau].sort_values(by='act')
            for _, row in sub_rg.iterrows():
                act = row['act']
                sim_rg = row['rg']
                
                results = [f"{tau:5.1f}", f"{act:5.0f}", f"{sim_rg:10.4f}"]
                
                if (tau, act) in cov_data:
                    C = cov_data[(tau, act)]
                    rg_disc = np.sqrt(max(reconstruct_rg2_discrete(C, W), 0))
                    results.append(f"{rg_disc:12.4f}")
                    
                    err_disc = sim_rg - rg_disc
                    results.append(f"{err_disc:10.4f}")
                else:
                    results += [f"{'N/A':>12}", f"{'N/A':>10}"]
                    
                fout.write(" | ".join(results) + "\n")
                
    print(f"Comparison table generated (Sim vs Discrete): {out_path}")

if __name__ == "__main__":
    main()
