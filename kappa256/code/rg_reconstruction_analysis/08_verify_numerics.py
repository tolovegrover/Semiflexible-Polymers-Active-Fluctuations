import numpy as np
import os

def build_W_matrix(N=63):
    """
    Computes the exact W_{nm} coefficient matrix mapping mode covariances 
    <a_n . a_m> back to the true discrete Rg^2.
    """
    # Centering matrix C: C r = r - r_cm
    # r_{k} - r_0 = sum_{i=0}^{k-1} b_i
    S = np.zeros((N+1, N))
    for k in range(1, N+1):
        S[k, :k] = 1.0
            
    C_op = np.eye(N+1) - 1.0/(N+1) * np.ones((N+1, N+1))
    
    # Rg^2 = 1/(N+1) |C r|^2 = 1/(N+1) b^T (S^T C^T C r) b
    M = (1.0/(N+1)) * (S.T @ C_op @ S)
    
    # Modes Transformation Matrix T
    q = np.arange(0, N + 1)[:, None] 
    j = np.arange(N)[None, :]       
    DCT_matrix = np.cos(np.pi * q * (j + 0.5) / N) * (2.0 / N)
    
    T = np.linalg.pinv(DCT_matrix)
    
    # Mode weights W
    W = T.T @ M @ T
    return W

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_file = os.path.join(script_dir, "../../data/covariance_reconstruction/tangent_covariance_matrices.txt")
    
    if not os.path.exists(matrix_file):
        print(f"Error: {matrix_file} not found.")
        return
        
    N = 63
    W = build_W_matrix(N)
    
    # Header
    print(f"{'Tau':>6} | {'Act':>5} | {'Sim Rg2':>15} | {'Recon Rg2':>15} | {'Difference':>15}")
    print("-" * 68)
    
    with open(matrix_file, 'r') as f:
        lines = f.readlines()[1:] # Skip header
        
    for line in lines:
        parts = line.split()
        if len(parts) < 5: continue
        
        tau = float(parts[0])
        act = float(parts[1])
        true_rg2 = float(parts[2])
        
        # Matrix starts from parts[5]
        flat_C = np.array(parts[5:], dtype=float)
        C = flat_C.reshape((N+1, N+1))
        
        # Reconstruct Rg2 = sum_{nm} C_nm * W_nm
        recon_rg2 = np.sum(C * W)
        diff = true_rg2 - recon_rg2
        
        print(f"{tau:6.1f} | {act:5.0g} | {true_rg2:15.8f} | {recon_rg2:15.8f} | {diff:15.8e}")

if __name__ == "__main__":
    main()
