# Exact Discrete Reconstruction of the Radius of Gyration ($R_g$)

## Overview
These scripts evaluate the mean squared radius of gyration $\langle R_g^2 \rangle$ for a discrete polymer chain under active fluctuations. The pipeline verifies that the spatial extent measured directly in $N$-body simulations matches the analytical reconstruction obtained by projecting the mode covariance matrix $C_{nm} = \langle \mathbf{a}_n \cdot \mathbf{a}_m \rangle$ onto an exact geometric weight matrix $W_{nm}$.

The pipeline compares two independent evaluations:
1. **Direct Simulation $\langle R_g \rangle$**: Extracted directly from bead coordinates $\mathbf{r}_i(t)$.
2. **Discrete Exact Reconstruction**: Evaluated from the full measured mode covariance matrix via:
   $$\langle R_g^2 \rangle = \sum_{n=0}^{N} \sum_{m=0}^{N} C_{nm} W_{nm}$$

---

## The Discrete Weight Matrix ($W$)

The exact geometric weight matrix is constructed in `build_W_matrix(N)`:
1. **Bond-to-Position Operator ($S$)**: Size $(N+1) \times N$ mapping bond vectors $\mathbf{b}_i = \mathbf{r}_{i+1} - \mathbf{r}_i$ to bead positions relative to $\mathbf{r}_0$:
   $$S_{ki} = 1 \quad \text{for } i < k, \quad 0 \text{ otherwise}$$
2. **Centering Operator ($C_{\text{op}}$)**: Projects coordinates relative to the center of mass:
   $$C_{\text{op}} = I - \frac{1}{N+1} \mathbf{1}\mathbf{1}^T$$
3. **Bond-Space Variance Operator ($M$)**:
   $$M = \frac{1}{N+1} S^T C_{\text{op}} S$$
4. **Mode-Space Weight Matrix ($W$)**: Transformed using the Discrete Cosine Transform operator basis ($T = \text{pinv}(\text{DCT})$):
   $$W = T^T M T$$

---

## Reconstruction Methods

1. **Discrete Exact**:
   Evaluates $\sum_{nm} C_{nm} W_{nm}$ using the full measured covariance matrix $C_{nm}$.
2. **Fitted Full**:
   Keeps the measured off-diagonal mode covariances $C_{nm}$ and $C_{00}$, but replaces diagonal bending modes $C_{nn}$ ($n \ge 1$) with the theoretical active-bath fit function.
3. **Fitted Diagonal Only**:
   Sets off-diagonal elements to zero ($C_{nm} = 0$ for $n \ne m$), testing the contribution of inter-mode correlations to the polymer coil size.

---

## Execution
Run `06_reconstruct_rg.py` to perform the reconstruction and generate comparison figures, and `08_verify_numerics.py` to check numerical consistency between direct and reconstructed values.