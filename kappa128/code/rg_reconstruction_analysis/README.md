# Algorithm: Exact Discrete Reconstruction of the Radius of Gyration (Rg​)
Overview

These scripts evaluate the mean squared radius of gyration ⟨Rg2​⟩ for a discrete polymer chain under active dynamics. Instead of relying on continuous integral approximations—which fail at finite scales due to the lack of discrete boundary conditions—this algorithm evaluates the polymer's spatial extent using exact finite-domain linear algebra.

The pipeline compares two independent evaluations of Rg​:

    Simulated ⟨Rg​⟩: Ground truth extracted directly from spatial coordinates.

    Discrete Exact Reconstruction: Calculated by projecting the full measured mode covariance matrix (Cnm​) onto an exact finite-geometry weight matrix (Wnm​).

Part 1: The Exact Discrete Weight Matrix (W)

The core of the algorithm resides in build_W_matrix(N). The goal is to calculate how much each discrete cosine mode pair (n,m) contributes to the physical size of the polymer.

Instead of evaluating complex nested piecewise trigonometric sums, the algorithm determines the spatial weights computationally via basis projection:

    Integration Matrix (S): We define an integration operator S of size (N+1)×N that maps bond vectors to bead positions. Ski​=1 for i<k, ensuring position rk​=∑i=0k−1​Δri​.

    Centering Operator (Cop​): To measure fluctuations around the center of mass, we define Cop​=I−N+11​J, where I is the identity matrix and J is a matrix of ones.

    Position-Space Weight Matrix (M): The un-transformed spatial variance matrix is formulated as:
    M=N+11​STCop​S

    Mode-Space Weight Matrix (W): We map M from physical bond space into mode space using the pseudo-inverse of the Discrete Cosine Transform matrix (T=pinv(DCT)). The exact geometric weight matrix is isolated as:
    W=TTMT

Part 2: Reconstruction Strategy
Method A: Discrete Exact Reconstruction

Implemented in reconstruct_rg2_discrete(C, W).

This method verifies the exact geometric theory against the raw simulation data. It takes the fully measured covariance matrix of the active modes, Cnm​=⟨an​⋅am​⟩, and performs an element-wise multiplication and global sum with the weight matrix:
⟨Rg2​⟩discrete​=n=0∑N​m=0∑N​Cnm​Wnm​

Because W enforces exact mathematical parity (e.g., zeroes out all coupling between even modes), this operation naturally filters the covariance matrix, outputting the exact structural volume.

Part 3: Outputs and Pipeline

The provided codes deploy this core algorithm to produce two aligned, professional outputs:

    Tabular Verification (verify_reconstruction.txt)

        Iterates through arrays of activity drives (fa​) and persistence times (τ).

        Generates a structured log computing the residuals/errors between the simulated Rg​ and the Discrete Exact Rg​.

    Publication Graphics (plot_rg_reconstruction_comparison_2.py)

        Plots ⟨Rg​⟩ as a function of fa​ using a standardized Stix font and colorblind-safe palette.

        Applies a legend layout to distinguish between the τ parameters (colors).