# Research Paper: Mode Analysis of Active Polymers

This repository contains the data, analysis pipeline, and publication plots for research on active polymers with varying stiffness ($\kappa=32, 128$).

## Scientific Background & Procedure

### Motivation
In active matter, the concept of a single "effective temperature" often fails because nonequilibrium forcing is distributed non-uniformly across different length and time scales. This study uses a **semiflexible polymer as a multiscale probe**. By decomposing the polymer's conformation into normal modes, we can resolve how an active bath (either an explicit bath of Active Brownian Particles or an implicit colored-noise model) selectively couples to different internal degrees of freedom.

### Core Procedure
1.  **Simulation**: We perform underdamped Langevin dynamics of a bead-spring polymer in a bath of self-propelled particles (ABPs).
2.  **Mode Extraction**: We extract tangent bond vectors and project them into a discrete cosine basis to find mode amplitudes $a_n$.
3.  **Spectral Analysis**: We calculate the steady-state variance $\langle a_n^2 \rangle$ for each mode.
4.  **Theoretical Fitting**: We fit the variance spectrum to a reduced colored-noise theory to extract effective persistence times and forcing strengths.
5.  **Reconstruction**: We use the full mode covariance matrix $C_{nm} = \langle a_n a_m \rangle$ to reconstruct global observables like the Radius of Gyration ($R_g$).

### Key Finding
Activity does **not** act as a uniform heat source. It selectively enhances **low-mode** (long-wavelength) fluctuations. The "persistence time" ($\tau$) of the bath acts as a spectral selection parameter: increasing $\tau$ shifts the spectral weight toward progressively longer wavelengths.

---

## Directory Structure
- `simulation_setup/`: LAMMPS input scripts and cluster submission automation.
- `kappa32/` & `kappa128/`: System-specific folders.
    - `data/`: Processed numerical datasets.
    - `plots/`: Final PDF figures used in the paper.
    - `code/`: Sequential Python scripts and `run_pipeline.sh`.

---

## Data Files Guide (Column Definitions)

### 1. Mode Fit Parameters (`MSD_Fit_Parameters.csv`)
- `f_a`: Active force magnitude.
- `tau`: Persistence time ($\tau$).
- `A`: Passive fluctuation amplitude ($k_B T / \kappa$).
- `C`: Effective tension-like parameter (suppresses low modes).
- `D`: Active enhancement strength.
- `E`: Coupling between persistence and mode relaxation.

### 2. Mode Variances (`tangent_mode_variances.txt`)
- `Tau`: Persistence time.
- `Act`: Active force ($f_a$).
- `n`: Mode number ($n=0$ is the zeroth/base mode, $n=1 \dots 63$ are bending modes).
- `Variance`: Measured steady-state variance $\langle a_n^2 \rangle$.

### 3. Bond Statistics
- **`bondlength_averaged.txt`**:
    - `Tau`, `Activity`, `Average_BondLength`: Mean length $\langle b \rangle$ of harmonic bonds.
- **`bond_fluctuation_variances.txt`**:
    - `Tau`, `Act`, `p`, `Variance`: Variance of the $p$-th internal bond-stretching mode.

### 4. Global Observables (`rg_averaged_from_folder.txt`)
- `Tau`, `Activity`, `Average_Rg`: Ensemble-averaged $R_g$ measured directly from simulation.

### 5. Covariance & Weights
- **`W_matrix.txt`**: The exact geometric weight matrix $W_{nm}$ used to map mode correlations back to $R_g^2$.
- **`covariance_reconstruction/tangent_covariance_matrices.txt`**:
    - `tau`, `act`: Parameters.
    - `True_Rg2`: The $R_g^2$ calculated from the coordinates.
    - `dummy2`, `dummy3`: Placeholders.
    - `C_flattened`: The full $64 \times 64$ covariance matrix $C_{nm}$, stored as a flattened row of 4096 values.

---

## How to Reproduce Results

1.  **Navigate to a system**: `cd kappa32/code`
2.  **Run the pipeline**: `./run_pipeline.sh`
3.  **View Plots**: The figures will be generated/updated in the `../plots/` directory.

## Simulation Setup
To regenerate raw data:
1. Ensure `in.explicitactivebathpolymer.lmp` (template) is present in your simulation directory.
2. Run `./generate_pre.sh` to create parameterized LAMMPS input files.
3. Run `python3 binpack_commands.py` to group simulations into efficient 32-core batches.
4. Run `./run_jobs.sh` to generate PBS scripts and submit to your cluster.
