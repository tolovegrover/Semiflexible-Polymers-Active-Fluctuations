import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# --- 1. Global LaTeX & Publication Aesthetics ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2
})

# Physical Figure Size (APS 1 quadrant)
fig_width = 3.375
fig_height = 2.6

# --- Data Processing ---
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    data.append({
                        'Tau': float(parts[0]),
                        'Activity': float(parts[1]),
                        'n': int(parts[2]),
                        'Variance': float(parts[3])
                    })
                except ValueError: pass
    return pd.DataFrame(data)

# --- Fitting Functions ---
def passive_func(x, A, B):
    return A / (x**2 + B)

def active_func(x, A, C, D, E):
    return (A / (x**2 + C)) * (1 + D / (1 + E**2 * (x**4 + C * x**2)))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "tangent_mode_variances.txt")
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return
        
    df = load_data(data_file)
    
    if df.empty or 'n' not in df.columns:
        print("No valid variance data found. Please ensure raw dump files exist in the position folder.")
        return
        
    # We fit over n >= 1, max_modes is typically 32 in the MATLAB script. Let's use up to 32 here.
    max_modes = 32
    df = df[(df['n'] >= 1) & (df['n'] <= max_modes)]
    
    unique_taus = sorted(df['Tau'].unique())
    unique_acts = sorted(df['Activity'].unique())
    
    fit_results = []
    
    # --- Batch Process Fits ---
    grouped = df.groupby(['Tau', 'Activity'])
    data_struct = {}
    
    for tau in unique_taus:
        # Step 1: Fit passive (Activity = 0) to get A and B
        try:
            pass_data = grouped.get_group((tau, 0))
            x_pass = pass_data['n'].values
            y_pass = pass_data['Variance'].values
            
            # Weights: 1 / y^2
            sigma_pass = y_pass
            popt_pass, _ = curve_fit(passive_func, x_pass, y_pass, p0=[1.0, 0.1], 
                                     bounds=([1e-5, 1e-5], [np.inf, np.inf]), sigma=sigma_pass, absolute_sigma=False)
            A_val, B_val = popt_pass
            
            x_fit = np.logspace(0, np.log10(max_modes), 200)
            y_fit_pass = passive_func(x_fit, A_val, B_val)
            
            data_struct[(tau, 0)] = {
                'x': x_pass, 'y': y_pass, 'x_fit': x_fit, 'y_fit': y_fit_pass,
                'A': A_val, 'B': B_val, 'C': np.nan, 'D': np.nan, 'E': np.nan
            }
        except KeyError:
            print(f"Warning: No passive data (Activity=0) for tau={tau}")
            continue
            
        fit_results.append({
            'f_a': 0.0, 'tau': tau, 'A': A_val, 'C': B_val, 'D': 0.0, 'E': 0.0
        })
            
        # Step 2: Fit active (Activity > 0) keeping A fixed
        last_p0 = [B_val, 10.0, 0.05] # Initial guess for the first active case
        for act in unique_acts:
            if act == 0: continue
            try:
                act_data = grouped.get_group((tau, act))
                x_act = act_data['n'].values
                y_act = act_data['Variance'].values
                
                # Wrapper to fix A
                def act_func_fixed_A(x, C, D, E):
                    return active_func(x, A_val, C, D, E)
                    
                sigma_act = y_act
                popt_act, _ = curve_fit(act_func_fixed_A, x_act, y_act, 
                                        p0=last_p0,
                                        bounds=([1e-5, 0, -np.inf], [np.inf, np.inf, np.inf]),
                                        sigma=sigma_act, absolute_sigma=False, maxfev=100000)
                C_val, D_val, E_val = popt_act
                last_p0 = [C_val, D_val, E_val] # Use successful fit as guess for next act
                
                y_fit_act = active_func(x_fit, A_val, C_val, D_val, E_val)
                
                data_struct[(tau, act)] = {
                    'x': x_act, 'y': y_act, 'x_fit': x_fit, 'y_fit': y_fit_act,
                    'A': A_val, 'B': np.nan, 'C': C_val, 'D': D_val, 'E': E_val
                }
                
                fit_results.append({
                    'f_a': act, 'tau': tau, 'A': A_val, 'C': C_val, 'D': D_val, 'E': E_val
                })
            except KeyError:
                continue
            except Exception as e:
                print(f"Fit failed for tau={tau}, act={act}: {e}")
                
    # Save parameters
    if fit_results:
        results_df = pd.DataFrame(fit_results)
        results_df.to_csv(os.path.join(script_dir, "MSD_Fit_Parameters.csv"), index=False)
        print("Exported fitting parameters to MSD_Fit_Parameters.csv")
    
    # --- Generate Plots ---
    # Turbo colormap equivalent for older matplotlib
    cmap = plt.get_cmap('jet')
    
    # 1. Fixed Tau, Varying Activity
    for tau in unique_taus:
        tau_acts = [act for act in unique_acts if (tau, act) in data_struct]
        if len(tau_acts) <= 1: continue # Only has passive or none
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Plot Passive
        # Plot Passive
        if (tau, 0) in data_struct:
            d = data_struct[(tau, 0)]
            plt.plot(d['x'], d['y'], 'sk', markerfacecolor='k', markersize=4, label='0')
            plt.plot(d['x_fit'], d['y_fit'], '-k')
            
        allowed_acts = [8, 16, 32, 64, 96]
        active_acts = [a for a in tau_acts if a > 0 and a in allowed_acts]
        
        for i, act in enumerate(active_acts):
            color = cmap(i / max(1, len(active_acts) - 1))
            d = data_struct[(tau, act)]
            label = f'${act:g}$'
            plt.plot(d['x'], d['y'], 'o', color=color, markerfacecolor=color, markersize=4, label=label)
            plt.plot(d['x_fit'], d['y_fit'], '-', color=color)
            
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'Mode number ($n$)')
        plt.ylabel(r'$\displaystyle \langle a_n^2 \rangle$')
        
        # Tau text
        plt.text(0.95, 0.95, rf'$\tau = {tau:g}$', transform=plt.gca().transAxes,
                 ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
                 
        leg = plt.legend(loc='lower left', ncol=3, frameon=False, title=r"Activity $f_a$", fontsize=8, columnspacing=1.0)
        leg.get_title().set_fontweight('bold')
        
        # Styling tweaks
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', which='both')
        
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, f'Plot_FixedTau_{tau:g}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

    # 2. Fixed Activity, Varying Tau
    for act in unique_acts:
        if act == 0: continue
        act_taus = [tau for tau in unique_taus if (tau, act) in data_struct and tau != 0.1]
        if not act_taus: continue
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Plot arbitrary passive as reference
        tau_ref = act_taus[0]
        if (tau_ref, 0) in data_struct:
             d = data_struct[(tau_ref, 0)]
             plt.plot(d['x'], d['y'], 'sk', markerfacecolor='0.7', markeredgecolor='none', markersize=4, label='0')
             plt.plot(d['x_fit'], d['y_fit'], '--k')
             
        for i, tau in enumerate(act_taus):
            color = cmap(i / max(1, len(act_taus) - 1))
            d = data_struct[(tau, act)]
            label = rf'${tau:g}$'
            plt.plot(d['x'], d['y'], 'o', color=color, markerfacecolor=color, markersize=4, label=label)
            plt.plot(d['x_fit'], d['y_fit'], '-', color=color)
            
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'Mode number ($n$)')
        plt.ylabel(r'$\displaystyle \langle a_n^2 \rangle$')
        
        # Activity text
        plt.text(0.95, 0.95, rf'$f_a = {act:g}$', transform=plt.gca().transAxes,
                 ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
                 
        leg = plt.legend(loc='lower left', ncol=3, frameon=False, title=r"Persistence Time $\tau$", fontsize=8, columnspacing=1.0)
        leg.get_title().set_fontweight('bold')
        
        # Styling tweaks
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', which='both')
        
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, f'Plot_FixedAct_{act:g}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

    print("Success! All PDFs generated.")

if __name__ == "__main__":
    main()
