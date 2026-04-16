import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# --- Global Style settings ---
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

def active_func(x, A, C, D, E):
    return (A / (x**2 + C)) * (1 + D / (1 + E**2 * (x**4 + C * x**2)))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    sim_rg_path = os.path.join(script_dir, "rg_averaged_from_folder.txt")
    tangent_var_path = os.path.join(script_dir, "tangent_mode_variances.txt")
    fit_param_path = os.path.join(script_dir, "MSD_Fit_Parameters.csv")
    
    # 1. Read Simulated Rg
    sim_rg_data = {}
    if os.path.exists(sim_rg_path):
        with open(sim_rg_path, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.split()
                if not parts: continue
                try:
                    tau = float(parts[0])
                    act = float(parts[1])
                    rg = float(parts[2])
                    sim_rg_data[(tau, act)] = rg
                except Exception:
                    pass
                    
    # 2. Read Tangent Variances
    var_data = {}
    if os.path.exists(tangent_var_path):
        df = pd.read_csv(tangent_var_path, sep=r'\s+', comment='#', names=['Tau', 'Activity', 'n', 'Variance'])
        for _, row in df.iterrows():
            tau = row['Tau']
            act = row['Activity']
            n = int(row['n'])
            var = row['Variance']
            if (tau, act) not in var_data:
                var_data[(tau, act)] = {}
            var_data[(tau, act)][n] = var

    # 3. Read Fit Parameters
    fit_params = {}
    if os.path.exists(fit_param_path):
        fit_df = pd.read_csv(fit_param_path)
        for _, row in fit_df.iterrows():
            tau = row['tau']
            act = row['f_a']
            fit_params[(tau, act)] = {
                'A': row['A'], 'C': row['C'], 'D': row['D'], 'E': row['E']
            }

    b_len = 2.0**(1.0/6.0)
    #b_len = 1.0
    N_segments = 63
    L = float(N_segments) * b_len
    
    calc_rg_data_var = {}
    calc_rg_data_fit = {}
    
    # Calculate Rg from modes and from fit
    for (tau, act), modes in var_data.items():
        if 0 not in modes:
            continue
            
        var_a0 = modes[0]
        base_rg2 = var_a0 * (N_segments**2) / 12.0
        
        rg2_var = base_rg2
        rg2_fit = base_rg2
        
        has_fit = (tau, act) in fit_params
        if has_fit:
            fp = fit_params[(tau, act)]
        
        for n in range(1, 64):
            if n not in modes:
                continue
                
            term_pref = (N_segments**2) / (2.0 * (n * np.pi)**2)
            #discrete_eigenvalue = 4.0 * (N_segments**2) * (np.sin(np.pi * n / (2.0 * N_segments))**2)
            #term_pref = (N_segments**2) / (2.0 * discrete_eigenvalue)
            correction = 1.0 - 8.0 / ((n * np.pi)**2) if n % 2 == 1 else 1.0
            In = term_pref * correction
            
            # From Variance
            rg2_var += modes[n] * In
            
            # From Fit
            if has_fit:
                var_fit_n = active_func(n, fp['A'], fp['C'], fp['D'], fp['E'])
                rg2_fit += var_fit_n * In
                
        calc_rg_data_var[(tau, act)] = np.sqrt(rg2_var)
        if has_fit:
            calc_rg_data_fit[(tau, act)] = np.sqrt(rg2_fit)
            
    # Compile results into a dataframe for easier plotting
    plot_data = []
    taus = sorted(list(set(k[0] for k in sim_rg_data.keys())))
    if 0.1 in taus:
        taus.remove(0.1)
    for tau in taus:
        acts = sorted(list(set(k[1] for k in sim_rg_data.keys() if k[0] == tau)))
        for act in acts:
            r_sim = sim_rg_data.get((tau, act), np.nan)
            r_var = calc_rg_data_var.get((tau, act), np.nan)
            r_fit = calc_rg_data_fit.get((tau, act), np.nan)
            plot_data.append({'Tau': tau, 'Act': act, 'Rg_Sim': r_sim, 'Rg_Var': r_var, 'Rg_Fit': r_fit})
            
    df_plot = pd.DataFrame(plot_data)
    
    # Save the expanded table
    table_path = os.path.join(script_dir, "Rg_Comparison_Table_3Sources.csv")
    df_plot.to_csv(table_path, index=False, float_format="%.6f")

    # --- Create Scatter Plot comparing the 3 sources ---
    # Modern colorblind-safe palette
    colors = ["#000000", "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    for i, tau in enumerate(taus):
        sub_df = df_plot[df_plot['Tau'] == tau].sort_values(by='Act')
        if sub_df.empty: continue
        
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        tau_label = f"${tau:g}$"
        
        # 1. Simulated Data (Scatter points)
        valid_sim = sub_df.dropna(subset=['Rg_Sim'])
        if not valid_sim.empty:
            ax.plot(valid_sim['Act'], valid_sim['Rg_Sim'], marker=m, linestyle='', color=c, 
                    markeredgecolor='k', markersize=7, label=tau_label, zorder=4)
        
        # 2. Manual Sum (Dashed lines)
        valid_var = sub_df.dropna(subset=['Rg_Var'])
        if not valid_var.empty:
            ax.plot(valid_var['Act'], valid_var['Rg_Var'], linestyle='--', linewidth=2.5, color=c, zorder=3, alpha=0.8)
        
        # 3. Fitted Theory (Solid lines)
        valid_fit = sub_df.dropna(subset=['Rg_Fit'])
        if not valid_fit.empty:
            ax.plot(valid_fit['Act'], valid_fit['Rg_Fit'], linestyle='-', linewidth=2.5, color=c, zorder=2)
            
    # Create custom legend entries for the plot styles
    from matplotlib.lines import Line2D
    style_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='Simulated'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2.5, label='Tangent Series'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5, label='Fitted Theory')
    ]
    
    # Place the main legend for generic tau colors inside the plot
    leg1 = ax.legend(loc='upper left', frameon=False, title="Persistence Time $\\tau$", ncol=3, columnspacing=1.0)
    leg1.get_title().set_fontweight('bold')
    ax.add_artist(leg1)
    
    # Add the style legend inside the plot
    leg2 = ax.legend(handles=style_legend_elements, loc='upper right', frameon=True, edgecolor='k', fancybox=False)
    
    ax.set_xlabel('Activity $f_a$')
    ax.set_ylabel('Radius of Gyration $R_g$')
    ax.set_ylim(top=26)
    
    # Clean up top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', top=False, right=False)
    
    plt.tight_layout()
    output_pdf = os.path.join(script_dir, "Rg_Scatter_3Sources_Professional.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"Saved professional plot to {output_pdf}")

if __name__ == "__main__":
    main()
