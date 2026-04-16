import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Global LaTeX & Publication Aesthetics ---
# Match exactly the style of plot_and_fit_modes.py
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
                        'Mode': int(parts[2]),
                        'Variance': float(parts[3])
                    })
                except ValueError: pass
    return pd.DataFrame(data)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "bond_fluctuation_variances.txt")
    
    if not os.path.exists(data_file):
        print(f"Waiting for {data_file} to be generated...")
        return
        
    df = load_data(data_file)
    if df.empty:
        print(f"No data found in {data_file}.")
        return

    # Store original df to access mode 0 later
    df_original = df.copy()

    # User requested to plot up to 32 modes for the spectrum
    max_modes = 32
    df = df[(df['Mode'] >= 1) & (df['Mode'] <= max_modes)]
    
    unique_taus = sorted(df['Tau'].unique())
    unique_acts = sorted(df['Activity'].unique())
    cmap = plt.get_cmap('jet') # Match the colormap used in the reference script
    
    # --- 1. Fixed Tau, Varying Activity ---
    # Will generate Plot_BondFluc_FixedTau_{tau}.pdf
    grouped = df.groupby(['Tau', 'Activity'])
    
    for tau in unique_taus:
        tau_acts = [act for act in unique_acts if (tau, act) in grouped.groups]
        if not tau_acts: continue
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Plot Activity = 0 as black squares reference
        if (tau, 0) in grouped.groups:
            d = grouped.get_group((tau, 0)).sort_values('Mode')
            plt.plot(d['Mode'], d['Variance'], 'sk-', markerfacecolor='k', markersize=4, label='0')
            
        # Select active activities to plot to match reference structure
        allowed_acts = [8, 16, 32, 64, 96]
        active_acts = [a for a in tau_acts if a > 0 and a in allowed_acts]
        
        for i, act in enumerate(active_acts):
            color = cmap(i / max(1, len(active_acts) - 1))
            d = grouped.get_group((tau, act)).sort_values('Mode')
            label = f'${act:g}$'
            plt.plot(d['Mode'], d['Variance'], 'o-', color=color, markerfacecolor=color, markersize=4, label=label)
            
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'Mode number ($p$)')
        plt.ylabel(r'$\displaystyle \langle \delta B_p^2 \rangle$')
        
        # Adjust limits with 10% log-space padding
        x_min, x_max = 1, 32
        log_x_min = np.log10(x_min)
        log_x_max = np.log10(x_max)
        x_padding = (log_x_max - log_x_min) * 0.05
        plt.xlim(10**(log_x_min - x_padding), 10**(log_x_max + x_padding))
        
        # Get automatic y-limits and expand them
        ax_curr = plt.gca()
        y_min, y_max = ax_curr.get_ylim()
        log_y_min = np.log10(y_min)
        log_y_max = np.log10(y_max)
        y_padding = (log_y_max - log_y_min) * 0.05
        plt.ylim(10**(log_y_min - y_padding), 10**(log_y_max + y_padding))

        # Parameter Info: Top right
        plt.text(0.95, 0.95, rf'$\tau = {tau:g}$', transform=plt.gca().transAxes,
                 ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
                 
        # Legend 1: Colors (Activities) placed just below the text
        leg1 = plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.88), ncol=2, frameon=False, title=r"Activity $f_a$", fontsize=8, columnspacing=1.0)
        leg1.get_title().set_fontweight('bold')
        plt.gca().add_artist(leg1)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', which='both')
        
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, f'Plot_BondFluc_FixedTau_{tau:g}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

    # --- 2. Fixed Activity, Varying Tau ---
    # Will generate Plot_BondFluc_FixedAct_{act}.pdf
    for act in unique_acts:
        if act == 0: continue
        act_taus = [tau for tau in unique_taus if (tau, act) in grouped.groups and tau != 0.1]
        if not act_taus: continue
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Plot arbitrary passive reference
        tau_ref = act_taus[0] if act_taus else (0.1 if 0.1 in unique_taus else None)
        if tau_ref is not None and (tau_ref, 0) in grouped.groups:
             d = grouped.get_group((tau_ref, 0)).sort_values('Mode')
             plt.plot(d['Mode'], d['Variance'], 'sk--', markerfacecolor='0.7', markeredgecolor='none', markersize=4, label='0')
             
        for i, tau in enumerate(act_taus):
            color = cmap(i / max(1, len(act_taus) - 1))
            d = grouped.get_group((tau, act)).sort_values('Mode')
            label = rf'${tau:g}$'
            plt.plot(d['Mode'], d['Variance'], 'o-', color=color, markerfacecolor=color, markersize=4, label=label)
            
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'Mode number ($p$)')
        plt.ylabel(r'$\displaystyle \langle \delta B_p^2 \rangle$')
        
        # Adjust limits with 10% log-space padding
        x_min, x_max = 1, 32
        log_x_min = np.log10(x_min)
        log_x_max = np.log10(x_max)
        x_padding = (log_x_max - log_x_min) * 0.05
        plt.xlim(10**(log_x_min - x_padding), 10**(log_x_max + x_padding))
        
        # Get automatic y-limits and expand them
        ax_curr = plt.gca()
        y_min, y_max = ax_curr.get_ylim()
        log_y_min = np.log10(y_min)
        log_y_max = np.log10(y_max)
        y_padding = (log_y_max - log_y_min) * 0.05
        plt.ylim(10**(log_y_min - y_padding), 10**(log_y_max + y_padding))

        # Parameter Info: Top right
        plt.text(0.95, 0.95, rf'$f_a = {act:g}$', transform=plt.gca().transAxes,
                 ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
                 
        # Legend 1: Colors (Taus) placed just below the text
        leg1 = plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.88), ncol=2, frameon=False, title=r"Persistence Time $\tau$", fontsize=8, columnspacing=1.0)
        leg1.get_title().set_fontweight('bold')
        plt.gca().add_artist(leg1)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', which='both')
        
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, f'Plot_BondFluc_FixedAct_{act:g}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()
        
    # --- 3. Total Fluctuation Variance vs Activity ---
    plt.rcParams.update({
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5
    })
    
    df_total = df.groupby(['Tau', 'Activity'])['Variance'].sum().reset_index()
    if not df_total.empty:
        colors = ["#000000", "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        
        plt.figure(figsize=(fig_width, fig_height))
        for i, tau in enumerate(sorted(df_total['Tau'].unique())):
            sub_df = df_total[df_total['Tau'] == tau].sort_values(by='Activity')
            if sub_df.empty: continue
            
            c = colors[i % len(colors)]
            m = markers[i % len(markers)]
            tau_label = f"${tau:g}$"
            
            plt.plot(sub_df['Activity'], sub_df['Variance'], marker=m, linestyle='-', color=c, 
                    markeredgecolor='k', markersize=4, label=tau_label)

        leg = plt.legend(loc='upper left', frameon=False, title=r"Persistence Time $\tau$", ncol=2, columnspacing=1.0)
        leg.get_title().set_fontweight('bold')
        
        plt.xlabel(r'Activity $f_a$')
        plt.ylabel(r'Total Variance $\sum \langle \delta B_p^2 \rangle$')
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', top=False, right=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, "Plot_BondFluc_TotalVar_vs_Act.pdf"), format='pdf', bbox_inches='tight')
        plt.close()

    # --- 4. 0th Mode Variance vs Activity ---
    df_0th = df_original[df_original['Mode'] == 0]
    if not df_0th.empty:
        plt.figure(figsize=(fig_width, fig_height))
        for i, tau in enumerate(sorted(df_0th['Tau'].unique())):
            sub_df = df_0th[df_0th['Tau'] == tau].sort_values(by='Activity')
            if sub_df.empty: continue
            
            c = colors[i % len(colors)]
            m = markers[i % len(markers)]
            tau_label = f"${tau:g}$"
            
            plt.plot(sub_df['Activity'], sub_df['Variance'], marker=m, linestyle='-', color=c, 
                    markeredgecolor='k', markersize=4, label=tau_label)

        leg = plt.legend(loc='upper left', frameon=False, title=r"Persistence Time $\tau$", ncol=2, columnspacing=1.0)
        leg.get_title().set_fontweight('bold')
        
        plt.xlabel(r'Activity $f_a$')
        plt.ylabel(r'Length Fluctuation Variance $\langle \delta B_0^2 \rangle$')
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', top=False, right=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, "Plot_BondFluc_TotalLengthVar_vs_Act.pdf"), format='pdf', bbox_inches='tight')
        plt.close()

    # --- 5. Average Bond Length vs Activity with Variance Errorbars ---
    blen_file = os.path.join(script_dir, "bondlength_averaged.txt")
    if os.path.exists(blen_file) and not df_0th.empty:
        # Load average bond length data
        blen_data = []
        with open(blen_file, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        blen_data.append({'Tau': float(parts[0]), 'Activity': float(parts[1]), 'b_len': float(parts[2])})
                    except ValueError: pass
        df_blen = pd.DataFrame(blen_data)
        
        if not df_blen.empty:
            plt.figure(figsize=(fig_width, fig_height))
            colors = ["#000000", "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
            markers = ['o', 's', '^', 'D', 'v', '<', '>']
            
            # Merge with 0th mode variance to get error bars
            # std_b_n(t) ~ sqrt(<B_0^2>) / 2 (since B_0 is 2*<b_n(t)>)
            df_err = pd.merge(df_blen, df_0th, on=['Tau', 'Activity'])
            df_err['err'] = 0.5 * np.sqrt(df_err['Variance'])
            
            for i, tau in enumerate(sorted(df_err['Tau'].unique())):
                if tau == 0.1: continue
                sub_df = df_err[df_err['Tau'] == tau].sort_values(by='Activity')
                if sub_df.empty: continue
                
                c = colors[i % len(colors)]
                m = markers[i % len(markers)]
                tau_label = f"${tau:g}$"
                
                # Plot the solid line and markers
                plt.plot(sub_df['Activity'], sub_df['b_len'], 
                         marker=m, linestyle='-', color=c, 
                         markeredgecolor='k', markersize=6, linewidth=1.5, label=tau_label)
                
                # Use a translucent shaded band for the error instead of caps for a cleaner aesthetic
                plt.fill_between(sub_df['Activity'], 
                                 sub_df['b_len'] - sub_df['err'], 
                                 sub_df['b_len'] + sub_df['err'], 
                                 color=c, alpha=0.2, edgecolor='none')
            leg = plt.legend(loc='upper left', frameon=False, title=r"Persistence Time $\tau$", ncol=2, columnspacing=1.0)
            leg.get_title().set_fontweight('bold')
            
            plt.xlabel(r'Activity $f_a$')
            plt.ylabel(r'Average Bond Length $\langle b \rangle$')
            
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(direction='in', top=False, right=False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(script_dir, "Plot_AvgBondLength_with_FluctuationErr_vs_Act.pdf"), format='pdf', bbox_inches='tight')
            plt.close()

    print("Successfully generated all plots based on plot_and_fit_modes.py styling.")

if __name__ == "__main__":
    main()
