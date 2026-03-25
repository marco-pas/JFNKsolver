import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set font for publication quality
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12
})

def generate_performance_plots(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return

    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # 1. Create a unified "Method" name (e.g., "AD_f32_GMRES")
    def make_method_name(row):
        lin = "AD" if "Automatic" in str(row.get('Linearization', '')) else "FD"
        prec = "f32" if "32" in str(row.get('Precision', '')) else "f64"
        krylov = str(row.get('Krylov solver', 'UNKNOWN')).upper()
        return f"{lin}_{prec}_{krylov}"

    # 2. Create a unified "Problem" name (e.g., "Burgers_TGV")
    def make_problem_name(row):
        # Extract folder from the File name or assume from Simulation config
        file_str = str(row.get('File', '')).lower()
        sim_type = str(row.get('Simulation', row.get('Simulation IC', row.get('Source Type', '')))).upper()
        
        if 'burgers' in file_str: pde = 'Burgers'
        elif 'raddiff' in file_str: pde = 'RadDiff'
        elif 'reactdiff' in file_str: pde = 'ReactDiff'
        elif 'maxw' in file_str: pde = 'Maxwell'
        else: pde = 'Unknown'
        
        return f"{pde}_{sim_type}"

    df['Method'] = df.apply(make_method_name, axis=1)
    df['Problem'] = df.apply(make_problem_name, axis=1)
    df['Time'] = pd.to_numeric(df['Total Solver Time'], errors='coerce')

    # Drop any runs that critically failed and didn't record a time
    df = df.dropna(subset=['Time'])

    # @@
    # PLOT 1: Global Dolan-Moré Performance Profile
    # @@
    print("Generating Global Performance Profile...")
    
    # Pivot to get a Matrix of [Problems (rows) x Methods (cols)]
    time_matrix = df.pivot(index='Problem', columns='Method', values='Time')
    
    # Fill missing combinations (like CG on Burgers) with Infinity
    time_matrix = time_matrix.fillna(np.inf)
    
    # Find the minimum time for each problem
    min_times = time_matrix.min(axis=1)
    
    # Calculate the normalized slowdown (tau)
    # tau = time / min_time. If time is inf, tau is inf.
    tau_matrix = time_matrix.divide(min_times, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define an array of tau values for the x-axis
    tau_vals = np.logspace(0, 5, 500) # From 1.0x to 100.0x slowdown
    
    methods = tau_matrix.columns
    for method in methods:
        method_taus = tau_matrix[method].values
        
        # Calculate the fraction of problems solved within tau
        fraction_solved = [np.mean(method_taus <= t) for t in tau_vals]
        
        # Make CG visually distinct (dashed line) since it's an exception
        linestyle = '--' if 'CG' in method and 'BICG' not in method else '-'
        linewidth = 2.5 if 'CG' in method else 1.5
        
        ax.plot(tau_vals, fraction_solved, label=method, linestyle=linestyle, linewidth=linewidth)

    ax.set_xscale('log')
    ax.set_xlabel(r'Slowdown Factor $(\tau)$')
    ax.set_ylabel('Fraction of Problems Solved')
    ax.set_title('Global Solver Performance Profile (Dolan-Moré)')
    
    # Shrink current axis by 20% to put legend outside
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('performance_profile_global.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved 'performance_profile_global.png'")

    # @@
    # PLOT 2: Localized SPD Problem (Reaction Diffusion) Bar Chart
    # @@
    print("Generating Localized ReactDiff Comparison...")
    
    # Filter only Reaction Diffusion data
    df_spd = df[df['Problem'].str.contains('ReactDiff', case=False)].copy()
    
    if not df_spd.empty:
        # Sort by Time for a clean waterfall effect
        df_spd = df_spd.sort_values('Time')
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Highlight CG methods in a different color
        colors = ['#ff7f0e' if 'CG' in m and 'BICG' not in m else '#1f77b4' for m in df_spd['Method']]
        
        bars = ax.bar(df_spd['Method'], df_spd['Time'], color=colors, edgecolor='black')
        
        ax.set_ylabel('Total Solver Time (s)')
        ax.set_title('Solver Comparison: Reaction Diffusion (SPD Matrix)')
        plt.xticks(rotation=45, ha='right')
        
        # Add the actual time values on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.02), 
                    f'{yval:.2f}s', ha='center', va='bottom', fontsize=9)
            
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('performance_reactdiff_only.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  -> Saved 'performance_reactdiff_only.png'")
    else:
        print("  -> No Reaction Diffusion data found to plot.")

if __name__ == "__main__":
    # Point this to whatever CSV your main script generates
    target_csv = "benchmark_results_cpu.csv" 
    generate_performance_plots(target_csv)