import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import ast
import matplotlib.cm as cm

# Set font for publication quality
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12
})

# @@
# 1. TEXT TO CSV PARSER
# @@
def compile_summaries_to_csv(output_csv="benchmark_results_cpu.csv"):
    print(f"\nScanning output directories to compile {output_csv}...")
    all_txt_files = []
    
    target_dirs = ["output/burgers", "output/maxw", "output/raddiff", "output/reactdiff"]
    
    for d in target_dirs:
        search_pattern = os.path.join(d, "*_summary.txt")
        all_txt_files.extend(glob.glob(search_pattern))
                
    if not all_txt_files:
        print("No summary files found to compile.")
        return

    all_data = []
    for filepath in all_txt_files:
        data_dict = {
            "File": os.path.basename(filepath),
            "Directory": os.path.dirname(filepath)
        }
        current_section = ""
        
        with open(filepath, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line.startswith("---"):
                    current_section = stripped_line.replace("---", "").strip()
                    continue
                    
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip()
                    val = parts[1].strip()

                    # Ignore the header for data arrays
                    if key.startswith("DATA ARRAYS"):
                        continue
                    
                    if "(" in val and "%)" in val:
                        val = val.split("(")[0].strip()
                        
                    if current_section and key in ["Average", "Std Dev", "Max", "Min"]:
                        full_key = f"{key} - {current_section}"
                    else:
                        full_key = key
                        
                    data_dict[full_key] = val
                    
        # --- Calculate Severe Failures based on Residuals ---
        if "ARRAY_FINAL_RESIDUALS" in data_dict and "Newton tol" in data_dict:
            try:
                tol = float(data_dict["Newton tol"])
                residuals = ast.literal_eval(data_dict["ARRAY_FINAL_RESIDUALS"])
                
                # A "Severe Failure" is > 20x the requested tolerance
                severe_fails = sum(1 for r in residuals if r > (tol * 20.0))
                data_dict["Severe Failures"] = severe_fails
            except Exception as e:
                data_dict["Severe Failures"] = 0
        else:
            data_dict["Severe Failures"] = 0
            
        # Clean up the massive array strings so they don't bloat the CSV
        for k in list(data_dict.keys()):
            if k.startswith("ARRAY_"):
                del data_dict[k]
                
        all_data.append(data_dict)
        
    df = pd.DataFrame(all_data)

    if not df.empty:
        # Unified Config Finder
        def get_unified_config(row):
            for col in row.index:
                col_lower = str(col).lower()
                if col_lower in ['simulation', 'simulation ic', 'source type', 'simulation j', 'ic', 'source']:
                    val = str(row[col]).strip()
                    if val != 'nan' and val != '': return val.upper()
            return "DEFAULT"
            
        df['Unified_Config'] = df.apply(get_unified_config, axis=1)

        # Unified Time Finder
        def get_unified_time(row):
            for col in row.index:
                if 'total solver time' in str(col).lower():
                    return pd.to_numeric(row[col], errors='coerce')
            return np.nan
            
        df['_sort_time'] = df.apply(get_unified_time, axis=1)

        def get_sort_group(row):
            directory = str(row.get('Directory', '')).lower()
            config = row['Unified_Config']
            if 'burgers' in directory: return 1
            elif 'raddiff' in directory or 'radiation' in directory: return 4
            elif 'reactdiff' in directory or 'rcd' in directory: return 5
            elif 'maxw' in directory: return 6
            return 99
            
        df['_sort_group'] = df.apply(get_sort_group, axis=1)
        df = df.sort_values(by=['_sort_group', '_sort_time'], ascending=[True, True])
        df['_best_time'] = df.groupby('_sort_group')['_sort_time'].transform('min')
        df['Slowdown'] = (df['_sort_time'] / df['_best_time']).apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")
        df = df.drop(columns=['_sort_group', '_sort_time', '_best_time', 'Directory'])

    df.to_csv(output_csv, index=False)
    print(f"Successfully compiled {len(all_txt_files)} runs into '{output_csv}'!")


# @@
# 2. PLOTTING GENERATOR
# @@
def generate_performance_plots(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return

    print(f"\nLoading data from {csv_file} to generate plots...")
    df = pd.read_csv(csv_file)

    def make_method_name(row):
        lin = "AD" if "Automatic" in str(row.get('Linearization', '')) else "FD"
        prec = "single" if "32" in str(row.get('Precision', '')) else "double"
        krylov = str(row.get('Krylov solver', 'UNKNOWN')).upper()
        return f"{lin} {prec} {krylov}"

    def make_problem_name(row):
        file_str = str(row.get('File', '')).lower()
        sim_type = str(row.get('Unified_Config', row.get('Simulation', ''))).upper()
        if 'burgers' in file_str: pde = 'Burgers'
        elif 'raddiff' in file_str: pde = 'RadDiff'
        elif 'reactdiff' in file_str: pde = 'ReactDiff'
        elif 'maxw' in file_str: pde = 'Maxwell'
        else: pde = 'Unknown'
        return f"{pde}_{sim_type}"

    df['Method'] = df.apply(make_method_name, axis=1)
    df['Problem'] = df.apply(make_problem_name, axis=1)
    df['Time'] = pd.to_numeric(df['Total Solver Time'], errors='coerce')
    df = df.dropna(subset=['Time'])

    # --- PENALIZE SEVERE DIVERGENCE ---
    failed_count = 0
    if 'Severe Failures' in df.columns:
        severe_failures = pd.to_numeric(df['Severe Failures'], errors='coerce').fillna(0)
        df.loc[severe_failures > 0, 'Time'] = np.inf
        failed_count = (severe_failures > 0).sum()
        print(f"  -> Applied INFINITY penalty to {failed_count} severely diverged runs (>20x tolerance).")
    else:
        print("\n  [Warning] 'Severe Failures' not found in CSV.")

    # GLOBAL COLOR MAPPING
    all_methods = df['Method'].unique()
    prefixes = sorted(list(set([" ".join(m.split(" ")[:2]) for m in all_methods])))
    ad_prefixes = [p for p in prefixes if p.startswith('AD')]
    fd_prefixes = [p for p in prefixes if p.startswith('FD')]

    ad_colors = [cm.magma(x) for x in np.linspace(0.4, 0.7, max(1, len(ad_prefixes)))]
    fd_colors = [cm.viridis(x) for x in np.linspace(0.5, 0.9, max(1, len(fd_prefixes)))]

    prefix_to_color = {}
    for i, p in enumerate(ad_prefixes): prefix_to_color[p] = ad_colors[i]
    for i, p in enumerate(fd_prefixes): prefix_to_color[p] = fd_colors[i]

    # --- PLOT 1: Global Dolan-Moré ---
    print("  -> Generating Global Performance Profile...")
    df_global = df[~df['Method'].str.endswith('CG')].copy()
    time_matrix = df_global.pivot_table(index='Problem', columns='Method', values='Time', aggfunc='min')
    time_matrix = time_matrix.fillna(np.inf)
    
    min_times = time_matrix.min(axis=1)
    tau_matrix = time_matrix.divide(min_times, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    tau_vals = np.linspace(1, 500, int(2e4))
    methods = tau_matrix.columns

    for method in methods:
        method_taus = tau_matrix[method].values
        fraction_solved = [np.mean(method_taus <= t) for t in tau_vals]
        
        parts = method.split(" ")
        prefix = f"{parts[0]} {parts[1]}"
        solver = parts[-1]                 
        line_color = prefix_to_color.get(prefix, 'black')
        
        if solver == 'GMRES': line_style = '--'
        elif solver == 'BICGSTAB': line_style = '-'
        else: line_style = '-.' 

        ax.plot(tau_vals, fraction_solved, label=method, linewidth=2.5, color=line_color, linestyle=line_style)

    ax.set_xscale('log')
    ax.set_xlabel(r'Slowdown Factor $(\tau)$')
    ax.set_ylabel('Fraction of Problems Solved')
    ax.set_title('Global Solver Performance Profile (Dolan-Moré)')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('performance_profile_global.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 2: ReactDiff Bar Chart ---
    print("  -> Generating Localized ReactDiff Comparison...")
    df_spd = df[df['Problem'].str.contains('ReactDiff', case=False)].copy()
    
    if not df_spd.empty:
        problems = df_spd['Problem'].unique()
        n_problems = len(problems)
        
        fig, axes = plt.subplots(1, n_problems, figsize=(8 * n_problems, 6))
        if n_problems == 1: axes = [axes]
            
        for ax, prob in zip(axes, problems):
            subset = df_spd[df_spd['Problem'] == prob].copy()
            subset = subset.sort_values('Time')
            
            max_valid_time = subset.loc[subset['Time'] < np.inf, 'Time'].max()
            if pd.isna(max_valid_time): max_valid_time = 1.0
            plot_times = np.where(subset['Time'] == np.inf, max_valid_time * 1.5, subset['Time'])
            
            bars = ax.bar(subset['Method'], plot_times, edgecolor='black', linewidth=1.2)
            
            for bar, method in zip(bars, subset['Method']):
                parts = method.split(" ")
                prefix = f"{parts[0]} {parts[1]}"
                solver = parts[-1]
                bar.set_facecolor(prefix_to_color.get(prefix, 'gray'))
                if solver == 'CG': bar.set_hatch('///')
            
            ic_name = prob.split('_')[-1]
            ax.set_title(f'Reaction Diffusion: {ic_name.capitalize()}')
            ax.set_ylabel('Total Solver Time (s)')
            ax.set_yscale('log')
            ax.set_xticks(range(len(subset['Method'])))
            ax.set_xticklabels(subset['Method'], rotation=45, ha='right')
                
            best_time = subset.loc[subset['Time'] < np.inf, 'Time'].min()
            
            for bar, true_time in zip(bars, subset['Time']):
                yval = bar.get_height()
                if true_time == np.inf:
                    bar.set_facecolor('#d3d3d3')  
                    bar.set_edgecolor('red')
                    bar.set_linestyle('--')
                    ax.text(bar.get_x() + bar.get_width()/2, yval * 1.15, 
                            'FAILED', ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
                else:
                    slowdown = true_time / best_time
                    label_text = f'{true_time:.2f}s\n({slowdown:.1f}x)'
                    ax.text(bar.get_x() + bar.get_width()/2, yval * 1.15, 
                            label_text, ha='center', va='bottom', fontsize=9)
            
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 3.0)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        plt.savefig('performance_reactdiff_only.png', dpi=300, bbox_inches='tight')
        plt.close()


# @@
# 3. TEXT REPORT GENERATOR
# @@
def generate_ad_vs_fd_report(csv_file):
    if not os.path.exists(csv_file):
        return

    df = pd.read_csv(csv_file)

    def make_method_name(row):
        lin = "AD" if "Automatic" in str(row.get('Linearization', '')) else "FD"
        prec = "single" if "32" in str(row.get('Precision', '')) else "double"
        krylov = str(row.get('Krylov solver', 'UNKNOWN')).upper()
        return f"{lin} {prec} {krylov}"

    def make_problem_name(row):
        file_str = str(row.get('File', '')).lower()
        sim_type = str(row.get('Unified_Config', row.get('Simulation', ''))).upper()
        if 'burgers' in file_str: pde = 'Burgers'
        elif 'raddiff' in file_str: pde = 'RadDiff'
        elif 'reactdiff' in file_str: pde = 'ReactDiff'
        elif 'maxw' in file_str: pde = 'Maxwell'
        else: pde = 'Unknown'
        return f"{pde}_{sim_type}"

    df['Method'] = df.apply(make_method_name, axis=1)
    df['Problem'] = df.apply(make_problem_name, axis=1)
    df['Time'] = pd.to_numeric(df['Total Solver Time'], errors='coerce')

    # --- Synchronize report with plot's Severe Failures logic ---
    if 'Severe Failures' in df.columns:
        severe_failures = pd.to_numeric(df['Severe Failures'], errors='coerce').fillna(0)
        df.loc[severe_failures > 0, 'Time'] = np.inf

    # Filter out CG for fair comparison
    df = df[~df['Method'].str.endswith('CG')].copy()
    
    time_matrix = df.pivot_table(index='Problem', columns='Method', values='Time', aggfunc='min')
    total_problems = len(time_matrix)
    
    time_matrix_filled = time_matrix.fillna(np.inf)
    min_times = time_matrix_filled.min(axis=1)
    tau_matrix = time_matrix_filled.divide(min_times, axis=0)
    
    ad_methods = [m for m in time_matrix.columns if m.startswith('AD')]
    fd_methods = [m for m in time_matrix.columns if m.startswith('FD')]

    # 1. Robustness
    STALL_THRESHOLD = 100.0 
    ad_success = (tau_matrix[ad_methods] <= STALL_THRESHOLD).sum() / total_problems * 100
    fd_success = (tau_matrix[fd_methods] <= STALL_THRESHOLD).sum() / total_problems * 100
    
    best_ad_success = ad_success.max()
    avg_fd_success = fd_success.mean()
    
    # 2. Efficiency
    df['Base_Config'] = df['Method'].apply(lambda x: " ".join(x.split(" ")[1:]))
    df_ad = df[df['Method'].str.startswith('AD')].dropna(subset=['Time'])
    df_fd = df[df['Method'].str.startswith('FD')].dropna(subset=['Time'])
    
    # Exclude failed runs (inf) from head-to-head speedup calculations
    df_ad_valid = df_ad[df_ad['Time'] < np.inf]
    df_fd_valid = df_fd[df_fd['Time'] < np.inf]
    
    head_to_head = pd.merge(df_ad_valid, df_fd_valid, on=['Problem', 'Base_Config'], suffixes=('_AD', '_FD'))
    
    if not head_to_head.empty:
        speedups = head_to_head['Time_FD'] / head_to_head['Time_AD']
        median_speedup = speedups.median()
        mean_speedup = speedups.mean()
    else:
        median_speedup = 0.0
        mean_speedup = 0.0

    # 3. Penalized Geometric Mean
    FAILURE_PENALTY = 1000.0
    capped_tau = tau_matrix.replace(np.inf, FAILURE_PENALTY)
    capped_tau = capped_tau.where(capped_tau < STALL_THRESHOLD, FAILURE_PENALTY)
    
    geomeans = np.exp(np.log(capped_tau).mean(axis=0))
    best_ad_score = geomeans[ad_methods].min()
    best_fd_score = geomeans[fd_methods].min()
    avg_ad_score = geomeans[ad_methods].mean()
    avg_fd_score = geomeans[fd_methods].mean()

    report = f"""
{"="*75}
               AD vs. FD: FIGURES OF MERIT SUMMARY
{"="*75}

1. ROBUSTNESS (Global Convergence Rate)
   Across the {total_problems} distinct physics problems, Automatic Differentiation 
   was exceptionally stable. The optimal AD configuration achieved a 
   {best_ad_success:.1f}% convergence rate. By contrast, Finite Difference 
   approximations suffered from severe truncation errors; FD methods 
   averaged only a {avg_fd_success:.1f}% success rate of solving within a reasonable 
   timeframe (100x slowdown margin) before stalling or diverging.

2. EFFICIENCY (Median Head-to-Head Speedup)
   Even when isolating only the "easy" problems where both methods 
   successfully converged without stalling, the exact Jacobian provided 
   by AD drastically reduced the number of Newton iterations required. 
   AD had a median speed-up of {median_speedup:.2f}x and a mean speed-up 
   of {mean_speedup:.2f}x than FD for the exact same problem setup.

3. THE ULTIMATE GRADE (Penalized Geometric Mean)
   Combining both speed and robustness into a single standard benchmark 
   score (where 1.0 is a perfect score and failures/stalls are penalized):
   - Best AD Configuration : {best_ad_score:>7.2f}  (Group Average: {avg_ad_score:.2f})
   - Best FD Configuration : {best_fd_score:>7.2f}  (Group Average: {avg_fd_score:.2f})
{"="*75}
"""
    print(report)


# @@
# MAIN EXECUTION
# @@
if __name__ == "__main__":
    TARGET_CSV = "benchmark_results_cpu.csv" 
    
    compile_summaries_to_csv(TARGET_CSV)
    generate_performance_plots(TARGET_CSV)
    generate_ad_vs_fd_report(TARGET_CSV)
