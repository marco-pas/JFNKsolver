import os
import glob
import time
import argparse
import itertools
import traceback
import pandas as pd
import numpy as np

# Import the 4 unified JFNK simulation scripts
import burgersSolver
import raddiffSolver
import reactdiffSolver
import maxwSolver

# ------------ 1. DIRECTORY CLEANUP ------------ 

def cleanup_old_files():
    print("Cleaning up old outputs (*.gif, *.png, *.txt)...")
    folders_to_clean = [
        "output/burgers", 
        "output/maxw", 
        "output/raddiff", 
        "output/reactdiff"
    ]
    
    deleted_count = 0
    for folder in folders_to_clean:
        if os.path.exists(folder):
            for ext in ['*.gif', '*.png', '*.txt']:
                for filepath in glob.glob(os.path.join(folder, ext)):
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                    except Exception as e:
                        print(f"  Failed to delete {filepath}: {e}")
                        
    print(f"Cleaned {deleted_count} files.\n")


# ------------ 2. TEXT TO CSV PARSER & RANKER ------------  

def compile_summaries_to_csv(output_csv="benchmark_results.csv"):
    print("\nCompiling and ranking all summary TXT files into CSV...")
    all_txt_files = []
    
    target_dirs = [
        "output/burgers", 
        "output/maxw", 
        "output/raddiff", 
        "output/reactdiff"
    ]
    
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

                    # Ignore massive array data
                    if key.startswith("ARRAY_") or key.startswith("DATA ARRAYS"):
                        continue
                    
                    if "(" in val and "%)" in val:
                        val = val.split("(")[0].strip()
                        
                    if current_section and key in ["Average", "Std Dev", "Max", "Min"]:
                        full_key = f"{key} - {current_section}"
                    else:
                        full_key = key
                        
                    data_dict[full_key] = val
                    
        all_data.append(data_dict)
        
    df = pd.DataFrame(all_data)

    if not df.empty:
        # --- RANKING AND SORTING LOGIC ---
        # --- ROBUST CONFIGURATION FINDER ---
        def get_unified_config(row):
            # Hunts for any column name that suggests it's the simulation type
            for col in row.index:
                col_lower = str(col).lower()
                if col_lower in ['simulation', 'simulation ic', 'source type', 'simulation j', 'ic', 'source']:
                    val = str(row[col]).strip()
                    if val != 'nan' and val != '':
                        return val.upper()
            return "DEFAULT"
            
        # Create a unified column so the plotting script never has to guess
        df['Unified_Config'] = df.apply(get_unified_config, axis=1)

        # --- ROBUST TIME FINDER ---
        def get_unified_time(row):
            for col in row.index:
                if 'total solver time' in str(col).lower():
                    return pd.to_numeric(row[col], errors='coerce')
            return np.nan
            
        df['_sort_time'] = df.apply(get_unified_time, axis=1)

        def get_sort_group(row):
            directory = str(row.get('Directory', '')).lower()
            config = str(row['Unified_Config']).upper()
            
            # 1. Hunt for the specific physical parameter and grid size in the parsed TXT columns
            param_suffix = "_default"
            nx_suffix = ""
            for col in row.index:
                col_lower = str(col).lower()
                
                # Check for grid size (adjust these strings based on what your solvers actually print)
                if col_lower in ['nx', 'grid x', 'grid_x', 'n', 'resolution']:
                    nx_suffix = f"_Nx:{str(row[col]).strip()}"

                # Check for physical parameters
                elif 'burgers' in directory and col_lower in ['nu', 'viscosity', 'kinematic viscosity']:
                    param_suffix = f"_nu:{str(row[col]).strip()}"
                elif 'raddiff' in directory and col_lower in ['eps', 'epsilon', 'opacity']:
                    param_suffix = f"_eps:{str(row[col]).strip()}"
                elif 'reactdiff' in directory and col_lower in ['d', 'diff', 'diffusion']:
                    param_suffix = f"_diff:{str(row[col]).strip()}"
                elif 'maxw' in directory and col_lower in ['chi', 'kerr']:
                    param_suffix = f"_chi:{str(row[col]).strip()}"

            full_suffix = f"{param_suffix}{nx_suffix}"

            # 2. Assign the base group + the dynamically found parameters
            if 'burgers' in directory:
                if 'TGV' in config: return f"1.1{full_suffix}"
                elif '4VC' in config: return f"1.2{full_suffix}"
                elif 'DSL' in config: return f"1.3{full_suffix}"
                else: return f"1.9{full_suffix}"
                
            elif 'raddiff' in directory:
                if 'SU_OLSON' in config: return f"2.1{full_suffix}"
                elif 'DYNAMIC' in config: return f"2.2{full_suffix}"
                else: return f"2.9{full_suffix}"
                
            elif 'reactdiff' in directory:
                if 'GAUSSIAN' in config: return f"3.1{full_suffix}"
                elif 'SINUSOIDAL' in config: return f"3.2{full_suffix}"
                else: return f"3.9{full_suffix}"
                
            elif 'maxw' in directory:
                if 'GAUSSIAN' in config: return f"4.1{full_suffix}"
                elif 'DIPOLE' in config: return f"4.2{full_suffix}"
                else: return f"4.9{full_suffix}"
                
            return f"9.9{full_suffix}"
            
        df['_sort_group'] = df.apply(get_sort_group, axis=1)
        df = df.sort_values(by=['_sort_group', '_sort_time'], ascending=[True, True])
        
        df['_best_time'] = df.groupby('_sort_group')['_sort_time'].transform('min')
        df['Slowdown'] = (df['_sort_time'] / df['_best_time']).apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")

        df = df.drop(columns=['_sort_group', '_sort_time', '_best_time', 'Directory'])

    df.to_csv(output_csv, index=False)
    print(f"Successfully compiled and ranked {len(all_txt_files)} runs into '{output_csv}'!")


# ------------ 3. MAIN BENCHMARK EXECUTION ------------ 

if __name__ == "__main__":

    scan_time_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Master PDE Benchmark Suite")
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], required=True, 
                        help="Target hardware: 'cpu' or 'gpu'")
    args = parser.parse_args()

    cleanup_old_files()

    # ---- Global Solver Tolerances (Precision Dependent) ----
    KRYLOV_TOL  = {'float32': 1e-5, 'float64': 1e-7} # 1e-6 1e-8
    NEWTON_TOL  = {'float32': 1e-3, 'float64': 1e-5} # 1e-4 1e-6
    KRYLOV_ITER = 100
    NEWTON_ITER = 15
    NEWTON_ITER_Maxw = 30
    MAX_BT_ITER = 15

    # Physical Params (Constants that are not being scanned)
    MU0     = 1.0       # Maxw
    EPS0    = 1.0       # Maxw
    COURANT = 2       # solver is implicit
    OMEGA_STEPS = 20     # Maxwell steps

    # ---- Hardware Specific Overrides & Grid Scans ----
    if args.device == 'cpu':
        TM_STEPS   = 50
        GRID_SIZES = [64, 128] # Two distinct grid sizes for CPU
    else:
        TM_STEPS   = 50
        GRID_SIZES = [256, 512] # Two distinct grid sizes for GPU

    print(f"\n{'='*50}\nSTARTING BENCHMARK SUITE ON {args.device.upper()}\n{'='*50}\n")
    
    # ------------ SUITE 1: BURGERS EQUATION ------------ 
    
    print("\n--- Queuing Burgers' Equation ---")
    b_precisions = ['float32', 'float64']
    b_ics        = ['TGV', 'DSL', '4VC']
    b_nus        = [0.1, 0.05, 0.01]
    b_ads        = [True, False]
    b_solvers    = ['gmres', 'bicgstab']
    
    # Added GRID_SIZES to the cartesian product
    b_combos = list(itertools.product(b_precisions, b_ics, b_nus, b_ads, GRID_SIZES, b_solvers))
    
    for i, (prec, ic, nu_val, ad, n_val, solver) in enumerate(b_combos):
        print(f"  [Burgers {i+1}/{len(b_combos)}] {prec} | {ic} | NU:{nu_val} | Nx:{n_val} | AD:{ad} | {solver.upper()}")
        try:
            burgersSolver.runSimulation(
                device=args.device, PRECISION=prec, BC_X='periodic', BC_Y='periodic', SIMULATION_IC=ic,
                verbose=False, useAD=ad, maxBackTrackingIter=MAX_BT_ITER, nu=nu_val,
                steps=TM_STEPS, Nx=n_val, Ny=n_val, Courant=COURANT, KrylovSolver=solver, # Replaced TM_N with n_val
                KrylovTol=KRYLOV_TOL[prec], KrylovIter=KRYLOV_ITER, 
                NewtonNonlinTol=NEWTON_TOL[prec], NewtonIter=NEWTON_ITER,
                plot_steps=TM_STEPS, gif_fps=10, displayPlot=False, figFolder="output/burgers",
                save_steps=-1, dataFolder=""
            )
        except Exception as e:
            print(f"    -> FAILED: {e}")

    
    # ------------ SUITE 2: RADIATIVE DIFFUSION (SU-OLSON) ------------ 
    
    print("\n--- Queuing Radiative Diffusion ---")
    rd_precisions = ['float32', 'float64']
    rd_profiles   = ['CLASSIC_SU_OLSON', 'DYNAMIC']
    rd_eps        = [1.0, 0.1, 0.01]
    rd_ads        = [True, False]
    rd_solvers    = ['gmres', 'bicgstab']
    
    # Added GRID_SIZES to the cartesian product
    rd_combos = list(itertools.product(rd_precisions, rd_profiles, rd_eps, rd_ads, GRID_SIZES, rd_solvers))
    
    for i, (prec, profile, eps_val, ad, n_val, solver) in enumerate(rd_combos):
        print(f"  [RadDiff {i+1}/{len(rd_combos)}] {prec} | {profile} | EPS:{eps_val} | Nx:{n_val} | AD:{ad} | {solver.upper()}")
        
        # Map the profile to the correct internal arguments!
        if profile == 'CLASSIC_SU_OLSON':
            sim_ic      = 'SO'
            src_type    = 'central'
            bc_x        = 'dirichlet'
            bc_y        = 'periodic'
        else: # DYNAMIC
            sim_ic      = 'SO'
            src_type    = 'pulsar'
            bc_x        = 'dirichlet'
            bc_y        = 'dirichlet'

        try:
            raddiffSolver.runSimulation(
                device=args.device, PRECISION=prec, BC_X=bc_x, BC_Y=bc_y, SIMULATION_TYPE=profile,
                SIMULATION_IC=sim_ic, SOURCE_TYPE=src_type, verbose=False, useAD=ad, 
                maxBackTrackingIter=MAX_BT_ITER, epsilon=eps_val, Q0=1.0, x_src=0.5, tau_src=float('inf'), 
                Courant=COURANT, steps=TM_STEPS, Nx=n_val, Ny=n_val, KrylovSolver=solver, # Replaced TM_N with n_val
                KrylovTol=KRYLOV_TOL[prec], KrylovIter=KRYLOV_ITER, 
                NewtonNonlinTol=NEWTON_TOL[prec], NewtonIter=NEWTON_ITER,
                plot_steps=TM_STEPS, gif_fps=10, displayPlot=False, figFolder="output/raddiff",
                save_steps=-1, dataFolder=""
            )
        except Exception as e:
            print(f"    -> FAILED: {e}")
            traceback.print_exc()  

    
    # ------------ SUITE 3: REACTION DIFFUSION ------------ 
    
    print("\n--- Queuing Reaction Diffusion ---")
    rcd_precisions = ['float32', 'float64']
    rcd_ics        = ['gaussian', 'sinusoidal']
    rcd_diffs      = [0.1, 0.01, 0.001]
    rcd_ads        = [True, False]
    rcd_solvers    = ['gmres', 'bicgstab', 'cg'] 
    
    # Added GRID_SIZES to the cartesian product
    rcd_combos = list(itertools.product(rcd_precisions, rcd_ics, rcd_diffs, rcd_ads, GRID_SIZES, rcd_solvers))
    
    for i, (prec, ic, diff_val, ad, n_val, solver) in enumerate(rcd_combos):
        print(f"  [ReactDiff {i+1}/{len(rcd_combos)}] {prec} | {ic} | DIFF:{diff_val} | Nx:{n_val} | AD:{ad} | {solver.upper()}")
        try:
            reactdiffSolver.runSimulation(
                device=args.device, PRECISION=prec, BC_X='dirichlet', BC_Y='dirichlet', 
                SIMULATION_IC=ic, verbose=False, useAD=ad, maxBackTrackingIter=MAX_BT_ITER,
                D=diff_val, steps=TM_STEPS, Nx=n_val, Ny=n_val, Courant=COURANT, KrylovSolver=solver, # Replaced TM_N with n_val
                KrylovTol=KRYLOV_TOL[prec], KrylovIter=KRYLOV_ITER, 
                NewtonNonlinTol=NEWTON_TOL[prec], NewtonIter=NEWTON_ITER,
                plot_steps=TM_STEPS, gif_fps=10, displayPlot=False, figFolder="output/reactdiff",
                save_steps=-1, dataFolder=""
            )
        except Exception as e:
            print(f"    -> FAILED: {e}")

    
    # ------------ SUITE 4: MAXWELL (EIGENVALUE/SWEEP) ------------ 
    
    print("\n--- Queuing Maxwell Equation ---")
    mx_precisions = ['float32', 'float64']
    mx_sources    = ['dipole', 'gaussian_center']
    mx_chis       = [0.5, 0.1, 0.05]
    mx_ads        = [True, False]
    mx_solvers    = ['gmres', 'bicgstab']
    
    # Added GRID_SIZES to the cartesian product
    mx_combos = list(itertools.product(mx_precisions, mx_sources, mx_chis, mx_ads, GRID_SIZES, mx_solvers))
    
    for i, (prec, source, chi_val, ad, n_val, solver) in enumerate(mx_combos):
        print(f"  [Maxwell {i+1}/{len(mx_combos)}] {prec} | {source} | CHI:{chi_val} | Nx:{n_val} | AD:{ad} | {solver.upper()}")
        try:
            maxwSolver.runSimulation(
                device=args.device, 
                PRECISION=prec, 
                SIMULATION_J=source,     
                useAD=ad, 
                verbose=False,
                mu0=MU0, eps0=EPS0, chi=chi_val,
                omega_start=5.0, omega_stop=200.0, omega_steps=OMEGA_STEPS,
                Nx=n_val, Ny=n_val, KrylovSolver=solver, # Replaced MAXW_N with n_val
                KrylovTol=KRYLOV_TOL[prec], KrylovIter=KRYLOV_ITER,
                NewtonTol=NEWTON_TOL[prec], NewtonIter=NEWTON_ITER_Maxw, maxBackTrackingIter=MAX_BT_ITER,
                figFolder="output/maxw", save_field_pic=5 
            )
        except Exception as e:
            print(f"    -> FAILED: {e}")
            traceback.print_exc()  

    scan_time_end = time.perf_counter()

    print(f"\n    ---> Final time for Benchmark is: {scan_time_end-scan_time_start:.2f} s")

    # Compile the final data
    csv_name = f"benchmark_results_{args.device}.csv"
    compile_summaries_to_csv(csv_name)
    
    print(f"\n{'='*50}\nALL BENCHMARKS FINISHED ON {args.device.upper()}\n{'='*50}")