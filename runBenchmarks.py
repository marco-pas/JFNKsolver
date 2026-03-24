import os
import glob
import time
import argparse
import itertools
import traceback
import pandas as pd

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
    
    # Explicitly define ONLY the target directories (no subdirectories)
    target_dirs = [
        "output/burgers", 
        "output/maxw", 
        "output/raddiff", 
        "output/reactdiff"
    ]
    
    # Grab only _summary.txt files directly inside these 4 folders
    for d in target_dirs:
        search_pattern = os.path.join(d, "*_summary.txt")
        all_txt_files.extend(glob.glob(search_pattern))
                
    if not all_txt_files:
        print("No summary files found to compile.")
        return

    all_data = []
    for filepath in all_txt_files:
        # We now explicitly track the Directory to guarantee perfect sorting
        data_dict = {
            "File": os.path.basename(filepath),
            "Directory": os.path.dirname(filepath)
        }
        current_section = ""
        
        with open(filepath, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                
                # Track which section of the summary we are in to prevent key collisions
                if stripped_line.startswith("---"):
                    current_section = stripped_line.replace("---", "").strip()
                    continue
                    
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip()
                    val = parts[1].strip()
                    
                    # Remove percentage brackets from standard dev strings if they exist
                    if "(" in val and "%)" in val:
                        val = val.split("(")[0].strip()
                        
                    # Append the section name to the keys for repeated stats (Average, Max, Min)
                    if current_section and key in ["Average", "Std Dev", "Max", "Min"]:
                        full_key = f"{key} - {current_section}"
                    else:
                        full_key = key
                        
                    data_dict[full_key] = val
                    
        all_data.append(data_dict)
        
    df = pd.DataFrame(all_data)

    # --- RANKING AND SORTING LOGIC ---
    if not df.empty:
        def get_sort_group(row):
            # Use the Directory to guarantee we know what PDE this is
            directory = str(row.get('Directory', '')).lower()
            
            # Grab the config parameter (Named differently across solvers)
            config_raw = row.get('Simulation', row.get('Simulation IC', row.get('Source Type', '')))
            config = str(config_raw).strip().upper()
            
            # Rank 1-3: Burgers
            if 'burgers' in directory:
                if 'TGV' in config: return 1
                elif '4VC' in config: return 2
                elif 'DSL' in config: return 3
                else: return 3.5
                
            # Rank 4: Radiative Diffusion
            elif 'raddiff' in directory or 'radiation' in directory:
                return 4
                
            # Rank 5: Reaction Diffusion
            elif 'reactdiff' in directory or 'rcd' in directory:
                return 5
                
            # Rank 6-7: Maxwell
            elif 'maxw' in directory:
                if 'GAUSSIAN' in config: return 6
                elif 'DIPOLE' in config: return 7
                else: return 7.5
                
            return 99
            
        # Create temporary sorting columns
        df['_sort_group'] = df.apply(get_sort_group, axis=1)
        # Convert "Total Solver Time" to float so pandas sorts it mathematically, not alphabetically
        df['_sort_time'] = pd.to_numeric(df['Total Solver Time'], errors='coerce')
        
        # Sort by simulation group, then by solver time (fastest to slowest)
        df = df.sort_values(by=['_sort_group', '_sort_time'], ascending=[True, True])
        
        # --- CALCULATE SLOWDOWN ---
        # Group by the simulation type and find the absolute fastest time for that specific group
        df['_best_time'] = df.groupby('_sort_group')['_sort_time'].transform('min')
        # Divide the current row's time by the group's best time and format as "1.00x", "2.54x", etc.
        df['Slowdown'] = (df['_sort_time'] / df['_best_time']).apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")

        # Drop the temporary helper columns before saving
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
    KRYLOV_TOL  = {'float32': 1e-4, 'float64': 1e-8}
    NEWTON_TOL  = {'float32': 1e-3, 'float64': 1e-6}
    KRYLOV_ITER = 100
    NEWTON_ITER = 15
    MAX_BT_ITER = 15
    PHYS_PARAM  = 1234.0   # Placeholder physics parameter requested

    # Physical Params
    NU      = 0.05      # Burgers' eq viscosity
    EPSILON = 0.1       # Rad diff
    DIFF    = 0.01      # Diff coeff react diff
    MU0     = 1.0       # Maxw
    EPS0    = 1.0       # Maxw
    COURANT = 1         # solver is implcit
    OMEGA_STEPS = 3     # Maxwell steps

    # ---- Hardware Specific Overrides ----
    if args.device == 'cpu':
        TM_STEPS = 3
        TM_N     = 256
        MAXW_N   = 128
    else:
        TM_STEPS = 12
        TM_N     = 512
        MAXW_N   = 256

    print(f"\n{'='*50}\nSTARTING BENCHMARK SUITE ON {args.device.upper()}\n{'='*50}\n")
    
    # ------------ SUITE 1: BURGERS EQUATION ------------ 
    
    print("\n--- Queuing Burgers' Equation ---")
    b_precisions = ['float32', 'float64']
    b_ics        = ['TGV', 'DSL', '4VC']
    b_ads        = [True, False]
    b_solvers    = ['gmres', 'bicgstab']
    
    b_combos = list(itertools.product(b_precisions, b_ics, b_ads, b_solvers))
    
    for i, (prec, ic, ad, solver) in enumerate(b_combos):
        print(f"  [Burgers {i+1}/{len(b_combos)}] {prec} | {ic} | AD:{ad} | {solver.upper()}")
        try:
            burgersSolver.runSimulation(
                device=args.device, PRECISION=prec, BC_X='periodic', BC_Y='periodic', SIMULATION_IC=ic,
                verbose=False, useAD=ad, maxBackTrackingIter=MAX_BT_ITER, nu=PHYS_PARAM,
                steps=TM_STEPS, Nx=TM_N, Ny=TM_N, Courant=COURANT, KrylovSolver=solver,
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
    rd_ads        = [True, False]
    rd_solvers    = ['gmres', 'bicgstab']
    
    rd_combos = list(itertools.product(rd_precisions, rd_ads, rd_solvers))
    
    for i, (prec, ad, solver) in enumerate(rd_combos):
        print(f"  [RadDiff {i+1}/{len(rd_combos)}] {prec} | CLASSIC_SU_OLSON | AD:{ad} | {solver.upper()}")
        try:
            raddiffSolver.runSimulation(
                device=args.device, PRECISION=prec, BC_X='dirichlet', BC_Y='periodic', 
                SIMULATION_IC='SO', SOURCE_TYPE='central', verbose=False, useAD=ad, 
                maxBackTrackingIter=MAX_BT_ITER, epsilon=EPSILON, Q0=1.0, x_src=0.5, tau_src=float('inf'), 
                Courant=COURANT, steps=TM_STEPS, Nx=TM_N, Ny=TM_N, KrylovSolver=solver,
                KrylovTol=KRYLOV_TOL[prec], KrylovIter=KRYLOV_ITER, 
                NewtonNonlinTol=NEWTON_TOL[prec], NewtonIter=NEWTON_ITER,
                plot_steps=TM_STEPS, gif_fps=10, displayPlot=False, figFolder="output/raddiff",
                save_steps=-1, dataFolder=""
            )
        except Exception as e:
            print(f"    -> FAILED: {e}")

    
    # ------------ SUITE 3: REACTION DIFFUSION ------------ 
    
    print("\n--- Queuing Reaction Diffusion ---")
    rcd_precisions = ['float32', 'float64']
    rcd_ads        = [True, False]
    rcd_solvers    = ['gmres', 'bicgstab', 'cg'] # <--- includes CG!
    
    rcd_combos = list(itertools.product(rcd_precisions, rcd_ads, rcd_solvers))
    
    for i, (prec, ad, solver) in enumerate(rcd_combos):
        print(f"  [ReactDiff {i+1}/{len(rcd_combos)}] {prec} | gaussian | AD:{ad} | {solver.upper()}")
        try:
            reactdiffSolver.runSimulation(
                device=args.device, PRECISION=prec, BC_X='dirichlet', BC_Y='dirichlet', 
                SIMULATION_IC='gaussian', verbose=False, useAD=ad, maxBackTrackingIter=MAX_BT_ITER,
                D=DIFF, steps=TM_STEPS, Nx=TM_N, Ny=TM_N, Courant=COURANT, KrylovSolver=solver,
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
    mx_sources    = ['dipole', 'gaussian_center']  # <--- Added gaussian_center
    mx_ads        = [True, False]
    mx_solvers    = ['gmres', 'bicgstab']
    
    # Add mx_sources to the combinatorial sweep
    mx_combos = list(itertools.product(mx_precisions, mx_sources, mx_ads, mx_solvers))
    
    for i, (prec, source, ad, solver) in enumerate(mx_combos):
        print(f"  [Maxwell {i+1}/{len(mx_combos)}] {prec} | {source} | AD:{ad} | {solver.upper()}")
        try:
            maxwSolver.runSimulation(
                device=args.device, 
                PRECISION=prec, 
                SIMULATION_J=source,     # <--- Dynamically sets 'dipole' or 'gaussian_center'
                useAD=ad, 
                verbose=False,
                mu0=PHYS_PARAM, eps0=PHYS_PARAM, omega_start=5.0, omega_stop=200.0, omega_steps=OMEGA_STEPS,
                Nx=MAXW_N, Ny=MAXW_N, KrylovSolver=solver, 
                KrylovTol=KRYLOV_TOL[prec], KrylovIter=KRYLOV_ITER,
                NewtonTol=NEWTON_TOL[prec], NewtonIter=NEWTON_ITER, maxBackTrackingIter=MAX_BT_ITER,
                figFolder="output/maxw", save_field_pic=5  # Saves image every 5th frequency
            )
        except Exception as e:
            print(f"    -> FAILED: {e}")

    scan_time_end = time.perf_counter()

    print(f"\n    ---> Final time for Benchmark is: {scan_time_end-scan_time_start:.2f} s")

    # Compile the final data
    csv_name = f"benchmark_results_{args.device}.csv"
    compile_summaries_to_csv(csv_name)
    
    print(f"\n{'='*50}\nALL BENCHMARKS FINISHED ON {args.device.upper()}\n{'='*50}")