import time
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    import cupyx.scipy.sparse.linalg as cp_spla
    from bicgstabCuPy import bicgstab as bicgstab_cupy
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Warning: CuPy not found. GPU benchmark will be skipped.")

# Suppress NumPy/SciPy RuntimeWarnings (like overflow in dot product)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def build_poisson_2d(N, dtype=np.float64):
    """
    Builds a 2D Poisson sparse matrix (5-point stencil) of size (N^2, N^2).
    """
    main_diag = 2.0 * np.ones(N, dtype=dtype)
    off_diag = -1.0 * np.ones(N - 1, dtype=dtype)
    D1 = sp.diags([main_diag, off_diag, off_diag], [0, -1, 1], format='csr', dtype=dtype)
    A = sp.kronsum(D1, D1, format='csr')
    return A.astype(dtype)

def run_benchmark(grids = [64, 128, 256, 512], num_runs=10):
    precisions = [
        ('float64', 'Double Precision (float64)', np.float64),
        ('float32', 'Single Precision (float32)', np.float32)
    ]

    # Dictionary to store stats for plotting and final table
    plot_data = {
        'float32': {'N': [], 'avg': [], 'std': [], 'min': [], 'max': []},
        'float64': {'N': [], 'avg': [], 'std': [], 'min': [], 'max': []}
    }

    for prec_key, prec_name, dtype in precisions:
        warmed_up = False

        for N in grids:
            unknowns = N**2
            tol = 1e-3 if dtype == np.float32 else 1e-6

            A_cpu = build_poisson_2d(N, dtype=dtype)
            b_cpu = np.ones(unknowns, dtype=dtype)

            if not HAS_GPU:
                print(f"{N}x{N:<7} | {unknowns:<8} | N/A GPU not found")
                continue

            A_gpu = cp_sp.csr_matrix(A_cpu)
            b_gpu = cp.array(b_cpu)

            # Warmup GPU
            if not warmed_up:
                bicgstab_cupy(A_gpu, b_gpu, rtol=tol)
                cp.cuda.Stream.null.synchronize()
                warmed_up = True

            cpu_times = []
            gpu_times = []

            # Run benchmark loop
            for _ in range(num_runs):
                # CPU Timing
                t0_cpu = time.perf_counter()
                x_cpu, info_cpu = spla.bicgstab(A_cpu, b_cpu, rtol=tol)
                cpu_times.append(time.perf_counter() - t0_cpu)

                # GPU Timing
                t0_gpu = time.perf_counter()
                x_gpu, info_gpu = bicgstab_cupy(A_gpu, b_gpu, rtol=tol)
                cp.cuda.Stream.null.synchronize() 
                gpu_times.append(time.perf_counter() - t0_gpu)

                print(f'{prec_key} - {N}x{N}: Run {_+1}/{num_runs}')

            # Calculate statistics based on the speedups of individual runs
            cpu_times = np.array(cpu_times)
            gpu_times = np.array(gpu_times)
            speedups = cpu_times / gpu_times

            avg_spd = np.mean(speedups)
            std_spd = np.std(speedups)
            max_spd = np.max(speedups)
            min_spd = np.min(speedups)

            # Store for plotting & table
            plot_data[prec_key]['N'].append(f"{N}x{N}")
            plot_data[prec_key]['avg'].append(avg_spd)
            plot_data[prec_key]['std'].append(std_spd)
            plot_data[prec_key]['max'].append(max_spd)
            plot_data[prec_key]['min'].append(min_spd)

    # --- Print Final Summary Table ---
    if HAS_GPU and plot_data['float32']['N']:
        print("\n" + "="*85)
        print(f"{'FINAL SPEEDUP SUMMARY (CuPy vs SciPy)':^85}")
        print("="*85)
        print(f"{'Grid':<10} | {'Unknowns':<10} | {'FP32 Avg':<12} | {'FP32 Std':<10} | {'FP64 Avg':<12} | {'FP64 Std':<10}")
        print("-" * 85)
        
        for i, grid_str in enumerate(plot_data['float32']['N']):
            # Calculate unknowns from the grid string (e.g. '64x64' -> 4096)
            N_val = int(grid_str.split('x')[0])
            unknowns = N_val**2
            
            fp32_avg = plot_data['float32']['avg'][i]
            fp32_std = plot_data['float32']['std'][i]
            fp64_avg = plot_data['float64']['avg'][i]
            fp64_std = plot_data['float64']['std'][i]
            
            print(f"{grid_str:<10} | {unknowns:<10} | {fp32_avg:<11.2f}x | {fp32_std:<10.2f} | {fp64_avg:<11.2f}x | {fp64_std:<10.2f}")
        print("="*85 + "\n")

    if HAS_GPU:
        plot_results(plot_data)

def plot_results(data):
    """Generates a bar chart with std error bars and min/max scatter points."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    titles = [('float32', 'FP32'), ('float64', 'FP64')]

    for ax, (key, title) in zip(axes, titles):
        prec_data = data[key]
        if not prec_data['N']:
            continue
            
        x_pos = np.arange(len(prec_data['N']))
        avgs = np.array(prec_data['avg'])
        stds = np.array(prec_data['std'])
        maxs = np.array(prec_data['max'])
        mins = np.array(prec_data['min'])

        # Plot the main average bar
        ax.bar(x_pos, avgs, color='skyblue', edgecolor='black', alpha=0.8, label='Average Speedup')
        
        # Add Standard Deviation as traditional error bars
        ax.errorbar(x_pos, avgs, yerr=stds, fmt='none', ecolor='red', capsize=5, elinewidth=2, label='Std Dev')
        
        # Add Min and Max as scatter points
        ax.scatter(x_pos, maxs, color='green', marker='^', s=80, zorder=3, label='Max Speedup')
        ax.scatter(x_pos, mins, color='darkblue', marker='v', s=80, zorder=3, label='Min Speedup')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(prec_data['N'])
        ax.set_xlabel('Grid Size', fontsize=12)
        ax.set_ylabel('Speedup (CuPy time / SciPy time)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_yscale('log')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig('bicgstab_speedup_benchmark.png', dpi=300)
    print("Plot saved successfully to 'bicgstab_speedup_benchmark.png'")
    plt.show()

if __name__ == "__main__":
    run_benchmark(grids=[64, 128, 256, 512], num_runs=10)