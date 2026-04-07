import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set up publication-quality plot aesthetics
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
    "font.size": 12
})

def plot_sweep_data(csv_filename='output/kerr_sweep_data0.csv'):
    if not os.path.exists(csv_filename):
        print(f"Error: Could not find '{csv_filename}'. Make sure the simulation has finished generating the data.")
        return

    # Load the CSV data
    df = pd.read_csv(csv_filename)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2ca02c', '#d62728', '#1f77b4', '#f5b041'] # Distinct colors for different amplitudes

    # Group by Source Amplitude and plot each curve
    for i, (amp, group_data) in enumerate(df.groupby('Source_Amplitude')):
        w = group_data['Frequency_omega']
        
        # We normalize the Max Absolute E-field by the source amplitude 
        # so the linear and nonlinear responses can be plotted on the same scale
        normalized_E = group_data['Max_Abs_E'] / amp
        
        color = colors[i % len(colors)]
        ax.plot(w, normalized_E, linewidth=2.5, color=color, label=rf'$J_{{amp}}$ = {amp}')

    # Plot theoretical linear vacuum resonance line
    w_11 = np.pi * np.sqrt(2)
    # ax.axvline(x=w_11, color='black', linestyle='--', alpha=0.7, label=r'Linear Vacuum $\omega_{11}$')

    # Formatting the axes and legend
    ax.set_xlabel(r'Frequency $\omega$ (dimensionless)', fontsize=14)
    ax.set_ylabel(r'Normalized E-Field ($\max|\mathbf{E}| \, / \, J_{amp}$)', fontsize=14)
    ax.set_title('Power-Dependent Resonance Shift (Kerr Effect)', fontsize=14, pad=15)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=12, loc='best')

    # Save and display
    output_img = 'extra/kerr_sweep_plot.png'
    fig.savefig(output_img, dpi=200, bbox_inches='tight')
    print(f"SUCCESS! Plot saved to {output_img}")
    
    plt.show()

if __name__ == "__main__":
    plot_sweep_data()
