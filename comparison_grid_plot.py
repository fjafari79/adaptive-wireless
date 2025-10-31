#!/usr/bin/env python3
"""
Comparison Grid Plot Script
Creates a 2x2 grid of plots comparing four solution methods with different UDP probabilities.
All plots share y-axis and a common legend is placed on top.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "simulation_suite"))

# Import the CommunicationSimulator class
from comm_sim import CommunicationSimulator

def run_simulation(n_messages: int, n_vehicles: int, p_tcp: float, p_udp: float, num_runs: int = 10000) -> Dict:
    """Run simulation with given parameters for all four solution methods."""
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create simulator instance
    simulator = CommunicationSimulator(n_messages=n_messages, n_vehicles=n_vehicles)
    
    # Initialize results storage
    results = {
        'graph': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'success_count': 0},
        'mdp': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'success_count': 0},
        'estimated': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'success_count': 0},
        'expected_receivers': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'success_count': 0}
    }
    
    # Run simulations for all strategies silently
    null_output = StringIO()
    for strategy_name, strategy in [
        ("graph", simulator.load_strategy("graph", p_tcp, p_udp)),
        ("mdp", simulator.load_strategy("mdp", p_tcp, p_udp)),
        ("estimated", simulator.load_strategy("estimated", p_tcp, p_udp)),
        ("expected_receivers", simulator.load_strategy("expected_receivers", p_tcp, p_udp))
    ]:
        if strategy is not None:
            for _ in range(num_runs):
                with redirect_stdout(null_output), redirect_stderr(null_output):
                    transmissions, steps, success, tcp_trans, udp_trans = simulator.simulate_single_run(
                        strategy, strategy_name, p_tcp, p_udp
                    )
                
                results[strategy_name]['transmissions'].append(transmissions)
                results[strategy_name]['tcp_transmissions'].append(tcp_trans)
                results[strategy_name]['udp_transmissions'].append(udp_trans)
                if success:
                    results[strategy_name]['success_count'] += 1
    
    return results

def create_subplot(results: Dict, ax: plt.Axes, p_udp: float, show_legend: bool = False, max_transmissions_override: Optional[int] = None) -> None:
    """Create a single subplot with the given results."""
    strategies = {
        'graph': {'name': 'Graph Shortest Path', 'color': '#2E8B57', 'alpha': 0.8},
        'mdp': {'name': 'MDP Policy', 'color': '#FF6B35', 'alpha': 0.8},
        'estimated': {'name': 'Estimated Heuristic', 'color': '#4169E1', 'alpha': 0.8},
        'expected_receivers': {'name': 'Expected Receivers', 'color': '#9932CC', 'alpha': 0.8}
    }
    
    # Use provided max_transmissions or calculate it
    if max_transmissions_override is not None:
        max_transmissions = max_transmissions_override
    else:
        max_transmissions = 1
        for strategy in strategies.keys():
            if results[strategy]['transmissions']:
                max_transmissions = max(max_transmissions, max(results[strategy]['transmissions']))
    
    # Calculate positions for grouped bars
    n_strategies = len([s for s in strategies.keys() if results[s]['transmissions']])
    group_width = n_strategies * 0.3  # Increased from 0.25
    bar_width = 0.25  # Increased from 0.2
    gap_between_groups = 0.15  # Reduced from 0.2
    transmission_counts = list(range(1, max_transmissions + 1))
    group_positions = [i * (group_width + gap_between_groups) for i in range(len(transmission_counts))]
    
    legend_elements = []
    for idx, (strategy_key, strategy_config) in enumerate(strategies.items()):
        transmissions = results[strategy_key]['transmissions']
        tcp_transmissions = results[strategy_key]['tcp_transmissions']
        udp_transmissions = results[strategy_key]['udp_transmissions']
        
        if not transmissions:
            continue
            
        strategy_offset = (idx - (n_strategies - 1) / 2) * bar_width
        x_positions = [group_pos + strategy_offset for group_pos in group_positions]
        
        # Calculate PDF values
        total_pdf_values = np.zeros(max_transmissions)
        tcp_pdf_values = np.zeros(max_transmissions)
        udp_pdf_values = np.zeros(max_transmissions)
        
        for i, total_trans in enumerate(transmissions):
            if 1 <= total_trans <= max_transmissions:
                bin_idx = total_trans - 1
                total_pdf_values[bin_idx] += 1
                
                tcp_count = tcp_transmissions[i]
                udp_count = udp_transmissions[i]
                total_count = tcp_count + udp_count
                
                if total_count > 0:
                    tcp_proportion = tcp_count / total_count
                    udp_proportion = udp_count / total_count
                    tcp_pdf_values[bin_idx] += tcp_proportion
                    udp_pdf_values[bin_idx] += udp_proportion
                else:
                    tcp_pdf_values[bin_idx] += 0.5
                    udp_pdf_values[bin_idx] += 0.5
        
        total_pdf_values = total_pdf_values / len(transmissions)
        tcp_pdf_values = tcp_pdf_values / len(transmissions)
        udp_pdf_values = udp_pdf_values / len(transmissions)
        
        # Create bars
        tcp_bars = ax.bar(x_positions, tcp_pdf_values, bar_width,
                         color=strategy_config['color'],
                         alpha=strategy_config['alpha'],
                         edgecolor='black',
                         linewidth=0.5)
        
        udp_bars = ax.bar(x_positions, udp_pdf_values, bar_width,
                         bottom=tcp_pdf_values,
                         color=strategy_config['color'],
                         alpha=strategy_config['alpha'] * 0.6,
                         edgecolor='black',
                         linewidth=0.5,
                         hatch='///')
        
        if show_legend:
            legend_elements.extend([
                plt.Rectangle((0,0),1,1, 
                            facecolor=strategy_config['color'],
                            alpha=strategy_config['alpha'],
                            edgecolor='black',
                            label=f'{strategy_config["name"]} - Unicast'),
                plt.Rectangle((0,0),1,1,
                            facecolor=strategy_config['color'],
                            alpha=strategy_config['alpha'] * 0.6,
                            hatch='///',
                            edgecolor='black',
                            label=f'{strategy_config["name"]} - Broadcast')
            ])
    
    # Formatting
    ax.set_xlim(-0.5, group_positions[-1] + 0.5)
    ax.set_xticks(group_positions)
    ax.set_xticklabels([str(i) for i in range(1, max_transmissions + 1)])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    # Increase tick label padding significantly to avoid overlap in PDF
    ax.tick_params(axis='both', which='major', pad=15)  # Increased padding for tick labels
    
    # Add p_udp as text in the top-left corner of the plot
    ax.text(0.05, 0.95, f'$p_{{b}}={p_udp:.2f}$',
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=18)
    
    # Return legend elements if needed
    if show_legend:
        return legend_elements
    
    return legend_elements if show_legend else None

def main():
    """Main function to create grid of plots."""
    # Set up matplotlib parameters for LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 24,  # Increased
        "xtick.labelsize": 20,  # Increased
        "ytick.labelsize": 20,  # Increased
        "legend.fontsize": 18,  # Increased
    })
    
    # Fixed parameters
    n_messages = 1
    n_vehicles = 10
    p_tcp = 0.9
    udp_probabilities = [0.42, 0.43, 0.44, 0.45]
    num_runs = 10000
    
    # Create figure with 2x2 subplots with larger size optimized for PDF
    fig, axs = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    axs = axs.ravel()
    
    # First run all simulations and find global maximum
    all_results = []
    global_max_transmissions = 1
    
    print("Running simulations...")
    for p_udp in udp_probabilities:
        results = run_simulation(n_messages, n_vehicles, p_tcp, p_udp, num_runs)
        all_results.append(results)
        
        # Update global maximum
        for strategy in results:
            if results[strategy]['transmissions']:
                global_max_transmissions = max(global_max_transmissions, 
                                            max(results[strategy]['transmissions']))
    
    print(f"Maximum transmissions across all simulations: {global_max_transmissions}")
    
    # Create all plots using fixed maximum of 18 transmissions
    # Get legend elements from first plot
    legend_elements = create_subplot(all_results[0], axs[0], udp_probabilities[0], 
                                   show_legend=True, max_transmissions_override=16)
    
    # Create remaining plots without individual legends
    for idx in range(1, len(udp_probabilities)):
        create_subplot(all_results[idx], axs[idx], udp_probabilities[idx], 
                      show_legend=False, max_transmissions_override=16)
    
    # Create legend at the top of the figure
    if legend_elements:
        # Put legend above the plots with more rows and fewer columns
        legend = fig.legend(handles=legend_elements,
                          loc='center',
                          bbox_to_anchor=(0.5, 0.98),  # Moved closer to plots
                          ncol=2,  # Use 2 columns for 8 items = 4 rows
                          fontsize=18,
                          frameon=False,
                          columnspacing=1.0,
                          handletextpad=0.5)

    # Adjust subplot positioning with minimal space for legend
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.92, wspace=0.1, hspace=0.1)  # Increased top margin further    # Add common labels
    fig.text(0.5, 0.04, r'\textbf{Total Transmissions Required}', ha='center', va='center', fontsize=24)
    fig.text(0.02, 0.5, r'\textbf{Probability}', va='center', ha='center', rotation='vertical', fontsize=24)    # Save plot in high-quality PDF format
    results_dir = os.path.join(os.path.dirname(__file__), "custom_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as PDF for vector graphics (best for publications)
    filename_pdf = f'comparison_grid_m{n_messages}_v{n_vehicles}_ptcp{p_tcp:.3f}.pdf'
    filepath_pdf = os.path.join(results_dir, filename_pdf)
    plt.savefig(filepath_pdf, format='pdf', bbox_inches='tight', dpi=1200,
                facecolor='white', edgecolor='none', metadata={'Creator': 'Matplotlib'})
    
    # Also save high-resolution PNG as backup
    filename_png = f'comparison_grid_m{n_messages}_v{n_vehicles}_ptcp{p_tcp:.3f}.png'
    filepath_png = os.path.join(results_dir, filename_png)
    plt.savefig(filepath_png, format='png', bbox_inches='tight', dpi=600,
                facecolor='white', edgecolor='none')
    print(f"Plots saved to:\n PDF: {filepath_pdf}\n PNG: {filepath_png}")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPlot generation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")