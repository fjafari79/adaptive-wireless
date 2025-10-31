#!/usr/bin/env python3
"""
Custom Parameter Comparison Script
Compare four solution methods (Graph, MDP, Estimated, Expected Receivers) with user-specified parameters.
Shows PDF (Probability Distribution Function) comparison in a single plot.

Usage:
    python custom_parameter_comparison.py
    
Then enter your desired p_tcp and p_udp values when prompted.
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

# Set up matplotlib parameters for LaTeX rendering with thesis-appropriate sizes
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 16,  # Axis labels
    "xtick.labelsize": 14, # X tick labels
    "ytick.labelsize": 14, # Y tick labels
    "legend.fontsize": 14, # Legend text
})


# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "simulation_suite"))

# Import the CommunicationSimulator class
from comm_sim import CommunicationSimulator

def get_user_parameters():
    """Get simulation parameters from user input."""
    print("=" * 60)
    print("PySimulator - Custom Parameter Comparison")
    print("=" * 60)
    print()
    print("Enter your desired parameters for the communication simulation:")
    print("- n_messages: Number of messages (1-2 recommended)")
    print("- n_vehicles: Number of vehicles (3-15 recommended)")
    print("- p_tcp: TCP success probability (0.0 to 1.0)")
    print("- p_udp: UDP success probability (0.0 to 1.0)")
    print()
    
    # Get number of messages
    while True:
        try:
            n_messages = int(input("Enter number of messages (e.g., 1): "))
            if 1 <= n_messages <= 3:
                break
            else:
                total_states = 2 ** (n_messages * 10)  # Assuming 10 vehicles
                if total_states > 100000:
                    print(f"Warning: {n_messages} messages would create {total_states} states, which is too large.")
                    print("Please choose 1-2 messages for reasonable computation time.")
                else:
                    break
        except ValueError:
            print("Please enter a valid integer")
    
    # Get number of vehicles
    while True:
        try:
            n_vehicles = int(input("Enter number of vehicles (e.g., 10): "))
            if 1 <= n_vehicles <= 20:
                total_states = 2 ** (n_messages * n_vehicles)
                if total_states > 100000:
                    print(f"Warning: {n_messages} messages × {n_vehicles} vehicles = {total_states} states")
                    print("This is too large for efficient computation. Please reduce parameters.")
                    continue
                else:
                    break
            else:
                print("Please enter a value between 1 and 20")
        except ValueError:
            print("Please enter a valid integer")
    
    # Display state space info
    total_states = 2 ** (n_messages * n_vehicles)
    print(f"Configuration: {n_messages} message(s), {n_vehicles} vehicles = {total_states} total states")
    
    # Get TCP probability
    while True:
        try:
            p_tcp = float(input("Enter p_tcp (e.g., 0.95): "))
            if 0.0 <= p_tcp <= 1.0:
                break
            else:
                print("Please enter a value between 0.0 and 1.0")
        except ValueError:
            print("Please enter a valid number")
    
    # Get UDP probability
    while True:
        try:
            p_udp = float(input("Enter p_udp (e.g., 0.8): "))
            if 0.0 <= p_udp <= 1.0:
                break
            else:
                print("Please enter a value between 0.0 and 1.0")
        except ValueError:
            print("Please enter a valid number")
    
    return n_messages, n_vehicles, p_tcp, p_udp

def run_custom_simulation(n_messages: int, n_vehicles: int, p_tcp: float, p_udp: float, num_runs: int = 1000):
    """Run simulation with custom parameters for all four solution methods."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create simulator instance with user-specified parameters
    simulator = CommunicationSimulator(n_messages=n_messages, n_vehicles=n_vehicles)
    
    # Load strategies silently
    graph_strategy = simulator.load_strategy("graph", p_tcp, p_udp)
    mdp_strategy = simulator.load_strategy("mdp", p_tcp, p_udp)
    estimated_strategy = simulator.load_strategy("estimated", p_tcp, p_udp)
    expected_receivers_strategy = simulator.load_strategy("expected_receivers", p_tcp, p_udp)
    
    # Initialize results storage with TCP/UDP breakdown
    results = {
        'graph': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'success_count': 0},
        'mdp': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'success_count': 0},
        'estimated': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'success_count': 0},
        'expected_receivers': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'success_count': 0}
    }
    
    # Run simulations for all strategies silently
    # Redirect stdout to suppress debug messages
    null_output = StringIO()
    for strategy_name, strategy in [
        ("graph", graph_strategy),
        ("mdp", mdp_strategy),
        ("estimated", estimated_strategy),
        ("expected_receivers", expected_receivers_strategy)
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

def plot_pdf_comparison(results: Dict, n_messages: int, n_vehicles: int, p_tcp: float, p_udp: float, save_plot: bool = True):
    """Create a single plot comparing PDFs of all four solution methods with TCP/UDP breakdown."""
    
    # Strategy configuration
    strategies = {
        'graph': {'name': 'Graph Shortest Path', 'color': '#2E8B57', 'alpha': 0.8},
        'mdp': {'name': 'MDP Policy', 'color': '#FF6B35', 'alpha': 0.8},
        'estimated': {'name': 'Estimated Heuristic', 'color': '#4169E1', 'alpha': 0.8},
        'expected_receivers': {'name': 'Expected Receivers', 'color': '#9932CC', 'alpha': 0.8}
    }
    
    # Create figure with thesis-appropriate size
    plt.figure(figsize=(10, 7))
    
    # Find the actual maximum transmission count observed across all strategies
    max_transmissions = 1
    for strategy in strategies.keys():
        if results[strategy]['transmissions']:
            max_transmissions = max(max_transmissions, max(results[strategy]['transmissions']))
    
    # No buffer needed, use exactly what we observe
    
    # Calculate bin positions for grouped bars with gaps between transmission counts
    n_strategies = len([s for s in strategies.keys() if results[s]['transmissions']])
    
    # Create grouped bar positions
    # Each group (transmission count) will have all strategies clustered together
    # with gaps between different transmission counts
    group_width = n_strategies * 0.2  # Total width of each group
    bar_width = 0.15  # Width of individual bars
    gap_between_groups = 0.3  # Gap between transmission count groups
    
    # Calculate positions for each transmission count group
    transmission_counts = list(range(1, max_transmissions + 1))
    group_positions = []
    
    for i, trans_count in enumerate(transmission_counts):
        center_pos = i * (group_width + gap_between_groups)
        group_positions.append(center_pos)
    
    # Plot PDF bars for each strategy with TCP/UDP breakdown
    for idx, (strategy_key, strategy_config) in enumerate(strategies.items()):
        transmissions = results[strategy_key]['transmissions']
        tcp_transmissions = results[strategy_key]['tcp_transmissions']
        udp_transmissions = results[strategy_key]['udp_transmissions']
        
        if not transmissions:
            continue
        
        # Cap transmission values at max_transmissions
        capped_transmissions = [min(x, max_transmissions) for x in transmissions]
        
        # Calculate PDF for each bin with TCP/UDP breakdown
        total_pdf_values = np.zeros(max_transmissions)
        tcp_pdf_values = np.zeros(max_transmissions)
        udp_pdf_values = np.zeros(max_transmissions)
        
        for i, total_trans in enumerate(capped_transmissions):
            if 1 <= total_trans <= max_transmissions:
                bin_idx = total_trans - 1
                total_pdf_values[bin_idx] += 1
                
                # For this simulation run, calculate the proportion of TCP/UDP
                tcp_count = tcp_transmissions[i]
                udp_count = udp_transmissions[i]
                total_count = tcp_count + udp_count
                
                if total_count > 0:
                    # Calculate the proportion for this single run
                    tcp_proportion = tcp_count / total_count
                    udp_proportion = udp_count / total_count
                    
                    # Add this run's contribution to the PDF (will normalize later)
                    tcp_pdf_values[bin_idx] += tcp_proportion
                    udp_pdf_values[bin_idx] += udp_proportion
                else:
                    # If no transmissions recorded, distribute equally
                    tcp_pdf_values[bin_idx] += 0.5
                    udp_pdf_values[bin_idx] += 0.5
        
        # Normalize all PDF values to get probabilities
        total_pdf_values = total_pdf_values / len(transmissions)
        tcp_pdf_values = tcp_pdf_values / len(transmissions)
        udp_pdf_values = udp_pdf_values / len(transmissions)
        
        # Calculate positions for this strategy within each group
        strategy_offset = (idx - (n_strategies - 1) / 2) * bar_width
        x_positions = [group_pos + strategy_offset for group_pos in group_positions]
        
        # Create stacked bars - TCP on bottom (solid), UDP on top (hatched)
        tcp_bars = plt.bar(x_positions, tcp_pdf_values, bar_width,
                          label=f'{strategy_config["name"]} - TCP',
                          color=strategy_config['color'],
                          alpha=strategy_config['alpha'],
                          edgecolor='black',
                          linewidth=0.5)
        
        udp_bars = plt.bar(x_positions, udp_pdf_values, bar_width, bottom=tcp_pdf_values,
                          label=f'{strategy_config["name"]} - UDP',
                          color=strategy_config['color'],
                          alpha=strategy_config['alpha'] * 0.6,
                          edgecolor='black',
                          linewidth=0.5,
                          hatch='///')  # Cross hatching for UDP
        
        # Calculate statistics
        mean_trans = np.mean(transmissions)
        std_trans = np.std(transmissions)
        mean_tcp = np.mean(tcp_transmissions)
        mean_udp = np.mean(udp_transmissions)
        tcp_percentage = mean_tcp / mean_trans * 100 if mean_trans > 0 else 0
        udp_percentage = mean_udp / mean_trans * 100 if mean_trans > 0 else 0
        success_rate = results[strategy_key]['success_count'] / len(transmissions)
        
        print(f"{strategy_config['name']} Statistics:")
        print(f"  Mean transmissions: {mean_trans:.2f} ± {std_trans:.2f}")
        print(f"  TCP transmissions: {mean_tcp:.2f} ({tcp_percentage:.1f}%)")
        print(f"  UDP transmissions: {mean_udp:.2f} ({udp_percentage:.1f}%)")
        print(f"  Range: [{min(transmissions)}, {max(transmissions)}]")
        print(f"  Success rate: {success_rate:.3f} ({results[strategy_key]['success_count']}/{len(transmissions)})")
        print()
    
    # Formatting with LaTeX for thesis
    plt.xlabel(r'\textbf{Total Transmissions Required}', labelpad=10)
    plt.ylabel(r'\textbf{Probability}', labelpad=10)
    
    # Set x-axis with grouped positioning - only show observed transmission counts
    plt.xlim(-0.5, group_positions[-1] + 0.5)
    plt.xticks(group_positions, [str(i) for i in range(1, max_transmissions + 1)])
    
    # Format y-axis as decimal probabilities
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # Make tick labels more prominent
    plt.tick_params(axis='both', which='major', pad=8)
    
    # Create custom legend
    legend_elements = []
    for strategy_key, strategy_config in strategies.items():
        if results[strategy_key]['transmissions']:
            # Unicast (TCP) legend entry (solid)
            legend_elements.append(plt.Rectangle((0,0),1,1, 
                                               facecolor=strategy_config['color'], 
                                               alpha=strategy_config['alpha'],
                                               edgecolor='black',
                                               label=fr'{strategy_config["name"]} - Unicast'))
            # Broadcast (UDP) legend entry (hatched)
            legend_elements.append(plt.Rectangle((0,0),1,1, 
                                               facecolor=strategy_config['color'], 
                                               alpha=strategy_config['alpha'] * 0.6,
                                               hatch='///',
                                               edgecolor='black',
                                               label=fr'{strategy_config["name"]} - Broadcast'))
    
    # Place legend inside the plot in the upper right corner
    legend = plt.legend(handles=legend_elements, fontsize=12, 
                       loc='upper right')
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Make the plot look professional
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        # Create custom results directory
        results_dir = os.path.join(os.path.dirname(__file__), "custom_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save as high-quality PDF (vector graphics, best for publications)
        filename_pdf = f'pdf_tcp_udp_breakdown_p_tcp_{p_tcp:.3f}_p_udp_{p_udp:.3f}.pdf'
        filepath_pdf = os.path.join(results_dir, filename_pdf)
        plt.savefig(filepath_pdf, format='pdf', bbox_inches='tight', dpi=1200,
                   facecolor='white', edgecolor='none', metadata={'Creator': 'Matplotlib'})
        
        # Also save as PNG for quick viewing
        filename_png = f'pdf_tcp_udp_breakdown_p_tcp_{p_tcp:.3f}_p_udp_{p_udp:.3f}.png'
        filepath_png = os.path.join(results_dir, filename_png)
        plt.savefig(filepath_png, format='png', bbox_inches='tight', dpi=600,
                   facecolor='white', edgecolor='none')
        
        print(f"Plots saved to:\n PDF: {filepath_pdf}\n PNG: {filepath_png}")
    
    plt.show()

def main():
    """Main function to run custom parameter comparison."""
    
    # Get parameters from user
    n_messages, n_vehicles, p_tcp, p_udp = get_user_parameters()
    num_runs = int(input("\nEnter number of simulation runs (default 1000): ") or "1000")
    
    # Run simulation and create plot
    results = run_custom_simulation(n_messages, n_vehicles, p_tcp, p_udp, num_runs)
    plot_pdf_comparison(results, n_messages, n_vehicles, p_tcp, p_udp, save_plot=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you're running this script from the PySimulator root directory.")
