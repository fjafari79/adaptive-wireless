#!/usr/bin/env python3
"""
Custom Parameter Comparison Script - Modified Simulation Version
Compare four solution methods (Graph, MDP, Estimated, Expected Receivers) with user-specified parameters.
Shows PDF (Probability Distribution Function) and CDF (Cumulative Distribution Function) comparison plots.

State Encoding:
- 0 = vehicle already has/received the message
- 1 = vehicle needs to receive the message
- Terminal state: all zeros (all vehicles have all messages)

Initial State Options:
- All-zeros: All vehicles already have all messages (simulation would be complete)
- All-ones: All vehicles need all messages (realistic starting point)
- Random: Each vehicle randomly needs/doesn't need each message
- Partial: Some percentage of vehicles need messages
- Clustered: Some vehicles need messages, others already have them

Usage:
    python custom_parameter_comparison.py
    
Then enter your desired parameters when prompted.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Time factor for TCP actions (TCP takes longer than UDP) - must match comm_sim.py
TCP_TIME_FACTOR = 1.2  # TCP actions take 20% more time than UDP actions

# Import the CommunicationSimulator class from current directory
from comm_sim import CommunicationSimulator

def get_user_parameters():
    """Get simulation parameters from user input."""
    print("=" * 60)
    print("PySimulator - Custom Parameter Comparison (Modified Version)")
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
                print(f"Please enter a value between 1 and 3. You entered: {n_messages}")
                print("Values above 3 create too many states for efficient computation.")
                continue  # This was missing! Ask again
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

def get_bin_size():
    """Get bin size preference from user."""
    print("\nChoose histogram bin size:")
    print("1. Fine (0.1) - Very detailed, many narrow bins")
    print("2. Medium (0.2) - Balanced detail (default)")
    print("3. Coarse (0.5) - Fewer, wider bins") 
    print("4. Very Coarse (1.0) - Minimal bins")
    print("5. Custom - Enter your own bin size")
    
    while True:
        try:
            choice = input("Enter choice (1-5, default 2): ").strip() or "2"
            
            if choice == "1":
                return 0.1
            elif choice == "2":
                return 0.2
            elif choice == "3":
                return 0.5
            elif choice == "4":
                return 1.0
            elif choice == "5":
                custom_bin = float(input("Enter custom bin size (e.g., 0.3): "))
                if 0.01 <= custom_bin <= 2.0:
                    return custom_bin
                else:
                    print("Please enter a bin size between 0.01 and 2.0")
            else:
                print("Please enter 1, 2, 3, 4, or 5")
        except ValueError:
            print("Please enter a valid number")

def get_initial_state_option():
    """Get user preference for initial state generation."""
    print("\nChoose initial state generation:")
    print("1. All-zeros (all vehicles have received all messages) - Simulation complete")
    print("2. All-ones (all vehicles need to receive all messages) - Realistic start")
    print("3. Random (each vehicle randomly needs/doesn't need each message)")
    print("4. Partial coverage (some percentage of vehicles still need messages)")
    print("5. Clustered (some vehicles need messages, others already have them)")
    
    while True:
        try:
            choice = input("Enter choice (1-5, default 1): ").strip() or "1"
            
            if choice == "1":
                return "all_zeros", {}
            elif choice == "2":
                return "all_ones", {}
            elif choice == "3":
                prob = float(input("Enter probability each vehicle has each message (0.0-1.0, e.g., 0.3): "))
                if 0.0 <= prob <= 1.0:
                    return "random", {"probability": prob}
                else:
                    print("Please enter a probability between 0.0 and 1.0")
            elif choice == "4":
                coverage = float(input("Enter percentage of vehicles with full coverage (0.0-1.0, e.g., 0.2): "))
                if 0.0 <= coverage <= 1.0:
                    return "partial", {"coverage": coverage}
                else:
                    print("Please enter a coverage between 0.0 and 1.0")
            elif choice == "5":
                cluster_size = float(input("Enter fraction of vehicles in informed cluster (0.0-1.0, e.g., 0.3): "))
                if 0.0 <= cluster_size <= 1.0:
                    return "clustered", {"cluster_size": cluster_size}
                else:
                    print("Please enter a cluster size between 0.0 and 1.0")
            else:
                print("Please enter 1, 2, 3, 4, or 5")
        except ValueError:
            print("Please enter a valid number")

def generate_initial_state(n_messages, n_vehicles, state_type, params):
    """
    Generate initial state based on the specified type and parameters.
    
    Returns:
        tuple: (initial_state, description) where initial_state is a binary state vector
    """
    state_size = n_messages * n_vehicles
    
    if state_type == "all_zeros":
        # All vehicles already have all messages (simulation would be complete)
        initial_state = tuple([0] * state_size)
        description = "All-zeros (all vehicles already have all messages)"
        
    elif state_type == "all_ones":
        # Most realistic: all vehicles need all messages initially
        initial_state = tuple([1] * state_size)
        description = "All-ones (all vehicles need all messages)"
        
    elif state_type == "random":
        # Each bit is randomly 0 or 1 with given probability
        # 0 = already has message, 1 = needs message
        prob = params["probability"]
        state_bits = []
        for i in range(state_size):
            if random.random() < prob:
                state_bits.append(1)  # needs message
            else:
                state_bits.append(0)  # already has message
        initial_state = tuple(state_bits)
        description = f"Random (p={prob:.2f} for needing each message)"
        
    elif state_type == "partial":
        # Some percentage of vehicles need no messages, others need all messages
        coverage = params["coverage"]
        n_needing_messages = int(coverage * n_vehicles)
        
        state_bits = [0] * state_size  # Start with all vehicles having all messages
        
        # Randomly select which vehicles need messages
        vehicles_needing_messages = random.sample(range(n_vehicles), n_needing_messages)
        
        for vehicle in vehicles_needing_messages:
            for message in range(n_messages):
                bit_index = message * n_vehicles + vehicle
                state_bits[bit_index] = 1  # This vehicle needs this message
                
        initial_state = tuple(state_bits)
        description = f"Partial coverage ({coverage:.1%} vehicles need messages)"
        
    elif state_type == "clustered":
        # Create clusters: some vehicles need all messages, others already have all
        cluster_size = params["cluster_size"]
        n_needing_messages = int(cluster_size * n_vehicles)
        
        state_bits = [0] * state_size  # Start with all vehicles having all messages
        
        # First n_needing_messages vehicles need all messages
        for vehicle in range(n_needing_messages):
            for message in range(n_messages):
                bit_index = message * n_vehicles + vehicle
                state_bits[bit_index] = 1  # This vehicle needs this message
                
        initial_state = tuple(state_bits)
        description = f"Clustered ({cluster_size:.1%} vehicles need messages, others already have them)"
        
    else:
        # Fallback to all-zeros
        initial_state = tuple([0] * state_size)
        description = "All-zeros (fallback)"
    
    return initial_state, description

def run_custom_simulation(n_messages: int, n_vehicles: int, p_tcp: float, p_udp: float, num_runs: int = 1000, 
                         initial_state_type: str = "all_zeros", initial_state_params: dict = {}):
    """Run simulation with custom parameters for all four solution methods."""
    
    # Use current time as seed for different results each run
    import time
    current_seed = int(time.time())
    random.seed(current_seed)
    np.random.seed(current_seed)
    print(f"Using random seed: {current_seed} (based on current time)")
    
    # Generate initial state description
    sample_initial_state, initial_state_description = generate_initial_state(n_messages, n_vehicles, initial_state_type, initial_state_params)
    
    # Create simulator instance with user-specified parameters
    simulator = CommunicationSimulator(n_messages=n_messages, n_vehicles=n_vehicles)
    
    print(f"\nRunning simulation with:")
    print(f"  Messages: {simulator.n_messages}")
    print(f"  Vehicles: {simulator.n_vehicles}")
    print(f"  p_tcp: {p_tcp}")
    print(f"  p_udp: {p_udp}")
    print(f"  Simulation runs: {num_runs}")
    print(f"  Initial state: {initial_state_description}")
    print(f"  Sample initial state vector: {sample_initial_state}")
    print(f"  State encoding: 0 = already has message, 1 = needs message")
    
    # Add detailed state breakdown for better understanding
    print(f"  State vector format: [{n_messages} messages × {n_vehicles} vehicles = {len(sample_initial_state)} bits]")
    if n_messages <= 3 and n_vehicles <= 15:  # Only show details for reasonable configurations
        print(f"  State breakdown:")
        print(f"    Vector: {sample_initial_state}")
        print(f"    Matrix format (rows=messages, cols=vehicles):")
        
        # Convert to matrix format for display
        state_matrix = []
        for msg in range(n_messages):
            start_idx = msg * n_vehicles
            end_idx = start_idx + n_vehicles
            msg_state = list(sample_initial_state[start_idx:end_idx])
            state_matrix.append(msg_state)
        
        # Print matrix with nice formatting
        print(f"         Vehicles: {' '.join(f'{i:2d}' for i in range(n_vehicles))}")
        for msg in range(n_messages):
            print(f"    Message {msg}: {' '.join(f'{bit:2d}' for bit in state_matrix[msg])}")
        print(f"    (0 = has message, 1 = needs message)")
    
    print()
    
    # Load strategies
    print("Loading/generating solution strategies...")
    graph_strategy = simulator.load_strategy("graph", p_tcp, p_udp)
    mdp_strategy = simulator.load_strategy("mdp", p_tcp, p_udp)
    estimated_strategy = simulator.load_strategy("estimated", p_tcp, p_udp)
    expected_receivers_strategy = simulator.load_strategy("expected_receivers", p_tcp, p_udp)
    
    # Debug: Check if Expected Receivers strategy loaded
    print(f"Strategy loading results:")
    print(f"  Graph: {'✓' if graph_strategy is not None else '✗'}")
    print(f"  MDP: {'✓' if mdp_strategy is not None else '✗'}")
    print(f"  Estimated: {'✓' if estimated_strategy is not None else '✗'}")
    print(f"  Expected Receivers: {'✓' if expected_receivers_strategy is not None else '✗'}")
    
    # Initialize results storage with TCP/UDP breakdown and time tracking
    results = {
        'graph': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'total_times': [], 'success_count': 0},
        'mdp': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'total_times': [], 'success_count': 0},
        'estimated': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'total_times': [], 'success_count': 0},
        'expected_receivers': {'transmissions': [], 'tcp_transmissions': [], 'udp_transmissions': [], 'total_times': [], 'success_count': 0}
    }
    
    # Run Graph simulations
    if graph_strategy is not None:
        print(f"Running {num_runs} Graph simulations...")
        for run in range(num_runs):
            if (run + 1) % 200 == 0:
                print(f"  Graph run {run + 1}/{num_runs}")
            
            # Generate custom initial state for this run
            initial_state, _ = generate_initial_state(n_messages, n_vehicles, initial_state_type, initial_state_params)
            
            transmissions, steps, success, tcp_trans, udp_trans, total_time = simulator.simulate_single_run(
                graph_strategy, "graph", p_tcp, p_udp, initial_state=initial_state
            )
            
            results['graph']['transmissions'].append(transmissions)
            results['graph']['tcp_transmissions'].append(tcp_trans)
            results['graph']['udp_transmissions'].append(udp_trans)
            results['graph']['total_times'].append(total_time)
            if success:
                results['graph']['success_count'] += 1
    else:
        print("Graph strategy not available - skipping Graph simulations")
    
    # Run MDP simulations
    if mdp_strategy is not None:
        print(f"Running {num_runs} MDP simulations...")
        for run in range(num_runs):
            if (run + 1) % 200 == 0:
                print(f"  MDP run {run + 1}/{num_runs}")
            
            # Generate custom initial state for this run
            initial_state, _ = generate_initial_state(n_messages, n_vehicles, initial_state_type, initial_state_params)
            
            transmissions, steps, success, tcp_trans, udp_trans, total_time = simulator.simulate_single_run(
                mdp_strategy, "mdp", p_tcp, p_udp, initial_state=initial_state
            )
            
            results['mdp']['transmissions'].append(transmissions)
            results['mdp']['tcp_transmissions'].append(tcp_trans)
            results['mdp']['udp_transmissions'].append(udp_trans)
            results['mdp']['total_times'].append(total_time)
            if success:
                results['mdp']['success_count'] += 1
    else:
        print("MDP strategy not available - skipping MDP simulations")
    
    # Run Estimated simulations
    if estimated_strategy is not None:
        print(f"Running {num_runs} Estimated simulations...")
        for run in range(num_runs):
            if (run + 1) % 200 == 0:
                print(f"  Estimated run {run + 1}/{num_runs}")
            
            # Generate custom initial state for this run
            initial_state, _ = generate_initial_state(n_messages, n_vehicles, initial_state_type, initial_state_params)
            
            transmissions, steps, success, tcp_trans, udp_trans, total_time = simulator.simulate_single_run(
                estimated_strategy, "estimated", p_tcp, p_udp, initial_state=initial_state
            )
            
            results['estimated']['transmissions'].append(transmissions)
            results['estimated']['tcp_transmissions'].append(tcp_trans)
            results['estimated']['udp_transmissions'].append(udp_trans)
            results['estimated']['total_times'].append(total_time)
            if success:
                results['estimated']['success_count'] += 1
    else:
        print("Estimated strategy not available - skipping Estimated simulations")
    
    # Run Expected Receivers simulations
    if expected_receivers_strategy is not None:
        print(f"Running {num_runs} Expected Receivers simulations...")
        for run in range(num_runs):
            if (run + 1) % 200 == 0:
                print(f"  Expected Receivers run {run + 1}/{num_runs}")
            
            # Generate custom initial state for this run
            initial_state, _ = generate_initial_state(n_messages, n_vehicles, initial_state_type, initial_state_params)
            
            transmissions, steps, success, tcp_trans, udp_trans, total_time = simulator.simulate_single_run(
                expected_receivers_strategy, "expected_receivers", p_tcp, p_udp, initial_state=initial_state
            )
            
            results['expected_receivers']['transmissions'].append(transmissions)
            results['expected_receivers']['tcp_transmissions'].append(tcp_trans)
            results['expected_receivers']['udp_transmissions'].append(udp_trans)
            results['expected_receivers']['total_times'].append(total_time)
            if success:
                results['expected_receivers']['success_count'] += 1
    else:
        print("Expected Receivers strategy not available - skipping Expected Receivers simulations")
    
    return results

def plot_pdf_comparison(results: Dict, n_messages: int, n_vehicles: int, p_tcp: float, p_udp: float, bin_size: float = 0.2, save_plot: bool = True):
    """Create a single plot comparing PDFs of all four solution methods based on total time taken."""
    
    # Debug: Check which strategies have results
    print(f"\nPlotting results for strategies:")
    for strategy_key in results.keys():
        result_count = len(results[strategy_key]['transmissions'])
        print(f"  {strategy_key}: {result_count} results")
    
    # Strategy configuration
    strategies = {
        'graph': {'name': 'Graph Shortest Path', 'color': '#2E8B57', 'alpha': 0.8},
        'mdp': {'name': 'MDP Policy', 'color': '#FF6B35', 'alpha': 0.8},
        'estimated': {'name': 'Estimated Heuristic', 'color': '#4169E1', 'alpha': 0.8},
        'expected_receivers': {'name': 'Expected Receivers', 'color': '#9932CC', 'alpha': 0.8}
    }
    
    # Create figure
    plt.figure(figsize=(16, 8))
    
    # Find the maximum time across all strategies for binning
    max_time = 1.0
    for strategy in strategies.keys():
        if results[strategy]['total_times']:
            max_time = max(max_time, max(results[strategy]['total_times']))
    
    # Create time bins - ADJUSTABLE BIN SIZE
    time_step = bin_size  # Use the passed bin size parameter
    
    max_time_capped = min(max_time + time_step, 20.0)  # Cap at 20 for visualization
    time_bins = np.arange(time_step, max_time_capped + time_step, time_step)
    n_bins = len(time_bins)
    
    # Calculate bin positions for grouped bars with gaps between time bins
    n_strategies = len([s for s in strategies.keys() if results[s]['total_times']])
    
    # Create grouped bar positions
    # Each group (time bin) will have all strategies clustered together
    # with gaps between different time bins
    group_width = n_strategies * 0.2  # Total width of each group
    bar_width = 0.15  # Width of individual bars
    gap_between_groups = 0.3  # Gap between time bin groups
    
    # Calculate positions for each time bin group
    group_positions = []
    for i in range(n_bins):
        center_pos = i * (group_width + gap_between_groups)
        group_positions.append(center_pos)
    
    # Debug: Check which strategies are considered active
    active_strategy_names = [s for s in strategies.keys() if results[s]['total_times']]
    print(f"Active strategies for plotting: {active_strategy_names}")
    print(f"Number of strategies: {n_strategies}")
    print(f"Time bins: {time_bins[:5]}... (total: {n_bins} bins)")
    
    # Track which strategies have data for statistics
    active_strategies = []
    
    # Plot PDF bars for each strategy with TCP/UDP breakdown based on time
    for idx, (strategy_key, strategy_config) in enumerate(strategies.items()):
        print(f"Processing strategy {idx}: {strategy_key} ({strategy_config['name']})")
        
        transmissions = results[strategy_key]['transmissions']
        tcp_transmissions = results[strategy_key]['tcp_transmissions']
        udp_transmissions = results[strategy_key]['udp_transmissions']
        total_times = results[strategy_key]['total_times']
        
        print(f"  Results count: {len(total_times)}")
        
        if not total_times:
            print(f"  ❌ Skipping {strategy_key} - no results")
            continue
        
        print(f"  ✅ Processing {strategy_key} with {len(total_times)} results")
        print(f"  Time range: [{min(total_times):.2f}, {max(total_times):.2f}]")
        active_strategies.append((strategy_key, strategy_config))
        
        # Calculate PDF for each time bin with TCP/UDP breakdown
        total_pdf_values = np.zeros(n_bins)
        tcp_pdf_values = np.zeros(n_bins)
        udp_pdf_values = np.zeros(n_bins)
        
        for i, total_time in enumerate(total_times):
            # Find which time bin this result belongs to
            bin_idx = None
            for j, bin_edge in enumerate(time_bins):
                if total_time <= bin_edge:
                    bin_idx = j
                    break
            
            # If time exceeds all bins, put it in the last bin
            if bin_idx is None:
                bin_idx = n_bins - 1
            
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
        total_pdf_values = total_pdf_values / len(total_times)
        tcp_pdf_values = tcp_pdf_values / len(total_times)
        udp_pdf_values = udp_pdf_values / len(total_times)
        
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
        
        # Calculate statistics including time metrics
        mean_trans = np.mean(transmissions)
        std_trans = np.std(transmissions)
        mean_tcp = np.mean(tcp_transmissions)
        mean_udp = np.mean(udp_transmissions)
        tcp_percentage = mean_tcp / mean_trans * 100 if mean_trans > 0 else 0
        udp_percentage = mean_udp / mean_trans * 100 if mean_trans > 0 else 0
        success_rate = results[strategy_key]['success_count'] / len(transmissions)
        
        # Time statistics with TCP time factor
        total_times = results[strategy_key]['total_times']
        mean_time = np.mean(total_times)
        std_time = np.std(total_times)
        
        print(f"{strategy_config['name']} Statistics:")
        print(f"  Mean transmissions: {mean_trans:.2f} ± {std_trans:.2f}")
        print(f"  TCP transmissions: {mean_tcp:.2f} ({tcp_percentage:.1f}%)")
        print(f"  UDP transmissions: {mean_udp:.2f} ({udp_percentage:.1f}%)")
        print(f"  Mean total time: {mean_time:.2f} ± {std_time:.2f} (with TCP time factor {TCP_TIME_FACTOR})")
        print(f"  Range: [{min(transmissions)}, {max(transmissions)}]")
        print(f"  Success rate: {success_rate:.3f} ({results[strategy_key]['success_count']}/{len(transmissions)})")
        print()
    
    # Formatting
    plt.xlabel('Total Time Required (time units)', fontsize=20)
    plt.ylabel('Probability', fontsize=20)
    
    # Build title with run counts for each method
    run_counts = []
    for strategy_key, strategy_config in strategies.items():
        if results[strategy_key]['transmissions']:
            n_runs = len(results[strategy_key]['transmissions'])
            run_counts.append(str(n_runs))
    
    run_info = " | ".join(run_counts)
    
    plt.title(rf'PDF Comparison: Four Solution Methods (Improved)' + '\n' +
              rf'1 message and 10 vehicles, Parameters: $p_{{u}}={p_tcp}$, $p_{{b}}={p_udp}$, Time factor = {TCP_TIME_FACTOR}',
              fontsize=13, fontweight='bold', pad=20)    
    # Set x-axis with grouped positioning and time labels
    plt.xlim(-0.5, group_positions[-1] + 0.5)
    time_labels = [f'{time_bins[i]:.1f}' if i < len(time_bins) else f'{time_bins[-1]:.1f}+' 
                   for i in range(n_bins)]
    plt.xticks(group_positions[::5], time_labels[::5])
    #IMPORTANT
    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    
    # Create custom legend
    legend_elements = []
    for strategy_key, strategy_config in strategies.items():
        if results[strategy_key]['transmissions']:
            # TCP legend entry (solid)
            legend_elements.append(plt.Rectangle((0,0),1,1, 
                                               facecolor=strategy_config['color'], 
                                               alpha=strategy_config['alpha'],
                                               edgecolor='black',
                                               label=f'{strategy_config["name"]} - unicast'))
            # UDP legend entry (hatched)
            legend_elements.append(plt.Rectangle((0,0),1,1, 
                                               facecolor=strategy_config['color'], 
                                               alpha=strategy_config['alpha'] * 0.6,
                                               hatch='///',
                                               edgecolor='black',
                                               label=f'{strategy_config["name"]} - broadcast'))
    
    plt.legend(handles=legend_elements, fontsize=10, loc='upper right', ncol=2)
    
    # Add time statistics summary under the legend
    stats_text = "Time Statistics (Mean ± Std):\n"
    for strategy_key, strategy_config in strategies.items():
        if results[strategy_key]['total_times']:
            total_times = results[strategy_key]['total_times']
            mean_time = np.mean(total_times)
            std_time = np.std(total_times)
            stats_text += f"{strategy_config['name']}: {mean_time:.2f} ± {std_time:.2f}\n"
    
    # Add text box with statistics in the upper left corner
    plt.text(0.02, 0.98, stats_text.strip(), 
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
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
        
        filename = f'pdf_time_breakdown_m{n_messages}_v{n_vehicles}_p_u_{p_tcp:.3f}_p_b_{p_udp:.3f}_bins_{bin_size:.2f}.png'
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Plot saved to: {filepath}")
    
    plt.show()

def plot_cdf_comparison(results: Dict, n_messages: int, n_vehicles: int, p_tcp: float, p_udp: float, bin_size: float = 0.2, save_plot: bool = True):
    """Create a CDF plot comparing all four solution methods based on total time taken."""
    
    # Strategy configuration
    strategies = {
        'graph': {'name': 'Graph Shortest Path', 'color': '#2E8B57', 'alpha': 0.8},
        'mdp': {'name': 'MDP Policy', 'color': '#FF6B35', 'alpha': 0.8},
        'estimated': {'name': 'Estimated Heuristic', 'color': '#4169E1', 'alpha': 0.8},
        'expected_receivers': {'name': 'Expected Receivers', 'color': '#9932CC', 'alpha': 0.8}
    }
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Find the maximum time across all strategies
    max_time = 1.0
    for strategy in strategies.keys():
        if results[strategy]['total_times']:
            max_time = max(max_time, max(results[strategy]['total_times']))
    
    # Create time points for CDF (finer resolution for smooth curves)
    max_time_capped = min(max_time, 20.0)  # Cap at 20 for visualization
    time_points = np.linspace(0, max_time_capped, 1000)
    
    print(f"\nPlotting CDF for strategies with time range: [0, {max_time_capped:.2f}]")
    
    # Plot CDF for each strategy
    for strategy_key, strategy_config in strategies.items():
        total_times = results[strategy_key]['total_times']
        
        if not total_times:
            print(f"  ❌ Skipping {strategy_key} - no results")
            continue
        
        print(f"  ✅ Processing CDF for {strategy_key} with {len(total_times)} results")
        
        # Calculate CDF values
        cdf_values = []
        for time_point in time_points:
            # Count how many results are <= this time point
            count_below = sum(1 for t in total_times if t <= time_point)
            cdf_value = count_below / len(total_times)
            cdf_values.append(cdf_value)
        
        # Plot CDF curve
        plt.plot(time_points, cdf_values, 
                label=strategy_config['name'],
                color=strategy_config['color'],
                linewidth=2.5,
                alpha=strategy_config['alpha'])
        
        # Calculate and print statistics
        mean_time = np.mean(total_times)
        std_time = np.std(total_times)
        
        print(f"    Mean: {mean_time:.2f} ± {std_time:.2f}")
    
    # Formatting
    plt.xlabel('Total Time Required (time units)', fontsize=20)
    plt.ylabel('Cumulative Probability', fontsize=20)
    
    # Build title with run counts for each method
    run_counts = []
    for strategy_key, strategy_config in strategies.items():
        if results[strategy_key]['transmissions']:
            n_runs = len(results[strategy_key]['transmissions'])
            run_counts.append(str(n_runs))
    
    run_info = " | ".join(run_counts)
    
    plt.title(rf'CDF Comparison: Four Solution Methods' + '\n' +
              rf'1 message and 10 vehicles, Parameters: $p_{{u}}={p_tcp}$, $p_{{b}}={p_udp}$, Time factor = {TCP_TIME_FACTOR}',
              fontsize=13, fontweight='bold', pad=20)
  
    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    
    # Add horizontal reference lines for key percentiles
    for p in [0.25, 0.5, 0.75, 0.9]:
        plt.axhline(y=p, color='gray', linestyle='--', alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=11, loc='lower right')
    
    # Add time statistics summary (same format as PDF plot)
    stats_text = "Time Statistics (Mean ± Std):\n"
    for strategy_key, strategy_config in strategies.items():
        if results[strategy_key]['total_times']:
            total_times = results[strategy_key]['total_times']
            mean_time = np.mean(total_times)
            std_time = np.std(total_times)
            stats_text += f"{strategy_config['name']}: {mean_time:.2f} ± {std_time:.2f}\n"
    
    # Add text box with percentile statistics
    plt.text(0.02, 0.98, stats_text.strip(), 
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    plt.xlim(0, max_time_capped)
    plt.ylim(0, 1)
    
    # Make the plot look professional
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        # Create custom results directory
        results_dir = os.path.join(os.path.dirname(__file__), "custom_results")
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f'cdf_time_comparison_m{n_messages}_v{n_vehicles}_p_tcp_{p_tcp:.3f}_p_udp_{p_udp:.3f}_bins_{bin_size:.2f}.png'
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"CDF plot saved to: {filepath}")
    
    plt.show()

def main():
    """Main function to run custom parameter comparison."""
    
    # Get parameters from user
    n_messages, n_vehicles, p_tcp, p_udp = get_user_parameters()
    
    # Get bin size preference
    bin_size = get_bin_size()
    
    # Get initial state preference
    initial_state_type, initial_state_params = get_initial_state_option()
    
    # Ask for number of simulation runs
    print()
    while True:
        try:
            num_runs = int(input("Enter number of simulation runs (default 1000): ") or "1000")
            if num_runs > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid integer")
    
    print()
    print("Starting simulation...")
    
    # Generate a sample initial state for display
    sample_initial_state, initial_state_description = generate_initial_state(n_messages, n_vehicles, initial_state_type, initial_state_params)
    
    # Run simulation
    results = run_custom_simulation(n_messages, n_vehicles, p_tcp, p_udp, num_runs, initial_state_type, initial_state_params)
    
    # Create and display plots
    print("Creating time-based PDF comparison plot...")
    plot_pdf_comparison(results, n_messages, n_vehicles, p_tcp, p_udp, bin_size, save_plot=True)
    
    print("Creating time-based CDF comparison plot...")
    plot_cdf_comparison(results, n_messages, n_vehicles, p_tcp, p_udp, bin_size, save_plot=True)
    
    print("✅ Time-based parameter comparison completed successfully!")
    print()
    print("=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Initial State Configuration: {initial_state_description}")
    print(f"Sample Initial State Vector: {sample_initial_state}")
    print(f"State Encoding: 0 = already has message, 1 = needs message")
    print(f"Terminal State: All zeros (all vehicles have all messages)")
    
    # Show matrix format in summary for reasonable sizes
    if len(sample_initial_state) <= 45:  # Show for up to 3 messages × 15 vehicles
        print(f"\nInitial State Matrix (rows=messages, cols=vehicles):")
        state_matrix = []
        for msg in range(n_messages):
            start_idx = msg * n_vehicles
            end_idx = start_idx + n_vehicles
            msg_state = list(sample_initial_state[start_idx:end_idx])
            state_matrix.append(msg_state)
        
        print(f"     Vehicles: {' '.join(f'{i:2d}' for i in range(n_vehicles))}")
        for msg in range(n_messages):
            print(f"Message {msg}: {' '.join(f'{bit:2d}' for bit in state_matrix[msg])}")
    
    print()
    print("Key Insights:")
    print("PDF Plot:")
    print("- Compare the peak positions to see which method requires less total time")
    print("- TCP time factor (1.2x) creates non-integer time values (1.2, 2.4, 3.6, etc.)")
    print("- Strategies favoring TCP will show peaks at multiples of 1.2")
    print("- Strategies favoring UDP will show peaks at integer values (1.0, 2.0, 3.0, etc.)")
    print("- Compare the spread to see which method is more consistent in time performance")
    print()
    print("CDF Plot:")
    print("- Shows the probability of completing within a given time")
    print("- Curves to the left indicate faster methods (better performance)")
    print("- Steeper curves indicate more consistent performance")
    print("- Use percentile markers to compare reliability (e.g., 90% completion time)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you're running this script from the modified_simulation directory.")
