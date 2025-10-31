import numpy as np
import os
import csv
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

class CommunicationSimulator:
    def __init__(self, n_messages: int = 1, n_vehicles: int = 10):
        self.n_messages = n_messages
        self.n_vehicles = n_vehicles
        self.matrix_shape = (n_messages, n_vehicles)
    
    def f1(self, omega, p_tcp):
        """First function: omega/P_TCP"""
        if p_tcp == 0:
            return float('inf')
        return omega / p_tcp
    
    def f2(self, omega, p_udp):
        """Second function: 1 + (1/P_UDP - 1)^omega"""
        if p_udp == 0:
            return float('inf')
        if omega == 0:
            return 1
        return 1 + (1/p_udp - 1)**omega
    
    def calculate_omega(self, state_matrix: np.ndarray) -> int:
        """Calculate omega: number of messages still NOT received."""
        return int(np.sum(state_matrix))
    
    def matrix_to_index(self, matrix: np.ndarray) -> int:
        """Convert state matrix to state index."""
        return int("".join(map(str, matrix.flatten())), 2)
    
    def index_to_matrix(self, index: int) -> np.ndarray:
        """Convert state index to state matrix."""
        binary_string = bin(index)[2:].zfill(self.n_messages * self.n_vehicles)
        binary_array = np.array([int(bit) for bit in binary_string])
        return binary_array.reshape(self.matrix_shape)
    
    def get_initial_state(self, initial_pattern: str = "all_ones") -> np.ndarray:
        """Get initial state matrix. Flexible for different starting patterns."""
        if initial_pattern == "all_ones":
            return np.ones(self.matrix_shape, dtype=int)
        elif initial_pattern == "all_zeros":
            return np.zeros(self.matrix_shape, dtype=int)
        elif initial_pattern == "custom":
            # For future customization
            return np.ones(self.matrix_shape, dtype=int)
        else:
            return np.ones(self.matrix_shape, dtype=int)
    
    def is_terminal_state(self, state_matrix: np.ndarray) -> bool:
        """Check if we've reached the terminal state (all zeros)."""
        return np.sum(state_matrix) == 0
    
    def execute_tcp_action(self, state_matrix: np.ndarray, msg_id: int, vehicle_id: int, p_tcp: float) -> Tuple[np.ndarray, int]:
        """
        Execute TCP action on specific vehicle for specific message.
        Returns: (new_state, num_transmissions)
        """
        new_state = state_matrix.copy()
        num_transmissions = 0
        
        # Only attempt transmission if vehicle needs the message
        if state_matrix[msg_id, vehicle_id] == 1:
            num_transmissions = 1
            # TCP success probability
            if random.random() < p_tcp:
                new_state[msg_id, vehicle_id] = 0  # Vehicle received the message
        
        return new_state, num_transmissions
    
    def execute_udp_action(self, state_matrix: np.ndarray, msg_id: int, p_udp: float) -> Tuple[np.ndarray, int]:
        """
        Execute UDP broadcast action for specific message.
        Returns: (new_state, num_transmissions)
        """
        new_state = state_matrix.copy()
        num_transmissions = 1  # UDP broadcasts to all vehicles in one transmission
        
        # For each vehicle that needs the message
        for vehicle_id in range(self.n_vehicles):
            if state_matrix[msg_id, vehicle_id] == 1:
                # UDP success probability for each vehicle independently
                if random.random() < p_udp:
                    new_state[msg_id, vehicle_id] = 0  # Vehicle received the message
        
        return new_state, num_transmissions
    
    def parse_action(self, action_str: str) -> Tuple[str, int, Optional[int]]:
        """
        Parse action string to extract action type, message ID, and vehicle ID.
        Returns: (action_type, msg_id, vehicle_id)
        """
        if action_str.startswith("TCP_m_"):
            # Format: TCP_m_X_Y
            parts = action_str.split("_")
            action_type = "TCP"
            msg_id = int(parts[2])
            vehicle_id = int(parts[3])
            return action_type, msg_id, vehicle_id
        elif action_str.startswith("UDP_m_"):
            # Format: UDP_m_X
            parts = action_str.split("_")
            action_type = "UDP"
            msg_id = int(parts[2])
            return action_type, msg_id, None
        else:
            raise ValueError(f"Unknown action format: {action_str}")
    
    def execute_action(self, state_matrix: np.ndarray, action_str: str, p_tcp: float, p_udp: float) -> Tuple[np.ndarray, int]:
        """
        Execute an action and return new state and transmission cost.
        Returns: (new_state, num_transmissions)
        """
        if action_str in ["DESTINATION", "UNREACHABLE"]:
            return state_matrix, 0
        
        action_type, msg_id, vehicle_id = self.parse_action(action_str)
        
        if action_type == "TCP":
            return self.execute_tcp_action(state_matrix, msg_id, vehicle_id, p_tcp)
        elif action_type == "UDP":
            return self.execute_udp_action(state_matrix, msg_id, p_udp)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def load_strategy(self, strategy_type: str, p_tcp: float, p_udp: float, discount_factor: float = 0.9) -> Optional[Dict]:
        """Load strategy from Brain folder, generate if not exists."""
        if strategy_type.lower() == "graph":
            return self.load_graph_strategy(p_tcp, p_udp)
        elif strategy_type.lower() == "mdp":
            return self.load_mdp_strategy(p_tcp, p_udp, discount_factor)
        elif strategy_type.lower() == "estimated":
            return "ESTIMATED_STRATEGY"  # Special marker for estimated strategy
        elif strategy_type.lower() == "expected_receivers":
            return self.load_expected_receivers_strategy(p_tcp, p_udp)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    def load_graph_strategy(self, p_tcp: float, p_udp: float) -> Optional[Dict]:
        """Load Graph strategy from Brain/Graph folder."""
        from strategies.graph_solution import get_or_generate_solution
        
        print(f"Loading Graph strategy for p_tcp={p_tcp}, p_udp={p_udp}")
        strategy = get_or_generate_solution(self.n_messages, self.n_vehicles, p_tcp, p_udp)
        
        if strategy is None:
            print(f"Failed to load/generate Graph strategy")
            return None
        
        print(f"Loaded Graph strategy with {len(strategy)} states")
        return strategy
    
    def load_mdp_strategy(self, p_tcp: float, p_udp: float, discount_factor: float = 0.9) -> Optional[Dict]:
        """Load MDP strategy from Brain/MDP folder."""
        from strategies.mdp_solution import get_or_generate_mdp_solution
        
        print(f"Loading MDP strategy for p_tcp={p_tcp}, p_udp={p_udp}, discount={discount_factor}")
        strategy = get_or_generate_mdp_solution(self.n_messages, self.n_vehicles, p_tcp, p_udp, discount_factor)
        
        if strategy is None:
            print(f"Failed to load/generate MDP strategy (state space too large)")
            return None
        
        print(f"Loaded MDP strategy with {len(strategy)} states")
        
        # DEBUG: Check what action is recommended for initial state when p_udp = 0.1
        if p_udp == 0.1:
            initial_state_matrix = self.get_initial_state("all_ones")
            initial_state_id = self.matrix_to_index(initial_state_matrix)
            if initial_state_id in strategy:
                initial_action = strategy[initial_state_id]['action']
                print(f"DEBUG MDP p_udp=0.1: Initial state {initial_state_id} → action '{initial_action}'")
            else:
                print(f"DEBUG MDP p_udp=0.1: Initial state {initial_state_id} NOT FOUND in strategy!")
        
        return strategy
    
    def load_expected_receivers_strategy(self, p_tcp: float, p_udp: float) -> Optional[Dict]:
        """Load Expected Receivers strategy from Brain/ExpectedReceivers folder."""
        from strategies.expected_receivers_solution import load_expected_receivers_strategy
        
        print(f"Loading Expected Receivers strategy for p_tcp={p_tcp}, p_udp={p_udp}")
        strategy = load_expected_receivers_strategy(self.n_messages, self.n_vehicles, p_tcp, p_udp)
        
        if strategy is None:
            print(f"Failed to load Expected Receivers strategy")
            return None
        
        print(f"Loaded Expected Receivers strategy with {len(strategy)} states")
        return strategy
    
    def simulate_single_run(self, strategy: Dict, strategy_type: str, p_tcp: float, p_udp: float, 
                           initial_pattern: str = "all_ones", max_steps: int = 100) -> Tuple[int, int, bool, int, int]:
        """
        Simulate a single communication run using the given strategy.
        
        For stochastic systems, it's normal to stay in the same state multiple times
        due to transmission failures. This is counted as additional transmission attempts.
        
        Returns: (total_transmissions, steps_taken, success, tcp_transmissions, udp_transmissions)
        """
        # Initialize state
        current_state_matrix = self.get_initial_state(initial_pattern)
        current_state_id = self.matrix_to_index(current_state_matrix)
        
        total_transmissions = 0
        tcp_transmissions = 0
        udp_transmissions = 0
        steps_taken = 0
        
        while steps_taken < max_steps:
            # Check if terminal state reached
            if self.is_terminal_state(current_state_matrix):
                return total_transmissions, steps_taken, True, tcp_transmissions, udp_transmissions
            
            # Handle estimated strategy differently
            if strategy_type.lower() == "estimated":
                # Use estimated strategy decision making
                action_type, msg_id, vehicle_id = self.decide_action_estimated_strategy(
                    current_state_matrix, p_tcp, p_udp
                )
                
                if action_type == "TERMINAL":
                    return total_transmissions, steps_taken, True, tcp_transmissions, udp_transmissions
                
                # Execute estimated strategy action
                try:
                    if action_type == "TCP":
                        new_state_matrix, transmissions = self.execute_tcp_action(
                            current_state_matrix, msg_id, vehicle_id, p_tcp
                        )
                        tcp_transmissions += transmissions
                    elif action_type == "UDP":
                        new_state_matrix, transmissions = self.execute_udp_action(
                            current_state_matrix, msg_id, p_udp
                        )
                        udp_transmissions += transmissions
                    else:
                        print(f"Unknown action type: {action_type}")
                        return total_transmissions, steps_taken, False, tcp_transmissions, udp_transmissions
                    
                    total_transmissions += transmissions
                    steps_taken += 1
                    
                    # Update state
                    current_state_matrix = new_state_matrix
                    current_state_id = self.matrix_to_index(current_state_matrix)
                    
                except Exception as e:
                    print(f"Error executing estimated strategy action {action_type}: {e}")
                    return total_transmissions, steps_taken, False, tcp_transmissions, udp_transmissions
            
            elif strategy_type.lower() == "expected_receivers":
                # Use expected receivers strategy decision making
                action_type, msg_id, vehicle_id = self.decide_action_expected_receivers_strategy(
                    current_state_matrix, strategy, p_tcp, p_udp
                )
                
                if action_type == "TERMINAL":
                    return total_transmissions, steps_taken, True, tcp_transmissions, udp_transmissions
                
                # Execute expected receivers strategy action
                try:
                    if action_type == "TCP":
                        new_state_matrix, transmissions = self.execute_tcp_action(
                            current_state_matrix, msg_id, vehicle_id, p_tcp
                        )
                        tcp_transmissions += transmissions
                    elif action_type == "UDP":
                        new_state_matrix, transmissions = self.execute_udp_action(
                            current_state_matrix, msg_id, p_udp
                        )
                        udp_transmissions += transmissions
                    else:
                        print(f"Unknown action type: {action_type}")
                        return total_transmissions, steps_taken, False, tcp_transmissions, udp_transmissions
                    
                    total_transmissions += transmissions
                    steps_taken += 1
                    
                    # Update state
                    current_state_matrix = new_state_matrix
                    current_state_id = self.matrix_to_index(current_state_matrix)
                    
                except Exception as e:
                    print(f"Error executing expected receivers strategy action {action_type}: {e}")
                    return total_transmissions, steps_taken, False, tcp_transmissions, udp_transmissions
            
            else:
                # Strategy-based simulation (Graph/MDP)
                # Allow staying in same state multiple times - this is normal for stochastic systems
                
                # Get action from strategy
                if current_state_id not in strategy:
                    print(f"State {current_state_id} not found in {strategy_type} strategy")
                    return total_transmissions, steps_taken, False, tcp_transmissions, udp_transmissions
                
                if strategy_type.lower() == "graph":
                    action_str = strategy[current_state_id]['action']
                elif strategy_type.lower() == "mdp":
                    action_str = strategy[current_state_id]['action']
                else:
                    raise ValueError(f"Unknown strategy type: {strategy_type}")
                
                # ENHANCED DEBUG: Track MDP decisions with p_udp = 0.1
                if strategy_type.lower() == "mdp" and p_udp == 0.1 and random.random() < 0.01:
                    omega_current = self.calculate_omega(current_state_matrix)
                    print(f"MDP DEBUG: state_id={current_state_id}, omega={omega_current}, action={action_str}, "
                          f"step={steps_taken}, transmissions_so_far={total_transmissions}")
                
                # Handle special actions
                if action_str in ["DESTINATION", "UNREACHABLE"]:
                    return total_transmissions, steps_taken, action_str == "DESTINATION", tcp_transmissions, udp_transmissions
                
                # Execute action
                try:
                    new_state_matrix, transmissions = self.execute_action(
                        current_state_matrix, action_str, p_tcp, p_udp
                    )
                    
                    # Track TCP vs UDP transmissions based on action type
                    action_type, _, _ = self.parse_action(action_str)
                    if action_type == "TCP":
                        tcp_transmissions += transmissions
                    elif action_type == "UDP":
                        udp_transmissions += transmissions
                    
                    total_transmissions += transmissions
                    steps_taken += 1
                    
                    # Update state
                    current_state_matrix = new_state_matrix
                    current_state_id = self.matrix_to_index(current_state_matrix)
                    
                except Exception as e:
                    print(f"Error executing action {action_str}: {e}")
                    return total_transmissions, steps_taken, False, tcp_transmissions, udp_transmissions
        
        # Max steps reached - treat as successful completion with penalty
        return total_transmissions, steps_taken, True, tcp_transmissions, udp_transmissions
    
    def run_simulation_suite(self, p_tcp_values: List[float], p_udp_values: List[float], 
                           num_runs: int = 1000, initial_pattern: str = "all_ones") -> Dict:
        """Run comprehensive simulation for all parameter combinations."""
        results = {
            'parameters': {
                'n_messages': self.n_messages,
                'n_vehicles': self.n_vehicles,
                'num_runs': num_runs,
                'initial_pattern': initial_pattern
            },
            'simulations': []
        }
        
        total_combinations = len(p_tcp_values) * len(p_udp_values)
        current_combination = 0
        
        for p_tcp in p_tcp_values:
            for p_udp in p_udp_values:
                current_combination += 1
                print(f"\n=== Combination {current_combination}/{total_combinations}: p_tcp={p_tcp}, p_udp={p_udp} ===")
                
                # Load strategies
                print("Loading strategies...")
                graph_strategy = self.load_strategy("graph", p_tcp, p_udp)
                mdp_strategy = self.load_strategy("mdp", p_tcp, p_udp)
                estimated_strategy = self.load_strategy("estimated", p_tcp, p_udp)
                
                combination_results = {
                    'p_tcp': p_tcp,
                    'p_udp': p_udp,
                    'graph_results': {'transmissions': [], 'steps': [], 'success_count': 0, 'tcp_transmissions': [], 'udp_transmissions': []},
                    'mdp_results': {'transmissions': [], 'steps': [], 'success_count': 0, 'tcp_transmissions': [], 'udp_transmissions': []},
                    'estimated_results': {'transmissions': [], 'steps': [], 'success_count': 0, 'tcp_transmissions': [], 'udp_transmissions': []}
                }
                
                # Run Graph simulations
                if graph_strategy is not None:
                    print(f"Running {num_runs} Graph simulations...")
                    for run in range(num_runs):
                        if (run + 1) % 100 == 0:
                            print(f"  Graph run {run + 1}/{num_runs}")
                        
                        transmissions, steps, success, tcp_trans, udp_trans = self.simulate_single_run(
                            graph_strategy, "graph", p_tcp, p_udp, initial_pattern
                        )
                        
                        combination_results['graph_results']['transmissions'].append(transmissions)
                        combination_results['graph_results']['steps'].append(steps)
                        combination_results['graph_results']['tcp_transmissions'].append(tcp_trans)
                        combination_results['graph_results']['udp_transmissions'].append(udp_trans)
                        if success:
                            combination_results['graph_results']['success_count'] += 1
                else:
                    print("Graph strategy not available - skipping Graph simulations")
                
                # Run MDP simulations
                if mdp_strategy is not None:
                    print(f"Running {num_runs} MDP simulations...")
                    for run in range(num_runs):
                        if (run + 1) % 100 == 0:
                            print(f"  MDP run {run + 1}/{num_runs}")
                        
                        transmissions, steps, success, tcp_trans, udp_trans = self.simulate_single_run(
                            mdp_strategy, "mdp", p_tcp, p_udp, initial_pattern
                        )
                        
                        combination_results['mdp_results']['transmissions'].append(transmissions)
                        combination_results['mdp_results']['steps'].append(steps)
                        combination_results['mdp_results']['tcp_transmissions'].append(tcp_trans)
                        combination_results['mdp_results']['udp_transmissions'].append(udp_trans)
                        if success:
                            combination_results['mdp_results']['success_count'] += 1
                else:
                    print("MDP strategy not available - skipping MDP simulations")
                
                # Run Estimated simulations
                if estimated_strategy is not None:
                    print(f"Running {num_runs} Estimated simulations...")
                    for run in range(num_runs):
                        if (run + 1) % 100 == 0:
                            print(f"  Estimated run {run + 1}/{num_runs}")
                        
                        transmissions, steps, success, tcp_trans, udp_trans = self.simulate_single_run(
                            estimated_strategy, "estimated", p_tcp, p_udp, initial_pattern
                        )
                        
                        combination_results['estimated_results']['transmissions'].append(transmissions)
                        combination_results['estimated_results']['steps'].append(steps)
                        combination_results['estimated_results']['tcp_transmissions'].append(tcp_trans)
                        combination_results['estimated_results']['udp_transmissions'].append(udp_trans)
                        if success:
                            combination_results['estimated_results']['success_count'] += 1
                else:
                    print("Estimated strategy not available - skipping Estimated simulations")
                
                # Calculate statistics
                for strategy_type in ['graph_results', 'mdp_results', 'estimated_results']:
                    if combination_results[strategy_type]['transmissions']:
                        transmissions = combination_results[strategy_type]['transmissions']
                        tcp_transmissions = combination_results[strategy_type]['tcp_transmissions']
                        udp_transmissions = combination_results[strategy_type]['udp_transmissions']
                        
                        combination_results[strategy_type]['stats'] = {
                            'mean_transmissions': np.mean(transmissions),
                            'std_transmissions': np.std(transmissions),
                            'min_transmissions': np.min(transmissions),
                            'max_transmissions': np.max(transmissions),
                            'mean_tcp_transmissions': np.mean(tcp_transmissions),
                            'mean_udp_transmissions': np.mean(udp_transmissions),
                            'tcp_percentage': np.mean(tcp_transmissions) / np.mean(transmissions) * 100 if np.mean(transmissions) > 0 else 0,
                            'udp_percentage': np.mean(udp_transmissions) / np.mean(transmissions) * 100 if np.mean(transmissions) > 0 else 0,
                            'success_rate': combination_results[strategy_type]['success_count'] / num_runs
                        }
                    else:
                        combination_results[strategy_type]['stats'] = None
                
                results['simulations'].append(combination_results)
                
                # Print summary for this combination
                print(f"Results for p_tcp={p_tcp}, p_udp={p_udp}:")
                for strategy_type, strategy_name in [('graph_results', 'Graph'), ('mdp_results', 'MDP'), ('estimated_results', 'Estimated')]:
                    stats = combination_results[strategy_type]['stats']
                    success_count = combination_results[strategy_type]['success_count']
                    failed_count = num_runs - success_count
                    if stats:
                        completion_rate = stats['success_rate']
                        timeout_rate = 1 - completion_rate
                        print(f"  {strategy_name}: mean={stats['mean_transmissions']:.2f}±{stats['std_transmissions']:.2f}, "
                              f"range=[{stats['min_transmissions']:.0f}, {stats['max_transmissions']:.0f}], "
                              f"TCP={stats['mean_tcp_transmissions']:.1f}({stats['tcp_percentage']:.1f}%), "
                              f"UDP={stats['mean_udp_transmissions']:.1f}({stats['udp_percentage']:.1f}%), "
                              f"terminal_completion_rate={completion_rate:.3f} ({success_count}/{num_runs} reached terminal state, "
                              f"{failed_count} timed out at max_steps)")
                    else:
                        print(f"  {strategy_name}: No results available")
        
        return results
    
    def save_results_to_csv(self, results: Dict, filename: str = None) -> str:
        """Save simulation results to CSV file."""
        # Create Brain/Simulation directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        brain_dir = os.path.join(current_dir, "Brain")
        simulation_dir = os.path.join(brain_dir, "Simulation")
        os.makedirs(simulation_dir, exist_ok=True)
        
        if filename is None:
            filename = f"communication_simulation_m{self.n_messages}_v{self.n_vehicles}.csv"
        
        filepath = os.path.join(simulation_dir, filename)
        
        # Prepare CSV data
        csv_data = []
        for sim in results['simulations']:
            p_tcp = sim['p_tcp']
            p_udp = sim['p_udp']
            
            # Add rows for each strategy
            for strategy_type, strategy_name in [('graph_results', 'Graph'), ('mdp_results', 'MDP'), ('estimated_results', 'Estimated')]:
                stats = sim[strategy_type]['stats']
                if stats:
                    csv_data.append({
                        'strategy': strategy_name,
                        'p_tcp': p_tcp,
                        'p_udp': p_udp,
                        'mean_transmissions': stats['mean_transmissions'],
                        'std_transmissions': stats['std_transmissions'],
                        'min_transmissions': stats['min_transmissions'],
                        'max_transmissions': stats['max_transmissions'],
                        'mean_tcp_transmissions': stats['mean_tcp_transmissions'],
                        'mean_udp_transmissions': stats['mean_udp_transmissions'],
                        'tcp_percentage': stats['tcp_percentage'],
                        'udp_percentage': stats['udp_percentage'],
                        'success_rate': stats['success_rate'],
                        'num_runs': results['parameters']['num_runs']
                    })
        
        # Save to CSV
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['strategy', 'p_tcp', 'p_udp', 'mean_transmissions', 'std_transmissions',
                         'min_transmissions', 'max_transmissions', 'mean_tcp_transmissions', 
                         'mean_udp_transmissions', 'tcp_percentage', 'udp_percentage', 
                         'success_rate', 'num_runs']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def plot_results(self, results: Dict, save_plots: bool = True):
        """Plot simulation results as histograms for each (p_tcp, p_udp) pair."""
        import matplotlib.pyplot as plt
        
        # Extract data for plotting
        p_tcp_values = sorted(set(sim['p_tcp'] for sim in results['simulations']))
        p_udp_values = sorted(set(sim['p_udp'] for sim in results['simulations']))
        
        # Create plots directory
        if save_plots:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            brain_dir = os.path.join(current_dir, "Brain")
            plots_dir = os.path.join(brain_dir, "Plots")
            os.makedirs(plots_dir, exist_ok=True)
        
        # Calculate grid dimensions for subplots
        n_tcp = len(p_tcp_values)
        n_udp = len(p_udp_values)
        
        # Create a large figure with subplots for each (p_tcp, p_udp) combination
        fig_width = max(20, 4 * n_tcp)
        fig_height = max(12, 3 * n_udp)
        fig, axes = plt.subplots(n_udp, n_tcp, figsize=(fig_width, fig_height))
        
        # Ensure axes is always 2D array
        if n_udp == 1 and n_tcp == 1:
            axes = np.array([[axes]])
        elif n_udp == 1:
            axes = axes.reshape(1, -1)
        elif n_tcp == 1:
            axes = axes.reshape(-1, 1)
        
        print(f"\nCreating histogram plots for {n_tcp} × {n_udp} = {n_tcp * n_udp} parameter combinations...")
        
        # Create bins for consistent x-axis (1 to 15+) - shifted by 0.5
        bins = np.arange(0, 16, 1)  # Bin edges: 0, 1, 2, ..., 15
        bin_labels = [str(i) for i in range(1, 15)] + ['15+']
        
        # Store all histogram objects for interactive legend
        all_hist_objects = {'graph': [], 'mdp': [], 'estimated': []}
        legend_labels = {'graph': 'Shortest Path', 'mdp': 'MDP Policy', 'estimated': 'Estimated'}
        legend_colors = {'graph': 'blue', 'mdp': 'red', 'estimated': 'green'}
        
        # Plot histogram for each combination
        for i, p_udp in enumerate(p_udp_values):
            for j, p_tcp in enumerate(p_tcp_values):
                ax = axes[i, j]
                
                # Find corresponding simulation results
                sim_data = None
                for sim in results['simulations']:
                    if sim['p_tcp'] == p_tcp and sim['p_udp'] == p_udp:
                        sim_data = sim
                        break
                
                if sim_data:
                    # Get transmission data for all strategies
                    strategies_data = {}
                    for strat in ['graph', 'mdp', 'estimated']:
                        transmissions = sim_data[f'{strat}_results']['transmissions']
                        tcp_transmissions = sim_data[f'{strat}_results']['tcp_transmissions']
                        udp_transmissions = sim_data[f'{strat}_results']['udp_transmissions']
                        if transmissions:
                            strategies_data[strat] = {
                                'total': [min(x, 15) - 1 for x in transmissions],  # Shift by -1 for binning
                                'tcp': tcp_transmissions,
                                'udp': udp_transmissions
                            }
                    
                    # Plot grouped histograms with TCP/UDP breakdown
                    num_runs = results['parameters']['num_runs']
                    bar_width = 0.25  # Width for each strategy's bars
                    n_strategies = len(strategies_data)
                    
                    for idx, (strat, data) in enumerate(strategies_data.items()):
                        if not data['total']:
                            continue
                        
                        # Calculate total transmission counts for each bin
                        total_counts = np.zeros(15)  # 15 bins (0-14 representing 1-15 transmissions)
                        tcp_counts = np.zeros(15)
                        udp_counts = np.zeros(15)
                        
                        # Count occurrences in each bin
                        for k, total_trans in enumerate(data['total']):
                            if 0 <= total_trans < 15:
                                total_counts[total_trans] += 1
                                tcp_counts[total_trans] += data['tcp'][k]
                                udp_counts[total_trans] += data['udp'][k]
                        
                        # Calculate probabilities - each strategy's bars should sum to <= 1
                        total_probs = total_counts / num_runs
                        
                        # Calculate TCP/UDP proportions within each bin
                        tcp_props = np.zeros(15)
                        udp_props = np.zeros(15)
                        for bin_idx in range(15):
                            if total_counts[bin_idx] > 0:
                                total_transmissions_in_bin = tcp_counts[bin_idx] + udp_counts[bin_idx]
                                if total_transmissions_in_bin > 0:
                                    tcp_props[bin_idx] = tcp_counts[bin_idx] / total_transmissions_in_bin
                                    udp_props[bin_idx] = udp_counts[bin_idx] / total_transmissions_in_bin
                        
                        # Calculate actual TCP and UDP probabilities for plotting
                        tcp_probs = total_probs * tcp_props
                        udp_probs = total_probs * udp_props
                        
                        # Position bars for this strategy (grouped bars)
                        x_pos = np.arange(15) + (idx - (n_strategies-1)/2) * bar_width
                        
                        # Create stacked bars - TCP on bottom, UDP on top
                        tcp_bars = ax.bar(x_pos, tcp_probs, bar_width, 
                                        color=legend_colors[strat], alpha=0.8, 
                                        edgecolor='black', linewidth=0.5)
                        
                        udp_bars = ax.bar(x_pos, udp_probs, bar_width, bottom=tcp_probs,
                                        color=legend_colors[strat], alpha=0.4, 
                                        edgecolor='black', linewidth=0.5,
                                        hatch='///')  # Add hatching to distinguish UDP
                        
                        all_hist_objects[strat].extend(tcp_bars + udp_bars)
                    
                    # Set labels and title with LaTeX formatting
                    ax.set_xlabel('#transmissions', fontsize=10)
                    ax.set_ylabel('probability', fontsize=10)
                    ax.set_title(rf'$p_{{tcp}}={p_tcp:.2f}, p_{{udp}}={p_udp:.1f}$', fontsize=11, fontweight='bold')
                    
                    # Set x-axis range and ticks for grouped bars
                    ax.set_xlim(-0.5, 15)
                    ax.set_ylim(0, 1)  # Set y-axis limit to 1
                    ax.set_xticks(np.arange(0, 15))  # Center ticks on transmission counts
                    ax.set_xticklabels(bin_labels)
                    
                    # Add grid
                    ax.grid(True, alpha=0.3)
                    
                    # Create custom legend
                    legend_elements = []
                    for strat in ['graph', 'mdp', 'estimated']:
                        if strat in strategies_data:
                            # TCP legend entry
                            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=legend_colors[strat], 
                                                               alpha=0.8, label=f'{legend_labels[strat]} TCP'))
                            # UDP legend entry
                            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=legend_colors[strat], 
                                                               alpha=0.4, hatch='///', label=f'{legend_labels[strat]} UDP'))
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
                
                else:
                    # No data for this combination
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=12, fontweight='bold')
                    ax.set_title(rf'$p_{{tcp}}={p_tcp:.2f}, p_{{udp}}={p_udp:.1f}$', fontsize=11)
        
        # Save plots for individual strategies and combinations
        self._save_histogram_plots_separately(results, p_tcp_values, p_udp_values, plots_dir)
        self._save_cdf_plots_separately(results, p_tcp_values, p_udp_values, plots_dir)
        
        # Add simple legend to the main plot
        legend_handles = []
        legend_texts = []
        for strategy in ['graph', 'mdp', 'estimated']:
            # Check if any data exists for this strategy
            has_data = any(sim[f'{strategy}_results']['stats'] is not None 
                          for sim in results['simulations'])
            if has_data:
                handle = plt.Line2D([0], [0], color=legend_colors[strategy], linewidth=2, 
                                   label=legend_labels[strategy])
                legend_handles.append(handle)
                legend_texts.append(legend_labels[strategy])
        
        if legend_handles:
            fig.legend(legend_handles, legend_texts, 
                      loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                      ncol=len(legend_handles), fontsize=12)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.suptitle(rf'Transmission Cost Distributions ($m={self.n_messages}$, $v={self.n_vehicles}$)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        if save_plots:
            filename = f'transmission_histograms_m{self.n_messages}_v{self.n_vehicles}.png'
            filepath = os.path.join(plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Histogram plot saved to: {filepath}")
        
        plt.show()
        
        # Also create CDF plots, summary statistics plot, and TCP/UDP breakdown plot
        self._plot_cdf_results(results, save_plots, plots_dir)
        self._plot_tcp_udp_cdf_breakdown(results, save_plots, plots_dir)  # NEW - TCP/UDP breakdown CDF
        self._plot_summary_statistics(results, save_plots, plots_dir)
        # self._plot_tcp_udp_breakdown(results, save_plots, plots_dir)  # DISABLED - don't save TCP/UDP breakdown plots
    
    def _plot_cdf_results(self, results, save_plots=True, plots_dir=None):
        """Plot CDF results for each parameter combination comparing both strategies"""
        # Extract data for plotting
        p_tcp_values = sorted(set(sim['p_tcp'] for sim in results['simulations']))
        p_udp_values = sorted(set(sim['p_udp'] for sim in results['simulations']))
        
        # Calculate grid dimensions for subplots
        n_tcp = len(p_tcp_values)
        n_udp = len(p_udp_values)
        
        # Create a large figure with subplots for each (p_tcp, p_udp) combination
        fig_width = max(20, 4 * n_tcp)
        fig_height = max(12, 3 * n_udp)
        fig, axes = plt.subplots(n_udp, n_tcp, figsize=(fig_width, fig_height))
        
        # Ensure axes is always 2D array
        if n_udp == 1 and n_tcp == 1:
            axes = np.array([[axes]])
        elif n_udp == 1:
            axes = axes.reshape(1, -1)
        elif n_tcp == 1:
            axes = axes.reshape(-1, 1)
        
        print(f"\nCreating CDF plots for {n_tcp} × {n_udp} = {n_tcp * n_udp} parameter combinations...")
        
        # Store all line objects for interactive legend
        all_line_objects = {'graph': [], 'mdp': [], 'estimated': []}
        legend_labels = {'graph': 'Shortest Path', 'mdp': 'MDP Policy', 'estimated': 'Estimated'}
        legend_colors = {'graph': 'blue', 'mdp': 'red', 'estimated': 'green'}
        
        # Plot CDF for each combination
        for i, p_udp in enumerate(p_udp_values):
            for j, p_tcp in enumerate(p_tcp_values):
                ax = axes[i, j]
                
                # Find corresponding simulation results
                sim_data = None
                for sim in results['simulations']:
                    if sim['p_tcp'] == p_tcp and sim['p_udp'] == p_udp:
                        sim_data = sim
                        break
                
                if sim_data:
                    graph_transmissions = sim_data['graph_results']['transmissions']
                    mdp_transmissions = sim_data['mdp_results']['transmissions']
                    estimated_transmissions = sim_data['estimated_results']['transmissions']
                    
                    # Cap data at 15 for plotting
                    if graph_transmissions:
                        graph_data_capped = [min(cost, 15) for cost in graph_transmissions]
                        # Create CDF for Graph strategy
                        sorted_graph = np.sort(graph_data_capped)
                        graph_y = np.arange(1, len(sorted_graph) + 1) / len(sorted_graph)
                        line = ax.step(sorted_graph, graph_y, where='post', color=legend_colors['graph'], 
                               linewidth=2, label=legend_labels['graph'])
                        all_line_objects['graph'].extend(line)
                    
                    if mdp_transmissions:
                        mdp_data_capped = [min(cost, 15) for cost in mdp_transmissions]
                        # Create CDF for MDP strategy  
                        sorted_mdp = np.sort(mdp_data_capped)
                        mdp_y = np.arange(1, len(sorted_mdp) + 1) / len(sorted_mdp)
                        line = ax.step(sorted_mdp, mdp_y, where='post', color=legend_colors['mdp'], 
                               linewidth=2, label=legend_labels['mdp'])
                        all_line_objects['mdp'].extend(line)
                    
                    if estimated_transmissions:
                        estimated_data_capped = [min(cost, 15) for cost in estimated_transmissions]
                        # Create CDF for Estimated strategy  
                        sorted_estimated = np.sort(estimated_data_capped)
                        estimated_y = np.arange(1, len(sorted_estimated) + 1) / len(sorted_estimated)
                        line = ax.step(sorted_estimated, estimated_y, where='post', color=legend_colors['estimated'], 
                               linewidth=2, label=legend_labels['estimated'])
                        all_line_objects['estimated'].extend(line)
                    
                    # Set axis properties
                    ax.set_xlim(0.5, 15.5)
                    ax.set_ylim(0, 1)
                    ax.set_xticks(range(1, 16))
                    ax.set_xticklabels([str(i) if i < 15 else '15+' for i in range(1, 16)])
                    ax.grid(True, alpha=0.3)
                    
                    # Set title and labels
                    ax.set_title(rf'$p_{{tcp}}={p_tcp:.2f}, p_{{udp}}={p_udp:.1f}$', 
                               fontsize=11, fontweight='bold')
                    ax.set_xlabel('#transmissions', fontsize=10)
                    ax.set_ylabel('cumulative probability', fontsize=10)
                    
                    # Add legend to each subplot
                    ax.legend(loc='upper right', fontsize=8)
                
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(rf'$p_{{tcp}}={p_tcp:.2f}, p_{{udp}}={p_udp:.1f}$', 
                               fontsize=11, fontweight='bold')
        
        # Save plots for individual strategies and combinations
        # self._save_cdf_plots_separately(results, p_tcp_values, p_udp_values)  # DISABLED - don't save many files
        
        # Add simple legend to the main plot
        legend_handles = []
        legend_texts = []
        for strategy in ['graph', 'mdp', 'estimated']:
            # Check if any data exists for this strategy
            has_data = any(sim[f'{strategy}_results']['stats'] is not None 
                          for sim in results['simulations'])
            if has_data:
                handle = plt.Line2D([0], [0], color=legend_colors[strategy], linewidth=2, 
                                   label=legend_labels[strategy])
                legend_handles.append(handle)
                legend_texts.append(legend_labels[strategy])
        
        if legend_handles:
            fig.legend(legend_handles, legend_texts, 
                      loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                      ncol=len(legend_handles), fontsize=12)
        
        plt.suptitle('Cumulative Distribution Functions (CDFs) of Transmission Attempts', 
                    fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_plots and plots_dir:
            cdf_path = os.path.join(plots_dir, 'transmission_attempts_cdf.png')
            plt.savefig(cdf_path, dpi=300, bbox_inches='tight')
            print(f"CDF plot saved to: {cdf_path}")
        
        plt.show()
    
    def _plot_tcp_udp_cdf_breakdown(self, results, save_plots=True, plots_dir=None):
        """Plot CDF results with TCP/UDP breakdown showing total CDF with TCP/UDP boundary"""
        # Extract data for plotting
        p_tcp_values = sorted(set(sim['p_tcp'] for sim in results['simulations']))
        p_udp_values = sorted(set(sim['p_udp'] for sim in results['simulations']))
        
        # Calculate grid dimensions for subplots
        n_tcp = len(p_tcp_values)
        n_udp = len(p_udp_values)
        
        # Create a large figure with subplots for each (p_tcp, p_udp) combination
        fig_width = max(20, 4 * n_tcp)
        fig_height = max(12, 3 * n_udp)
        fig, axes = plt.subplots(n_udp, n_tcp, figsize=(fig_width, fig_height))
        
        # Ensure axes is always 2D array
        if n_udp == 1 and n_tcp == 1:
            axes = np.array([[axes]])
        elif n_udp == 1:
            axes = axes.reshape(1, -1)
        elif n_tcp == 1:
            axes = axes.reshape(-1, 1)
        
        print(f"\nCreating TCP/UDP breakdown CDF plots for {n_tcp} × {n_udp} = {n_tcp * n_udp} parameter combinations...")
        
        # Store all line objects for interactive legend
        all_line_objects = {'graph': [], 'mdp': [], 'estimated': []}
        legend_labels = {'graph': 'Shortest Path', 'mdp': 'MDP Policy', 'estimated': 'Estimated'}
        legend_colors = {'graph': 'blue', 'mdp': 'red', 'estimated': 'green'}
        
        # Plot CDF for each combination with TCP/UDP breakdown
        for i, p_udp in enumerate(p_udp_values):
            for j, p_tcp in enumerate(p_tcp_values):
                ax = axes[i, j]
                
                # Find corresponding simulation results
                sim_data = None
                for sim in results['simulations']:
                    if sim['p_tcp'] == p_tcp and sim['p_udp'] == p_udp:
                        sim_data = sim
                        break
                
                if sim_data:
                    # Plot total CDF with TCP/UDP breakdown for each strategy
                    for strategy in ['graph', 'mdp', 'estimated']:
                        transmissions = sim_data[f'{strategy}_results']['transmissions']
                        tcp_transmissions = sim_data[f'{strategy}_results']['tcp_transmissions'] 
                        udp_transmissions = sim_data[f'{strategy}_results']['udp_transmissions']
                        
                        if transmissions:
                            # Cap data at 15 for plotting
                            total_data_capped = [min(cost, 15) for cost in transmissions]
                            
                            # Create total CDF (solid line)
                            sorted_total = np.sort(total_data_capped)
                            total_y = np.arange(1, len(sorted_total) + 1) / len(sorted_total)
                            line = ax.step(sorted_total, total_y, where='post', color=legend_colors[strategy], 
                                   linewidth=2, label=f'{legend_labels[strategy]} Total')
                            all_line_objects[strategy].extend(line)
                            
                            # Create TCP portion CDF (filled area below the boundary)
                            tcp_ratios = []
                            for k, total_trans in enumerate(transmissions):
                                if total_trans > 0:
                                    tcp_ratio = tcp_transmissions[k] / total_trans
                                else:
                                    tcp_ratio = 0
                                tcp_ratios.append(tcp_ratio)
                            
                            # Calculate TCP boundary line
                            tcp_boundary_y = []
                            for k, y_val in enumerate(total_y):
                                # For each point in the CDF, calculate the average TCP ratio up to that point
                                indices_up_to_here = [idx for idx, val in enumerate(total_data_capped) if val <= sorted_total[k]]
                                if indices_up_to_here:
                                    avg_tcp_ratio = np.mean([tcp_ratios[idx] for idx in indices_up_to_here])
                                    tcp_boundary_y.append(y_val * avg_tcp_ratio)
                                else:
                                    tcp_boundary_y.append(0)
                            

                            # Draw TCP boundary line (dashed) - shows where TCP portion ends
                            ax.step(sorted_total, tcp_boundary_y, where='post', color=legend_colors[strategy], 
                                   linewidth=1.5, linestyle='--', alpha=0.8, 
                                   label=f'{legend_labels[strategy]} TCP')
                    
                    # Set axis properties
                    ax.set_xlim(0.5, 15.5)
                    ax.set_ylim(0, 1)
                    ax.set_xticks(range(1, 16))
                    ax.set_xticklabels([str(i) if i < 15 else '15+' for i in range(1, 16)])
                    ax.grid(True, alpha=0.3)
                    
                    # Set title and labels
                    ax.set_title(rf'$p_{{tcp}}={p_tcp:.2f}, p_{{udp}}={p_udp:.1f}$', 
                               fontsize=11, fontweight='bold')
                    ax.set_xlabel('#transmissions', fontsize=10)
                    ax.set_ylabel('cumulative probability', fontsize=10)
                    
                    # Add custom legend
                    legend_elements = []
                    for strat in ['graph', 'mdp', 'estimated']:
                        # Check if this strategy has data
                        strategy_transmissions = sim_data[f'{strat}_results']['transmissions']
                        if strategy_transmissions:
                            # Total CDF legend entry (solid line)
                            legend_elements.append(plt.Line2D([0], [0], color=legend_colors[strat], 
                                                            linewidth=2, 
                                                            label=f'{legend_labels[strat]} Total'))
                            # TCP boundary legend entry (dashed line)
                            legend_elements.append(plt.Line2D([0], [0], color=legend_colors[strat], 
                                                            linewidth=1.5, linestyle='--', alpha=0.8,
                                                            label=f'{legend_labels[strat]} TCP'))
                    
                    if legend_elements:
                        ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
                
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(rf'$p_{{tcp}}={p_tcp:.2f}, p_{{udp}}={p_udp:.1f}$', 
                               fontsize=11, fontweight='bold')
        
        # Add simple legend to the main plot
        legend_handles = []
        legend_texts = []
        for strategy in ['graph', 'mdp', 'estimated']:
            # Check if any data exists for this strategy
            has_data = any(sim[f'{strategy}_results']['stats'] is not None 
                          for sim in results['simulations'])
            if has_data:
                handle = plt.Line2D([0], [0], color=legend_colors[strategy], linewidth=2, 
                                   label=legend_labels[strategy])
                legend_handles.append(handle)
                legend_texts.append(legend_labels[strategy])
        
        if legend_handles:
            fig.legend(legend_handles, legend_texts, 
                      loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                      ncol=len(legend_handles), fontsize=12)
        
        plt.suptitle('TCP/UDP Breakdown - Cumulative Distribution Functions (CDFs)', 
                    fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_plots and plots_dir:
            cdf_path = os.path.join(plots_dir, 'tcp_udp_breakdown_cdf.png')
            plt.savefig(cdf_path, dpi=300, bbox_inches='tight')
            print(f"TCP/UDP breakdown CDF plot saved to: {cdf_path}")
        
        plt.show()
    
    def find_first_vehicle_needing_message(self, state_matrix: np.ndarray, msg_id: int) -> Optional[int]:
        """Find the first vehicle that needs a specific message."""
        for vehicle_id in range(self.n_vehicles):
            if state_matrix[msg_id, vehicle_id] == 1:
                return vehicle_id
        return None
    
    def find_first_message_needed(self, state_matrix: np.ndarray) -> Optional[int]:
        """Find the first message that is still needed by at least one vehicle."""
        for msg_id in range(self.n_messages):
            if np.sum(state_matrix[msg_id, :]) > 0:
                return msg_id
        return None
    
    def decide_action_estimated_strategy(self, state_matrix: np.ndarray, p_tcp: float, p_udp: float) -> Tuple[str, int, Optional[int]]:
        """
        Decide action using the estimated strategy (f1/f2 comparison).
        Returns: (action_type, msg_id, vehicle_id)
        """
        # Check if we're in terminal state
        if self.is_terminal_state(state_matrix):
            return "TERMINAL", 0, None
        
        # Find the first message that needs to be sent
        msg_id = self.find_first_message_needed(state_matrix)
        if msg_id is None:
            return "TERMINAL", 0, None
        
        # Calculate omega for this message
        omega = int(np.sum(state_matrix[msg_id, :]))
        
        # Calculate f1 and f2
        f1_value = self.f1(omega, p_tcp)
        f2_value = self.f2(omega, p_udp)
        
        # DEBUG: Print values for analysis (only occasionally to avoid spam)
        if random.random() < 0.01:  # Print ~1% of the time
            print(f"ESTIMATED DEBUG: omega={omega}, p_tcp={p_tcp}, p_udp={p_udp}")
            print(f"  f1={f1_value:.6f}, f2={f2_value:.6f}")
            print(f"  Choosing: {'TCP' if f1_value <= f2_value else 'UDP'}")
        
        # Choose action based on f1 vs f2 comparison
        if f1_value <= f2_value:
            # TCP is better - find first vehicle needing this message
            vehicle_id = self.find_first_vehicle_needing_message(state_matrix, msg_id)
            return "TCP", msg_id, vehicle_id
        else:
            # UDP is better
            return "UDP", msg_id, None
    
    def decide_action_expected_receivers_strategy(self, state_matrix: np.ndarray, strategy: Dict, p_tcp: float, p_udp: float) -> Tuple[str, int, Optional[int]]:
        """
        Decide action using the Expected Receivers strategy.
        Returns: (action_type, msg_id, vehicle_id)
        """
        # Check if we're in terminal state
        if self.is_terminal_state(state_matrix):
            return "TERMINAL", 0, None
        
        # Get current state ID
        current_state_id = self.matrix_to_index(state_matrix)
        
        # Check if strategy has decision for this state
        if current_state_id not in strategy:
            print(f"State {current_state_id} not found in Expected Receivers strategy")
            return "TERMINAL", 0, None
        
        # Get the action from the strategy (now in standard format)
        state_decision = strategy[current_state_id]
        action_str = state_decision['action']
        
        # Parse the action string
        if action_str.startswith("TCP_m_"):
            parts = action_str.split("_")
            msg_id = int(parts[2])
            # Find first vehicle that needs this message
            vehicle_id = self.find_first_vehicle_needing_message(state_matrix, msg_id)
            if vehicle_id is not None:
                return "TCP", msg_id, vehicle_id
        elif action_str.startswith("UDP_m_"):
            parts = action_str.split("_")
            msg_id = int(parts[2])
            # Check if this message still needs to be sent
            if np.sum(state_matrix[msg_id, :]) > 0:
                return "UDP", msg_id, None
        
        # If no valid action found, we're done
        return "TERMINAL", 0, None

    def _plot_summary_statistics(self, results: Dict, save_plots: bool, plots_dir: str):
        """Create a summary plot showing mean transmissions for all combinations."""
        
        # Extract data for plotting
        p_tcp_values = sorted(set(sim['p_tcp'] for sim in results['simulations']))
        p_udp_values = sorted(set(sim['p_udp'] for sim in results['simulations']))
        
        # Create summary plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
        
        # Plot 1: Heatmap for Graph strategy
        graph_data = np.full((len(p_udp_values), len(p_tcp_values)), np.nan)
        for i, p_udp in enumerate(p_udp_values):
            for j, p_tcp in enumerate(p_tcp_values):
                for sim in results['simulations']:
                    if sim['p_tcp'] == p_tcp and sim['p_udp'] == p_udp:
                        if sim['graph_results']['stats']:
                            graph_data[i, j] = sim['graph_results']['stats']['mean_transmissions']
                        break
        
        im1 = ax1.imshow(graph_data, cmap='Blues', aspect='auto')
        ax1.set_xticks(range(len(p_tcp_values)))
        ax1.set_yticks(range(len(p_udp_values)))
        ax1.set_xticklabels([f'{p:.2f}' for p in p_tcp_values])
        ax1.set_yticklabels([f'{p:.1f}' for p in p_udp_values])
        ax1.set_xlabel(r'$p_{tcp}$')
        ax1.set_ylabel(r'$p_{udp}$')
        ax1.set_title('Shortest Path Strategy - Mean Transmissions')
        plt.colorbar(im1, ax=ax1)
        
        # Add text annotations
        for i in range(len(p_udp_values)):
            for j in range(len(p_tcp_values)):
                if not np.isnan(graph_data[i, j]):
                    ax1.text(j, i, f'{graph_data[i, j]:.1f}', ha='center', va='center', 
                            color='white' if graph_data[i, j] > np.nanmean(graph_data) else 'black')
        
        # Plot 2: Heatmap for MDP strategy
        mdp_data = np.full((len(p_udp_values), len(p_tcp_values)), np.nan)
        for i, p_udp in enumerate(p_udp_values):
            for j, p_tcp in enumerate(p_tcp_values):
                for sim in results['simulations']:
                    if sim['p_tcp'] == p_tcp and sim['p_udp'] == p_udp:
                        if sim['mdp_results']['stats']:
                            mdp_data[i, j] = sim['mdp_results']['stats']['mean_transmissions']
                        break
        
        im2 = ax2.imshow(mdp_data, cmap='Reds', aspect='auto')
        ax2.set_xticks(range(len(p_tcp_values)))
        ax2.set_yticks(range(len(p_udp_values)))
        ax2.set_xticklabels([f'{p:.2f}' for p in p_tcp_values])
        ax2.set_yticklabels([f'{p:.1f}' for p in p_udp_values])
        ax2.set_xlabel(r'$p_{tcp}$')
        ax2.set_ylabel(r'$p_{udp}$')
        ax2.set_title('MDP Strategy - Mean Transmissions')
        plt.colorbar(im2, ax=ax2)
        
        # Add text annotations
        for i in range(len(p_udp_values)):
            for j in range(len(p_tcp_values)):
                if not np.isnan(mdp_data[i, j]):
                    ax2.text(j, i, f'{mdp_data[i, j]:.1f}', ha='center', va='center', 
                            color='white' if mdp_data[i, j] > np.nanmean(mdp_data) else 'black')
        
        # Plot 3: Heatmap for Estimated strategy
        estimated_data = np.full((len(p_udp_values), len(p_tcp_values)), np.nan)
        for i, p_udp in enumerate(p_udp_values):
            for j, p_tcp in enumerate(p_tcp_values):
                for sim in results['simulations']:
                    if sim['p_tcp'] == p_tcp and sim['p_udp'] == p_udp:
                        if sim['estimated_results']['stats']:
                            estimated_data[i, j] = sim['estimated_results']['stats']['mean_transmissions']
                        break
        
        im3 = ax3.imshow(estimated_data, cmap='Greens', aspect='auto')
        ax3.set_xticks(range(len(p_tcp_values)))
        ax3.set_yticks(range(len(p_udp_values)))
        ax3.set_xticklabels([f'{p:.2f}' for p in p_tcp_values])
        ax3.set_yticklabels([f'{p:.1f}' for p in p_udp_values])
        ax3.set_xlabel(r'$p_{tcp}$')
        ax3.set_ylabel(r'$p_{udp}$')
        ax3.set_title('Estimated Strategy - Mean Transmissions')
        plt.colorbar(im3, ax=ax3)
        
        # Add text annotations
        for i in range(len(p_udp_values)):
            for j in range(len(p_tcp_values)):
                if not np.isnan(estimated_data[i, j]):
                    ax3.text(j, i, f'{estimated_data[i, j]:.1f}', ha='center', va='center', 
                            color='white' if estimated_data[i, j] > np.nanmean(estimated_data) else 'black')
        
        plt.tight_layout()
        plt.suptitle(rf'Mean Transmission Costs Summary ($m={self.n_messages}$, $v={self.n_vehicles}$)', 
                     fontsize=14, fontweight='bold', y=1.02)
        
        if save_plots:
            filename = f'transmission_summary_m{self.n_messages}_v{self.n_vehicles}.png'
            filepath = os.path.join(plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to: {filepath}")
        
        plt.show()
    
    def _save_histogram_plots_separately(self, results: Dict, p_tcp_values: list, p_udp_values: list, plots_dir: str):
        """Save individual histogram plots for each parameter combination."""
        import matplotlib.pyplot as plt
        
        # Create separate plots subdirectory
        separate_plots_dir = os.path.join(plots_dir, "Separate_Histograms")
        os.makedirs(separate_plots_dir, exist_ok=True)
        
        # Color scheme for strategies
        legend_colors = {
            'graph': '#2E8B57',     # Sea Green
            'mdp': '#FF6B35',       # Orange Red  
            'estimated': '#4169E1'  # Royal Blue
        }
        
        legend_labels = {
            'graph': 'Graph Shortest Path',
            'mdp': 'MDP Solution',
            'estimated': 'Estimated Heuristic'
        }
        
        print(f"Saving separate histogram plots to: {separate_plots_dir}")
        
        for sim in results['simulations']:
            p_tcp = sim['p_tcp']
            p_udp = sim['p_udp']
            
            # Create individual plot for this combination
            plt.figure(figsize=(12, 8))
            
            # Collect data for all strategies
            all_transmissions = []
            labels = []
            colors = []
            
            for strategy_type, strategy_name in [('graph_results', 'graph'), ('mdp_results', 'mdp'), ('estimated_results', 'estimated')]:
                if sim[strategy_type]['stats'] is not None and sim[strategy_type]['transmissions']:
                    transmissions = sim[strategy_type]['transmissions']
                    all_transmissions.append(transmissions)
                    labels.append(legend_labels[strategy_name])
                    colors.append(legend_colors[strategy_name])
            
            if not all_transmissions:
                plt.close()
                continue
            
            # Create histogram
            plt.hist(all_transmissions, bins=30, alpha=0.7, label=labels, color=colors, density=True)
            
            # Formatting
            plt.xlabel('Total Transmissions Required', fontsize=13, fontweight='bold')
            plt.ylabel('Probability Density', fontsize=13, fontweight='bold')
            plt.title(f'Transmission Distribution: p_tcp={p_tcp:.2f}, p_udp={p_udp:.1f}', 
                     fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3, linestyle=':')
            
            # Add statistics text box
            stats_text = []
            for i, (strategy_type, strategy_name) in enumerate([('graph_results', 'graph'), ('mdp_results', 'mdp'), ('estimated_results', 'estimated')]):
                if sim[strategy_type]['stats'] is not None:
                    stats = sim[strategy_type]['stats']
                    stats_text.append(f"{legend_labels[strategy_name]}: {stats['mean_transmissions']:.1f}±{stats['std_transmissions']:.1f}")
            
            if stats_text:
                plt.text(0.98, 0.98, '\n'.join(stats_text), 
                        transform=plt.gca().transAxes,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='gray'),
                        fontsize=10, verticalalignment='top', horizontalalignment='right')
            
            # Make the plot look professional
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().tick_params(labelsize=11)
            
            plt.tight_layout()
            
            # Save individual plot
            filename = f'histogram_tcp{p_tcp:.2f}_udp{p_udp:.1f}.png'
            filepath = os.path.join(separate_plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()  # Close to save memory
            
        print(f"✅ Saved {len(results['simulations'])} separate histogram plots")
    
    def _save_cdf_plots_separately(self, results: Dict, p_tcp_values: list, p_udp_values: list, plots_dir: str):
        """Save individual CDF plots for each parameter combination."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create separate CDF plots subdirectory
        separate_cdf_dir = os.path.join(plots_dir, "Separate_CDFs")
        os.makedirs(separate_cdf_dir, exist_ok=True)
        
        # Color scheme for strategies
        legend_colors = {
            'graph': '#2E8B57',     # Sea Green
            'mdp': '#FF6B35',       # Orange Red  
            'estimated': '#4169E1'  # Royal Blue
        }
        
        legend_labels = {
            'graph': 'Graph Shortest Path',
            'mdp': 'MDP Solution',
            'estimated': 'Estimated Heuristic'
        }
        
        print(f"Saving separate CDF plots to: {separate_cdf_dir}")
        
        for sim in results['simulations']:
            p_tcp = sim['p_tcp']
            p_udp = sim['p_udp']
            
            # Create individual CDF plot for this combination
            plt.figure(figsize=(12, 8))
            
            # Collect data and plot CDFs
            has_data = False
            for strategy_type, strategy_name in [('graph_results', 'graph'), ('mdp_results', 'mdp'), ('estimated_results', 'estimated')]:
                if sim[strategy_type]['stats'] is not None and sim[strategy_type]['transmissions']:
                    transmissions = np.array(sim[strategy_type]['transmissions'])
                    
                    # Calculate CDF
                    sorted_data = np.sort(transmissions)
                    y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    
                    # Plot CDF as step function
                    plt.step(sorted_data, y_values, where='post', linewidth=2.5, 
                            label=legend_labels[strategy_name], color=legend_colors[strategy_name])
                    has_data = True
            
            if not has_data:
                plt.close()
                continue
            
            # Formatting
            plt.xlabel('Total Transmissions Required', fontsize=13, fontweight='bold')
            plt.ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
            plt.title(f'CDF Comparison: p_tcp={p_tcp:.2f}, p_udp={p_udp:.1f}', 
                     fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3, linestyle=':')
            
            # Format y-axis as percentages
            plt.gca().set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            plt.gca().set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
            
            # Add statistics text box
            stats_text = []
            for strategy_type, strategy_name in [('graph_results', 'graph'), ('mdp_results', 'mdp'), ('estimated_results', 'estimated')]:
                if sim[strategy_type]['stats'] is not None:
                    stats = sim[strategy_type]['stats']
                    stats_text.append(f"{legend_labels[strategy_name]}: {stats['mean_transmissions']:.1f}±{stats['std_transmissions']:.1f}")
            
            if stats_text:
                plt.text(0.98, 0.02, '\n'.join(stats_text), 
                        transform=plt.gca().transAxes,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='gray'),
                        fontsize=10, verticalalignment='bottom', horizontalalignment='right')
            
            # Make the plot look professional
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().tick_params(labelsize=11)
            
            plt.tight_layout()
            
            # Save individual CDF plot
            filename = f'cdf_tcp{p_tcp:.2f}_udp{p_udp:.1f}.png'
            filepath = os.path.join(separate_cdf_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()  # Close to save memory
            
        print(f"✅ Saved {len(results['simulations'])} separate CDF plots")


# Main execution
def main():
    """Main function to run the communication simulation."""
    # Import required libraries
    import os
    import sys
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import csv
    from typing import Dict, List, Tuple, Optional
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create simulator instance
    simulator = CommunicationSimulator(n_messages=1, n_vehicles=10)
    
    # Define parameter ranges for comprehensive analysis
    p_tcp_values = [0.7, 0.8, 0.9, 1.0]
    p_udp_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Run simulation suite
    print("Starting comprehensive communication simulation...")
    print(f"Parameters: {len(p_tcp_values)} TCP values × {len(p_udp_values)} UDP values = {len(p_tcp_values) * len(p_udp_values)} combinations")
    print(f"Runs per combination: 1000")
    print(f"Total runs: {len(p_tcp_values) * len(p_udp_values) * 1000}")
    
    results = simulator.run_simulation_suite(
        p_tcp_values=p_tcp_values,
        p_udp_values=p_udp_values,
        num_runs=1000,
        initial_pattern="all_ones"
    )
    
    # Save results to CSV
    simulator.save_results_to_csv(results)
    
    # Create and save all plots
    simulator.plot_results(results, save_plots=True)
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()
