#!/usr/bin/env python3
"""
Expected Receivers Strategy
A new estimation strategy that chooses protocol based on expected number of receivers.

For each message state (row), calculate:
- Expected receivers = p_udp * (number of ones in the row)
- If expected_receivers > p_tcp/TCP_TIME_FACTOR: choose UDP
- Otherwise: choose TCP

This strategy is based on the intuition that UDP is better when we expect many receivers,
while TCP is better when we expect few receivers. The TCP comparison is adjusted by the
time factor since TCP actions take longer.
"""

import numpy as np
import pandas as pd
import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Time factor for TCP actions (TCP takes longer than UDP)
TCP_TIME_FACTOR = 1.2  # TCP actions take 20% more time than UDP actions

def calculate_expected_receivers_strategy(n_messages: int, n_vehicles: int, p_tcp: float, p_udp: float) -> Dict:
    """
    Calculate the Expected Receivers strategy for given parameters.
    
    For each possible message state (combination of which vehicles have each message),
    decide whether to use TCP or UDP based on expected number of receivers.
    
    Parameters:
    n_messages: int - Number of messages in the system
    n_vehicles: int - Number of vehicles in the network
    p_tcp: float - TCP success probability
    p_udp: float - UDP success probability
    
    Returns:
    Dict containing the strategy decisions for each state
    """
    print(f"Calculating Expected Receivers strategy for m={n_messages}, v={n_vehicles}, p_tcp={p_tcp}, p_udp={p_udp}")
    print(f"TCP time factor: {TCP_TIME_FACTOR} (TCP threshold = p_tcp/{TCP_TIME_FACTOR} = {p_tcp/TCP_TIME_FACTOR:.4f})")
    
    # Generate all possible states
    # Each state is a binary matrix: rows=messages, cols=vehicles
    # state[i][j] = 1 if vehicle j has message i, 0 otherwise
    total_states = 2 ** (n_messages * n_vehicles)
    
    strategy_decisions = {}
    state_analysis = []
    
    print(f"Analyzing {total_states} possible states...")
    
    for state_id in range(total_states):
        # Convert state_id to binary matrix representation
        binary_repr = format(state_id, f'0{n_messages * n_vehicles}b')
        
        # Reshape into n_messages x n_vehicles matrix
        state_matrix = np.array([int(b) for b in binary_repr]).reshape(n_messages, n_vehicles)
        
        # For each message (row), decide protocol based on expected receivers
        message_decisions = []
        
        for msg_idx in range(n_messages):
            message_state = state_matrix[msg_idx]  # Which vehicles have this message
            ones_count = np.sum(message_state)      # Number of vehicles that have this message
            
            # Calculate expected number of receivers if we use UDP
            expected_receivers = p_udp * ones_count
            
            # Adjust TCP threshold by time factor (TCP takes longer, so threshold is lower)
            tcp_threshold = p_tcp / TCP_TIME_FACTOR
            
            # Decision rule: if expected_receivers > tcp_threshold, use UDP, otherwise TCP
            if expected_receivers > tcp_threshold:
                protocol_choice = 'UDP'
                chosen_prob = p_udp
            else:
                protocol_choice = 'TCP'
                chosen_prob = p_tcp
            
            message_decisions.append({
                'message_idx': msg_idx,
                'ones_count': ones_count,
                'expected_receivers': expected_receivers,
                'tcp_threshold': tcp_threshold,
                'protocol_choice': protocol_choice,
                'chosen_prob': chosen_prob,
                'comparison': f"{expected_receivers:.3f} {'>' if expected_receivers > tcp_threshold else '<='} {tcp_threshold:.3f}"
            })
        
        # Store strategy decision for this state
        strategy_decisions[state_id] = {
            'state_matrix': state_matrix.tolist(),
            'binary_repr': binary_repr,
            'message_decisions': message_decisions
        }
        
        # Collect analysis data
        state_analysis.append({
            'state_id': state_id,
            'binary_repr': binary_repr,
            'total_ones': np.sum(state_matrix),
            'tcp_messages': sum(1 for d in message_decisions if d['protocol_choice'] == 'TCP'),
            'udp_messages': sum(1 for d in message_decisions if d['protocol_choice'] == 'UDP'),
            'avg_expected_receivers': np.mean([d['expected_receivers'] for d in message_decisions]),
            'decisions_detail': [f"M{d['message_idx']}:{d['protocol_choice']}" for d in message_decisions]
        })
        
        # Progress indicator for large state spaces
        if total_states > 1000 and (state_id + 1) % (total_states // 10) == 0:
            progress = (state_id + 1) / total_states * 100
            print(f"  Progress: {progress:.1f}% ({state_id + 1}/{total_states})")
    
    print("Expected Receivers strategy calculation completed!")
    
    # Print summary statistics
    tcp_decisions = sum(d['tcp_messages'] for d in state_analysis)
    udp_decisions = sum(d['udp_messages'] for d in state_analysis)
    total_decisions = tcp_decisions + udp_decisions
    
    print(f"Strategy Summary:")
    print(f"  Total message-state combinations: {total_decisions}")
    print(f"  TCP chosen: {tcp_decisions} ({tcp_decisions/total_decisions*100:.1f}%)")
    print(f"  UDP chosen: {udp_decisions} ({udp_decisions/total_decisions*100:.1f}%)")
    
    return {
        'strategy_decisions': strategy_decisions,
        'state_analysis': state_analysis,
        'parameters': {
            'n_messages': n_messages,
            'n_vehicles': n_vehicles,
            'p_tcp': p_tcp,
            'p_udp': p_udp
        },
        'summary': {
            'total_states': total_states,
            'total_decisions': total_decisions,
            'tcp_decisions': tcp_decisions,
            'udp_decisions': udp_decisions,
            'tcp_percentage': tcp_decisions/total_decisions*100,
            'udp_percentage': udp_decisions/total_decisions*100
        }
    }

def save_expected_receivers_strategy(strategy_data: Dict, save_dir: str = None) -> str:
    """
    Save the Expected Receivers strategy to CSV files.
    
    Parameters:
    strategy_data: Dict - Output from calculate_expected_receivers_strategy()
    save_dir: str - Directory to save files (optional, defaults to Brain/ExpectedReceivers)
    
    Returns:
    str - Path to the main strategy file
    """
    params = strategy_data['parameters']
    n_messages = params['n_messages']
    n_vehicles = params['n_vehicles']
    p_tcp = params['p_tcp']
    p_udp = params['p_udp']
    
    # Create save directory
    if save_dir is None:
        current_dir = Path(__file__).parent
        brain_dir = current_dir / "Brain"
        save_dir = brain_dir / "ExpectedReceivers"
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename based on parameters
    filename_base = f"expected_receivers_m{n_messages}_v{n_vehicles}_ptcp{p_tcp:.3f}_pudp{p_udp:.3f}"
    
    # Save main strategy file (detailed decisions for each state)
    strategy_file = save_dir / f"{filename_base}_strategy.csv"
    print(f"Saving detailed strategy to: {strategy_file}")
    
    with open(strategy_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['state_id', 'binary_repr', 'message_idx', 'ones_count', 
                 'expected_receivers', 'tcp_threshold', 'protocol_choice', 'chosen_prob', 'comparison']
        writer.writerow(header)
        
        # Write data
        for state_id, state_data in strategy_data['strategy_decisions'].items():
            for msg_decision in state_data['message_decisions']:
                row = [
                    state_id,
                    state_data['binary_repr'],
                    msg_decision['message_idx'],
                    msg_decision['ones_count'],
                    round(msg_decision['expected_receivers'], 6),
                    round(msg_decision['tcp_threshold'], 6),
                    msg_decision['protocol_choice'],
                    round(msg_decision['chosen_prob'], 6),
                    msg_decision['comparison']
                ]
                writer.writerow(row)
    
    # Save summary analysis file
    summary_file = save_dir / f"{filename_base}_summary.csv"
    print(f"Saving state analysis summary to: {summary_file}")
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['state_id', 'binary_repr', 'total_ones', 'tcp_messages', 'udp_messages',
                 'avg_expected_receivers', 'decisions_detail']
        writer.writerow(header)
        
        # Write data
        for analysis in strategy_data['state_analysis']:
            row = [
                analysis['state_id'],
                analysis['binary_repr'],
                analysis['total_ones'],
                analysis['tcp_messages'],
                analysis['udp_messages'],
                round(analysis['avg_expected_receivers'], 6),
                '; '.join(analysis['decisions_detail'])
            ]
            writer.writerow(row)
    
    # Save metadata file
    metadata_file = save_dir / f"{filename_base}_metadata.csv"
    print(f"Saving metadata to: {metadata_file}")
    
    with open(metadata_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'value'])
        writer.writerow(['n_messages', n_messages])
        writer.writerow(['n_vehicles', n_vehicles])
        writer.writerow(['p_tcp', p_tcp])
        writer.writerow(['p_udp', p_udp])
        writer.writerow(['total_states', strategy_data['summary']['total_states']])
        writer.writerow(['total_decisions', strategy_data['summary']['total_decisions']])
        writer.writerow(['tcp_decisions', strategy_data['summary']['tcp_decisions']])
        writer.writerow(['udp_decisions', strategy_data['summary']['udp_decisions']])
        writer.writerow(['tcp_percentage', round(strategy_data['summary']['tcp_percentage'], 2)])
        writer.writerow(['udp_percentage', round(strategy_data['summary']['udp_percentage'], 2)])
        writer.writerow(['tcp_time_factor', TCP_TIME_FACTOR])
        writer.writerow(['strategy_description', f'Expected Receivers: Choose UDP if p_udp * ones_count > p_tcp/{TCP_TIME_FACTOR}, else TCP'])
    
    print(f"Expected Receivers strategy saved successfully!")
    print(f"Files created:")
    print(f"  - Strategy details: {strategy_file}")
    print(f"  - State analysis: {summary_file}")
    print(f"  - Metadata: {metadata_file}")
    
    return str(strategy_file)

def generate_expected_receivers_strategies():
    """
    Generate Expected Receivers strategies for common parameter combinations.
    """
    print("=" * 60)
    print("GENERATING EXPECTED RECEIVERS STRATEGIES")
    print("=" * 60)
    
    # Define parameter combinations to generate
    parameter_sets = [
        # Standard test cases
        {'n_messages': 1, 'n_vehicles': 10, 'p_tcp': 0.95, 'p_udp': 0.8},
        {'n_messages': 1, 'n_vehicles': 10, 'p_tcp': 0.9, 'p_udp': 0.7},
        {'n_messages': 1, 'n_vehicles': 10, 'p_tcp': 0.8, 'p_udp': 0.9},
        
        # Different vehicle counts
        {'n_messages': 1, 'n_vehicles': 5, 'p_tcp': 0.95, 'p_udp': 0.8},
        {'n_messages': 1, 'n_vehicles': 15, 'p_tcp': 0.95, 'p_udp': 0.8},
        
        # Multiple messages (if computationally feasible)
        {'n_messages': 2, 'n_vehicles': 5, 'p_tcp': 0.95, 'p_udp': 0.8},
    ]
    
    generated_files = []
    
    for i, params in enumerate(parameter_sets, 1):
        print(f"\n[{i}/{len(parameter_sets)}] Generating strategy for parameters:")
        print(f"  Messages: {params['n_messages']}")
        print(f"  Vehicles: {params['n_vehicles']}")
        print(f"  p_tcp: {params['p_tcp']}")
        print(f"  p_udp: {params['p_udp']}")
        
        # Check if state space is manageable
        total_states = 2 ** (params['n_messages'] * params['n_vehicles'])
        if total_states > 100000:
            print(f"  Skipping - state space too large ({total_states} states)")
            continue
        
        try:
            # Calculate strategy
            strategy_data = calculate_expected_receivers_strategy(**params)
            
            # Save strategy
            strategy_file = save_expected_receivers_strategy(strategy_data)
            generated_files.append(strategy_file)
            
        except Exception as e:
            print(f"  Error generating strategy: {e}")
            continue
    
    print(f"\n" + "=" * 60)
    print(f"GENERATION COMPLETE")
    print(f"Successfully generated {len(generated_files)} strategy files")
    for file_path in generated_files:
        print(f"  - {file_path}")
    print("=" * 60)
    
    return generated_files

def load_expected_receivers_strategy(n_messages: int, n_vehicles: int, p_tcp: float, p_udp: float) -> Optional[Dict]:
    """
    Load a previously generated Expected Receivers strategy from CSV file.
    If the strategy doesn't exist, generate it on-the-fly.
    
    Parameters:
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles  
    p_tcp: float - TCP success probability
    p_udp: float - UDP success probability
    
    Returns:
    Dict containing the loaded strategy or None if not found
    """
    # Construct expected filename
    current_dir = Path(__file__).parent
    brain_dir = current_dir / "Brain" / "ExpectedReceivers"
    filename_base = f"expected_receivers_m{n_messages}_v{n_vehicles}_ptcp{p_tcp:.3f}_pudp{p_udp:.3f}"
    strategy_file = brain_dir / f"{filename_base}_strategy.csv"
    
    # If strategy file doesn't exist, generate it
    if not strategy_file.exists():
        print(f"Expected Receivers strategy not found. Generating on-the-fly...")
        
        # Check if state space is manageable
        total_states = 2 ** (n_messages * n_vehicles)
        if total_states > 100000:
            print(f"State space too large ({total_states} states) - skipping Expected Receivers strategy")
            return None
        
        try:
            # Generate the strategy
            strategy_data = calculate_expected_receivers_strategy(n_messages, n_vehicles, p_tcp, p_udp)
            
            # Save it for future use
            save_expected_receivers_strategy(strategy_data)
            
            print(f"Generated and saved Expected Receivers strategy")
        except Exception as e:
            print(f"Error generating Expected Receivers strategy: {e}")
            return None
    
    print(f"Loading Expected Receivers strategy from: {strategy_file}")
    
    try:
        # Load strategy data
        strategy_decisions = {}
        
        with open(strategy_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                state_id = int(row['state_id'])
                
                if state_id not in strategy_decisions:
                    strategy_decisions[state_id] = {
                        'binary_repr': row['binary_repr'],
                        'message_decisions': []
                    }
                
                strategy_decisions[state_id]['message_decisions'].append({
                    'message_idx': int(row['message_idx']),
                    'ones_count': int(row['ones_count']),
                    'expected_receivers': float(row['expected_receivers']),
                    'protocol_choice': row['protocol_choice'],
                    'chosen_prob': float(row['chosen_prob']),
                    'comparison': row['comparison']
                })
        
        print(f"Successfully loaded strategy with {len(strategy_decisions)} states")
        
        # Convert to standard format (flat dictionary like Graph and MDP strategies)
        standard_strategy = {}
        
        for state_id, state_data in strategy_decisions.items():
            # Convert state_id back to matrix to see which messages need to be sent
            binary_repr = state_data['binary_repr']
            state_matrix = np.array([int(b) for b in binary_repr]).reshape(n_messages, n_vehicles)
            
            # Find the first message that still needs to be sent (has vehicles that need it)
            action_found = False
            for msg_decision in state_data['message_decisions']:
                msg_id = msg_decision['message_idx']
                
                # Check if this message still needs to be sent (any vehicle has 1 = needs message)
                message_row_sum = np.sum(state_matrix[msg_id, :])
                if message_row_sum > 0:
                    protocol_choice = msg_decision['protocol_choice']
                    
                    if protocol_choice == "TCP":
                        # For TCP, find the first vehicle that needs this message
                        vehicle_indices = np.where(state_matrix[msg_id, :] == 1)[0]
                        if len(vehicle_indices) > 0:
                            vehicle_id = vehicle_indices[0]  # First vehicle that needs it
                            action = f"TCP_m_{msg_id}_{vehicle_id}"
                        else:
                            continue  # Skip if no vehicle needs it (shouldn't happen)
                    else:
                        # For UDP, broadcast to all vehicles
                        action = f"UDP_m_{msg_id}"
                    
                    standard_strategy[state_id] = {
                        'action': action,
                        'next_state': None,  # Will be calculated during simulation
                        'strategy_type': 'expected_receivers',
                        'message_id': msg_id,
                        'protocol': protocol_choice
                    }
                    action_found = True
                    break
            
            # If no message needs to be sent, this is terminal state
            if not action_found:
                standard_strategy[state_id] = {
                    'action': 'TERMINAL',
                    'next_state': None,
                    'strategy_type': 'expected_receivers',
                    'message_id': None,
                    'protocol': None
                }
        
        print(f"Converted {len(standard_strategy)} states to standard format")
        return standard_strategy
        
    except Exception as e:
        print(f"Error loading strategy: {e}")
        return None

def main():
    """
    Main function to generate Expected Receivers strategies.
    """
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            # Generate multiple strategies
            generate_expected_receivers_strategies()
        elif sys.argv[1] == "test":
            # Test single strategy generation
            print("Testing Expected Receivers strategy generation...")
            strategy_data = calculate_expected_receivers_strategy(
                n_messages=1, n_vehicles=5, p_tcp=0.95, p_udp=0.8
            )
            save_expected_receivers_strategy(strategy_data)
            print("Test completed!")
    else:
        print("Expected Receivers Strategy Generator")
        print("Usage:")
        print("  python expected_receivers_solution.py generate  - Generate multiple strategies")
        print("  python expected_receivers_solution.py test     - Test with small example")

if __name__ == "__main__":
    main()
