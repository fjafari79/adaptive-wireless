import numpy as np
from mdptoolbox import mdp
from itertools import product
from scipy.sparse import csr_matrix, lil_matrix
import os
import csv

# Time factor for TCP actions (TCP takes longer than UDP)
TCP_TIME_FACTOR = 1.2  # TCP actions take 20% more time than UDP actions

# Risk factors for actions (UDP is more risky due to no delivery guarantees)
TCP_RISK_FACTOR = 0.05  # Low risk: TCP has reliability mechanisms
UDP_RISK_FACTOR = 0.15  # Higher risk: UDP has no delivery guarantees

# Reward tuning parameters
RELIABILITY_BONUS_WEIGHT = 0.1    # Bonus for TCP guaranteed delivery
URGENCY_PENALTY_WEIGHT = 0.05     # Penalty scaling for urgent delivery needs
COMPLETION_BONUS_WEIGHT = 0.2     # Bonus for nearing completion
EFFICIENCY_BONUS_WEIGHT = 0.05    # Bonus for UDP multi-delivery efficiency

def get_optimal_discount_factor(n_messages, n_vehicles, scenario_type="balanced"):
    """
    Suggest optimal discount factor based on problem characteristics.
    
    Parameters:
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles
    scenario_type: str - "urgent", "balanced", or "patient"
    
    Returns:
    float - Recommended discount factor
    """
    problem_complexity = n_messages * n_vehicles
    
    if scenario_type == "urgent":
        # Lower discount for time-critical scenarios
        base_gamma = 0.7
    elif scenario_type == "patient":
        # Higher discount for optimal long-term planning
        base_gamma = 0.95
    else:  # balanced
        base_gamma = 0.9
    
    # Adjust based on problem size
    if problem_complexity <= 6:       # Small problems (1msg×3vehicles, 2msg×3vehicles)
        gamma_adjustment = 0.05
    elif problem_complexity <= 15:    # Medium problems
        gamma_adjustment = 0.0
    else:                            # Large problems
        gamma_adjustment = -0.05      # Slightly lower for computational efficiency
    
    return min(0.99, max(0.5, base_gamma + gamma_adjustment))

def generate_binary_arrays(length):
    return [np.array(bits) for bits in product([0, 1], repeat=length)]

def matrix_to_index(binary_matrix):
    return int("".join(map(str, binary_matrix.flatten())), 2)

def index_to_matrix(index, shape):
    binary_string = bin(index)[2:].zfill(shape[0] * shape[1])
    binary_array = np.array([int(bit) for bit in binary_string])
    return binary_array.reshape(shape)

def get_message_vehicle_ids(action, num_clients):
    msg_id, vehicle_id = divmod(action, num_clients)
    return msg_id, vehicle_id

def get_all_reachable_states(current_state, msg_id, is_tcp, vehicle_id):
    num_msg, num_vehicle = np.shape(current_state)
    next_states = []
    if is_tcp:
        next_states.append(current_state)
        state = np.copy(current_state)
        if state[msg_id, vehicle_id] == 1:
            state[msg_id, vehicle_id] = 0
            next_states.append(state)
    else:
        current_msg_status = np.copy(current_state[msg_id])
        all_possibles = generate_binary_arrays(current_msg_status.size)
        for array in all_possibles:
            if np.any(array > current_msg_status):
                continue
            next_state = np.copy(current_state)
            next_state[msg_id] = array
            next_states.append(next_state)
    return next_states

def transition_prob(current_state, next_state, msg_id, p_success, is_tcp, vehicle_id):
    current_msg = current_state[msg_id]
    next_msg    = next_state[msg_id]
    num_received_msg = np.sum(current_msg) - np.sum(next_msg)

    prob = 0                            # default probability is zero
    if is_tcp:                          # TCP used

        if current_msg[vehicle_id] == 0:
            prob = 1
        elif num_received_msg == 1:
            prob = p_success
        elif num_received_msg == 0:
            prob = 1 - p_success

    else:                               # UDP used
        if num_received_msg < 0:
            prob = 0
        elif p_success == 0:
            if num_received_msg == 0:   # if p_udp = 0, the state won't change
                prob = 1
        elif p_success == 1:            # if p_udp = 1, all the messages would be received
            if np.sum(next_msg) == 0:
                prob = 1  
        else:
            prob = p_success ** num_received_msg * (1 - p_success) ** (np.sum(next_msg))

    return num_received_msg, prob

def calculate_risk_adjusted_reward(num_received_msg, is_tcp, current_state, n_messages, n_vehicles):
    """
    Calculate risk-adjusted reward incorporating time factor, risk penalty, and completion bonus.
    
    Parameters:
    num_received_msg: int - Number of messages received
    is_tcp: bool - Whether the action is TCP (True) or UDP (False)
    current_state: numpy array - Current state matrix
    n_messages: int - Total number of messages
    n_vehicles: int - Total number of vehicles
    
    Returns:
    float - Risk-adjusted reward
    """
    # Base reward with time factor
    if is_tcp:
        base_reward = num_received_msg / TCP_TIME_FACTOR  # Time penalty for TCP
        reliability_bonus = 0.1 * num_received_msg  # Bonus for guaranteed delivery
        risk_factor = TCP_RISK_FACTOR
    else:
        base_reward = num_received_msg
        reliability_bonus = 0.0  # No reliability bonus for UDP
        risk_factor = UDP_RISK_FACTOR
    
    # Calculate state-based factors
    total_undelivered = np.sum(current_state)  # Count of 1s (undelivered messages)
    max_possible_undelivered = n_messages * n_vehicles
    completion_ratio = 1.0 - (total_undelivered / max_possible_undelivered)
    
    # 1. Risk penalty: Increases with undelivered messages (encourages progress)
    risk_penalty = risk_factor * (total_undelivered / max_possible_undelivered)
    
    # 2. Urgency factor: Higher penalty when many messages remain undelivered
    urgency_penalty = 0.05 * (total_undelivered / max_possible_undelivered) ** 2
    
    # 3. Completion bonus: Exponential bonus as we approach full delivery
    completion_bonus = 0.2 * (completion_ratio ** 3) if completion_ratio > 0.5 else 0.0
    
    # 4. Efficiency reward: Bonus for delivering multiple messages in one action (UDP broadcast)
    efficiency_bonus = 0.05 * max(0, num_received_msg - 1) if not is_tcp else 0.0
    
    # Combine all factors
    total_reward = (base_reward + 
                   reliability_bonus + 
                   completion_bonus + 
                   efficiency_bonus - 
                   risk_penalty - 
                   urgency_penalty)
    
    return total_reward

def save_mdp_solution(policy, value_function, n_messages, n_vehicles, tcp_success, udp_success, discount_factor):
    """
    Save MDP solution to CSV file in Brain/MDP folder.
    
    Parameters:
    policy: array - Optimal policy from MDP solver
    value_function: array - Value function from MDP solver  
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles
    tcp_success: float - TCP success probability
    udp_success: float - UDP success probability
    discount_factor: float - MDP discount factor
    """
    # Create Brain/MDP directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    brain_dir = os.path.join(current_dir, "Brain")
    mdp_dir = os.path.join(brain_dir, "MDP")
    os.makedirs(mdp_dir, exist_ok=True)
    
    # Create filename with parameter template (standardized format: base + MDP-specific params)
    filename = generate_mdp_solution_filename(n_messages, n_vehicles, tcp_success, udp_success, discount_factor)
    filepath = os.path.join(mdp_dir, filename)
    
    # Prepare solution data
    solution_data = []
    num_actions_TCP = n_messages * n_vehicles
    
    for state_idx in range(len(policy)):
        state_matrix = index_to_matrix(state_idx, (n_messages, n_vehicles))
        optimal_action = policy[state_idx]
        state_value = value_function[state_idx]
        
        # Convert matrix to standardized string representation (rows separated by |)
        matrix_str = '|'.join(''.join(map(str, row)) for row in state_matrix)
        
        # Determine action description in standardized format
        if optimal_action < num_actions_TCP:
            msg_id, vehicle_id = divmod(optimal_action, n_vehicles)
            action_description = f"TCP_m_{msg_id}_{vehicle_id}"
        else:
            msg_id = optimal_action - num_actions_TCP
            action_description = f"UDP_m_{msg_id}"
        
        # Calculate next state by applying the action
        next_state = calculate_next_state_from_action(state_matrix, optimal_action, n_messages, n_vehicles)
        
        solution_data.append({
            'state_id': state_idx,
            'state_matrix': matrix_str,
            'action': action_description,
            'next_state': next_state,
            'state_value': state_value
        })
    
    # Save to CSV with standardized format
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['state_id', 'state_matrix', 'action', 'next_state', 'state_value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(solution_data)
    
    print(f"MDP solution saved to: {filepath}")
    return filepath

def load_mdp_solution(n_messages, n_vehicles, tcp_success, udp_success, discount_factor):
    """
    Load MDP solution from CSV file if it exists.
    
    Parameters:
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles
    tcp_success: float - TCP success probability
    udp_success: float - UDP success probability
    discount_factor: float - MDP discount factor
    
    Returns:
    dict or None - Solution data if file exists, None otherwise
    """
    # Create Brain/MDP directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    brain_dir = os.path.join(current_dir, "Brain")
    mdp_dir = os.path.join(brain_dir, "MDP")
    
    # Create filename with parameter template (standardized format: base + MDP-specific params)
    filename = generate_mdp_solution_filename(n_messages, n_vehicles, tcp_success, udp_success, discount_factor)
    filepath = os.path.join(mdp_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        # Load solution data from CSV
        solution_data = {}
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                state_id = int(row['state_id'])
                solution_data[state_id] = {
                    'state_matrix': row['state_matrix'],
                    'action': row['action'],
                    'next_state': int(row['next_state']),
                    'state_value': float(row['state_value'])
                }
        
        print(f"MDP solution loaded from: {filepath}")
        return solution_data
    
    except Exception as e:
        print(f"Error loading MDP solution: {e}")
        return None

def generate_mdp_solution_filename(n_messages, n_vehicles, tcp_success, udp_success, discount_factor):
    """
    Generate filename for MDP solution based on parameters (standardized format: base + MDP-specific params).
    
    Parameters:
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles
    tcp_success: float - TCP success probability
    udp_success: float - UDP success probability
    discount_factor: float - MDP discount factor
    
    Returns:
    str - Filename for the MDP solution
    """
    return f"m_{n_messages}_v_{n_vehicles}_TCP_{tcp_success}_UDP_{udp_success}_discount_{discount_factor}_tcpRisk_{TCP_RISK_FACTOR}_udpRisk_{UDP_RISK_FACTOR}.csv"

def get_or_generate_mdp_solution(n_messages, n_vehicles, tcp_success, udp_success, discount_factor):
    """
    Get existing MDP solution or generate a new one.
    
    Parameters:
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles
    tcp_success: float - TCP success probability
    udp_success: float - UDP success probability
    discount_factor: float - MDP discount factor
    
    Returns:
    dict - Solution data mapping state_id to solution info
    """
    # Try to load existing solution
    solution_data = load_mdp_solution(n_messages, n_vehicles, tcp_success, udp_success, discount_factor)
    
    if solution_data is not None:
        print("Using existing MDP solution.")
        return solution_data
    
    print("Generating new MDP solution...")
    print(f"TCP time factor: {TCP_TIME_FACTOR} (TCP rewards will be divided by this factor)")
    print(f"Risk factors - TCP: {TCP_RISK_FACTOR}, UDP: {UDP_RISK_FACTOR} (UDP is more risky)")
    
    # Generate new solution using existing MDP code
    num_states = 2 ** (n_messages * n_vehicles)
    print(f"Total states: {num_states}")
    
    # Warn for large state spaces and offer to skip
    if num_states > 16384:  # 2^14
        print("WARNING: Large state space detected!")
        print(f"With {num_states} states, MDP value iteration may take very long (10+ minutes)")
        print("Consider using smaller parameters for faster results.")
        
        # For extremely large state spaces, offer to skip
        if num_states > 65536:  # 2^16
            print(f"ERROR: State space too large ({num_states} states)")
            print("MDP solution skipped - use smaller parameters (e.g., fewer messages/vehicles)")
            return None
        
        print("Proceeding with optimized settings...")
    
    num_actions_TCP = n_messages * n_vehicles
    num_actions_UDP = n_messages
    num_actions = num_actions_TCP + num_actions_UDP
    
    # Use LIL matrix for efficient construction, then convert to CSR
    P = [lil_matrix((num_states, num_states)) for _ in range(num_actions)]
    R = [lil_matrix((num_states, num_states)) for _ in range(num_actions)]
    
    print(f"Building transition matrices for {num_actions} actions...")
    
    # Build transition and reward matrices
    for current_state_ind in range(num_states):
        # Progress reporting for large state spaces
        if num_states > 1024 and current_state_ind % max(1, num_states // 20) == 0:
            progress = (current_state_ind / num_states) * 100
            print(f"Matrix construction progress: {progress:.1f}% ({current_state_ind}/{num_states})")
        
        current_state = index_to_matrix(current_state_ind, (n_messages, n_vehicles))
        
        for action in range(num_actions):
            if action < num_actions_TCP:    # TCP action
                is_tcp = 1
                msg_id, vehicle_id = get_message_vehicle_ids(action, n_vehicles)
                p_msg_success = tcp_success
            else:                           # UDP action
                is_tcp = 0
                msg_id = action - num_actions_TCP
                vehicle_id = -1
                p_msg_success = udp_success
            
            next_states = get_all_reachable_states(current_state, msg_id, is_tcp, vehicle_id)
            
            for next_state in next_states:
                next_state_ind = matrix_to_index(next_state)
                num_received_msg, transition_p = transition_prob(current_state, next_state, msg_id, p_msg_success, is_tcp, vehicle_id)
                P[action][current_state_ind, next_state_ind] = transition_p
                
                # Calculate risk-adjusted reward with TCP time factor and risk penalties
                reward = calculate_risk_adjusted_reward(num_received_msg, is_tcp, current_state, n_messages, n_vehicles)
                
                R[action][current_state_ind, next_state_ind] = reward
    
    print("Matrix construction completed!")
    print("Converting matrices to CSR format for solver...")
    
    # Convert LIL matrices to CSR for efficient solving
    P = [matrix.tocsr() for matrix in P]
    R = [matrix.tocsr() for matrix in R]
    
    print(f"Starting MDP value iteration for {num_states} states...")
    print("This may take several minutes for large state spaces...")
    
    # Use value iteration with more lenient convergence criteria for large problems
    mdp_solver = mdp.ValueIteration(P, R, discount_factor, epsilon=0.01, max_iter=1000)
    
    try:
        mdp_solver.run()
        print(f"MDP solution converged after {mdp_solver.iter} iterations")
    except Exception as e:
        print(f"MDP solver encountered an issue: {e}")
        print("Trying with even more lenient settings...")
        # Try with very lenient settings as fallback
        mdp_solver = mdp.ValueIteration(P, R, discount_factor, epsilon=0.1, max_iter=500)
        mdp_solver.run()
        print(f"MDP solution converged after {mdp_solver.iter} iterations (lenient settings)")
    
    print("MDP value iteration completed!")
    
    # Save solution
    filepath = save_mdp_solution(mdp_solver.policy, mdp_solver.V, n_messages, n_vehicles, 
                                tcp_success, udp_success, discount_factor)
    
    # Return solution data
    solution_data = {}
    for state_idx in range(len(mdp_solver.policy)):
        state_matrix = index_to_matrix(state_idx, (n_messages, n_vehicles))
        optimal_action = mdp_solver.policy[state_idx]
        state_value = mdp_solver.V[state_idx]
        
        # Convert matrix to standardized string representation (rows separated by |)
        matrix_str = '|'.join(''.join(map(str, row)) for row in state_matrix)
        
        # Determine action description in standardized format
        if optimal_action < num_actions_TCP:
            msg_id, vehicle_id = divmod(optimal_action, n_vehicles)
            action_description = f"TCP_m_{msg_id}_{vehicle_id}"
        else:
            msg_id = optimal_action - num_actions_TCP
            action_description = f"UDP_m_{msg_id}"
        
        # Calculate next state by applying the action
        next_state = calculate_next_state_from_action(state_matrix, optimal_action, n_messages, n_vehicles)
        
        solution_data[state_idx] = {
            'state_matrix': matrix_str,
            'action': action_description,
            'next_state': next_state,
            'state_value': state_value
        }
    
    return solution_data

def calculate_next_state_from_action(state_matrix, action, n_messages, n_vehicles):
    """
    Calculate the next state after applying an action.
    
    Parameters:
    state_matrix: numpy array - Current state matrix
    action: int - Action to apply
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles
    
    Returns:
    int - Next state ID
    """
    num_actions_TCP = n_messages * n_vehicles
    next_matrix = state_matrix.copy()
    
    if action < num_actions_TCP:  # TCP action
        msg_id, vehicle_id = divmod(action, n_vehicles)
        if next_matrix[msg_id, vehicle_id] == 1:
            next_matrix[msg_id, vehicle_id] = 0
    else:  # UDP action
        msg_id = action - num_actions_TCP
        next_matrix[msg_id, :] = 0
    
    return matrix_to_index(next_matrix)

if __name__ == "__main__":
    # Example usage with trade-off analysis
    num_messages = 2
    num_clients = 3
    tcp_success = 0.95
    udp_success = 0.8

    print("=== MDP TRADE-OFF ANALYSIS ===")
    
    # Analyze different discount factors
    discount_factors = [0.7, 0.85, 0.95]
    
    for df in discount_factors:
        print(f"\n--- Discount Factor: {df} ---")
        optimal_df = get_optimal_discount_factor(num_messages, num_clients, "balanced")
        print(f"Recommended discount factor: {optimal_df}")
        
        if abs(df - optimal_df) < 0.1:
            print("✓ Good choice for balanced approach")
        elif df < optimal_df:
            print("⚡ More urgent/short-term focused")
        else:
            print("🎯 More patient/long-term focused")
    
    # Use recommended discount factor
    discount_factor = get_optimal_discount_factor(num_messages, num_clients, "balanced")
    print(f"\nUsing discount factor: {discount_factor}")

    # Generate solution with trade-off analysis
    print("\n=== MDP SOLUTION ===")
    solution_data = get_or_generate_mdp_solution(num_messages, num_clients, tcp_success, udp_success, discount_factor)

    # Analyze trade-offs in the solution
    if solution_data:
        print("\n=== TRADE-OFF ANALYSIS ===")
        tcp_actions = 0
        udp_actions = 0
        
        for state_id, solution in solution_data.items():
            action = solution['action']
            if action.startswith('TCP'):
                tcp_actions += 1
            else:
                udp_actions += 1
        
        total_actions = tcp_actions + udp_actions
        tcp_ratio = tcp_actions / total_actions
        udp_ratio = udp_actions / total_actions
        
        print(f"Policy composition:")
        print(f"  TCP actions: {tcp_actions}/{total_actions} ({tcp_ratio:.1%})")
        print(f"  UDP actions: {udp_actions}/{total_actions} ({udp_ratio:.1%})")
        
        if tcp_ratio > 0.6:
            print("📡 TCP-heavy strategy: Prioritizes reliability over speed")
        elif udp_ratio > 0.6:
            print("⚡ UDP-heavy strategy: Prioritizes speed and efficiency")
        else:
            print("⚖️ Balanced strategy: Good mix of reliability and efficiency")

    print("\nOptimal Policy (first 10 states):")
    num_states = 2 ** (num_messages * num_clients)
    for i in range(min(10, num_states)):  # Show first 10 states as example
        state_matrix = index_to_matrix(i, (num_messages, num_clients))
        if solution_data and i in solution_data:
            action = solution_data[i]['action']
            next_state = solution_data[i]['next_state']
            state_value = solution_data[i]['state_value']
            print(f"State {i}:")
            print(f"Matrix:\n{state_matrix}")
            print(f"Optimal action: {action}")
            print(f"Next state: {next_state}")
            print(f"State value: {state_value:.4f}")
        else:
            print(f"State {i}: No solution data")
        print("-" * 40)
    
    if num_states > 10:
        print(f"... and {num_states - 10} more states")
    
    print(f"\nTotal states: {num_states}")
    print(f"Solution saved/loaded successfully!")
