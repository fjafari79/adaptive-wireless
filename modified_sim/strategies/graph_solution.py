import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import os
import csv

# Parameters
n_messages = 2
n_vehicles = 3
matrix_shape = (n_messages, n_vehicles)
p_udp = 0.9  # Probability parameter for UDP weight calculation
p_tcp = 0.95  # Probability parameter for TCP weight calculation (p_tcp > p_udp)

# Time factor for TCP actions (TCP takes longer than UDP)
TCP_TIME_FACTOR = 1.2  # TCP actions take 20% more time than UDP actions

# Generate all possible binary matrices and compute their sums
def generate_states():
    states = []
    for bits in product([0, 1], repeat=n_messages * n_vehicles):
        matrix = np.array(bits, dtype=np.uint8).reshape(matrix_shape)
        total = np.sum(matrix)
        states.append((matrix, total))
    return states

def matrix_to_str(matrix):
    return "\n".join("".join(str(cell) for cell in row) for row in matrix)

def matrix_to_index(matrix):
    flat = matrix.flatten()
    return int("".join(map(str, flat)), 2)

def valid_udp_transitions(matrix):
    transitions = []
    for row_idx in range(n_messages):
        current_row = matrix[row_idx]
        ones_indices = [i for i, val in enumerate(current_row) if val == 1]
        n = len(ones_indices)

        for bits in range(0, 2**n):  # include the case where entire row goes to 0
            new_row = current_row.copy()
            for j, col in enumerate(ones_indices):
                if (bits >> j) & 1 == 0:
                    new_row[col] = 0
            new_matrix = matrix.copy()
            new_matrix[row_idx] = new_row
            if not np.array_equal(matrix, new_matrix):  # skip identity
                transitions.append((new_matrix, f"udp#{row_idx}"))
    return transitions

def count_flips(source_matrix, target_matrix):
    """
    Count the number of flips (1's becoming 0's) between two matrices.
    
    Parameters:
    source_matrix: numpy array - The source state matrix
    target_matrix: numpy array - The target state matrix
    
    Returns:
    int - Number of 1's that became 0's (delta)
    """
    delta = 0
    for i in range(source_matrix.shape[0]):
        for j in range(source_matrix.shape[1]):
            if source_matrix[i, j] == 1 and target_matrix[i, j] == 0:
                delta += 1
    return delta

def calculate_weight(delta, probability=None):
    """
    Calculate the weight based on the number of flips and probability.
    
    Parameters:
    delta: int - Number of flips (1's to 0's)
    probability: float - Probability parameter (uses global p_udp if None)
    
    Returns:
    float - The calculated weight: 1 + (1/p - 1)^delta
    """
    if probability is None:
        probability = p_udp
    
    # Handle special cases
    if probability == 0:
        # When p=0, the weight becomes infinite for any delta > 0
        # We'll use a large finite value instead
        if delta == 0:
            weight = 1.0
        else:
            weight = float('inf')  # or use a large finite value like 1e6
    elif probability == 1:
        # When p=1, (1/p - 1) = 0, so weight = 1 + 0^delta = 1 for any delta
        weight = 1.0
    else:
        # Normal case: weight = 1 + (1/p - 1)^delta
        weight = 1 + (1/probability - 1)**delta
    
    return weight

def calculate_tcp_weight():
    """
    Calculate the TCP weight based on p_tcp and apply time factor.
    
    Returns:
    float - The TCP weight: (1/p_tcp) * TCP_TIME_FACTOR
    """
    if p_tcp == 0:
        return float('inf')
    else:
        return (1 / p_tcp) * TCP_TIME_FACTOR

def is_single_flip_transition(source_matrix, target_matrix):
    """
    Check if the transition involves exactly one flip (1 to 0).
    
    Parameters:
    source_matrix: numpy array - The source state matrix
    target_matrix: numpy array - The target state matrix
    
    Returns:
    bool - True if exactly one element flips from 1 to 0
    """
    delta = count_flips(source_matrix, target_matrix)
    return delta == 1

def find_flipped_position(source_matrix, target_matrix):
    """
    Find the position of the single flipped bit (1 to 0).
    
    Parameters:
    source_matrix: numpy array - The source state matrix
    target_matrix: numpy array - The target state matrix
    
    Returns:
    tuple - (row, col) of the flipped position, or None if not a single flip
    """
    for i in range(source_matrix.shape[0]):
        for j in range(source_matrix.shape[1]):
            if source_matrix[i, j] == 1 and target_matrix[i, j] == 0:
                # Check if this is the only flip
                if count_flips(source_matrix, target_matrix) == 1:
                    return (i, j)
    return None

def assign_edge_weights(G, states_dict_param=None, weight_function=None):
    """
    Assign weights to edges in the graph using both UDP and TCP calculations.
    
    Parameters:
    G: NetworkX DiGraph - The graph to assign weights to
    states_dict_param: dict - Dictionary mapping node indices to (matrix, total) tuples
    weight_function: callable - Function that takes (source_node, target_node, edge_data) 
                                and returns a weight value. If None, uses default weights.
    
    Returns:
    NetworkX DiGraph - The graph with weights assigned to edges
    """
    global states_dict
    # Use parameter if provided, otherwise use global
    current_states_dict = states_dict_param if states_dict_param is not None else states_dict
    
    if weight_function is None:
        # Default weight function that considers both UDP and TCP
        def default_weight_function(source_node, target_node, edge_data):
            source_matrix = current_states_dict[source_node][0]
            target_matrix = current_states_dict[target_node][0]
            
            # Calculate UDP weight
            delta = count_flips(source_matrix, target_matrix)
            udp_weight = calculate_weight(delta)
            
            # Check if TCP weight applies (single flip transition)
            tcp_applicable = is_single_flip_transition(source_matrix, target_matrix)
            
            if tcp_applicable:
                tcp_weight = calculate_tcp_weight()
                # Choose minimum weight and set action type
                if udp_weight <= tcp_weight:
                    final_weight = udp_weight
                    action_type = 1  # UDP action (b=1)
                else:
                    final_weight = tcp_weight
                    action_type = 0  # TCP action (b=0)
            else:
                # Only UDP weight applies
                final_weight = udp_weight
                action_type = 1  # UDP action (b=1)
            
            return final_weight, action_type
        weight_function = default_weight_function
    
    # Assign weights and action types to all edges
    for source, target, data in G.edges(data=True):
        result = weight_function(source, target, data)
        if isinstance(result, tuple):
            weight, action_type = result
            G[source][target]['weight'] = weight
            G[source][target]['action_type'] = action_type  # 0=TCP, 1=UDP
            
            # Update action_info based on action_type
            if action_type == 0:  # TCP action
                # Find the flipped position for TCP action
                source_matrix = current_states_dict[source][0]
                target_matrix = current_states_dict[target][0]
                flipped_pos = find_flipped_position(source_matrix, target_matrix)
                if flipped_pos:
                    msg_id, vehicle_id = flipped_pos
                    G[source][target]['action_info'] = {
                        'msg_id': str(msg_id),
                        'vehicle_id': str(vehicle_id)
                    }
                else:
                    # Fallback if position not found
                    G[source][target]['action_info'] = {
                        'msg_id': '?',
                        'vehicle_id': '?'
                    }
            # UDP action_info is already set during graph creation, keep it as is
                    
        else:
            # Backward compatibility
            G[source][target]['weight'] = result
            G[source][target]['action_type'] = 1  # Default to UDP
    
    return G

def create_weighted_graph():
    """
    Create the weighted graph with UDP transitions.
    
    Returns:
    NetworkX DiGraph - The complete weighted graph
    """
    # Generate states and initialize graph
    global states_dict  # Make it accessible to weight functions
    states = generate_states()
    states_dict = {matrix_to_index(m): (m, total) for m, total in states}
    G = nx.DiGraph()

    # Add nodes
    for idx, (matrix, total) in states_dict.items():
        label = matrix_to_str(matrix)
        G.add_node(idx, label=label, layer=total)

    # Add UDP transitions
    for idx, (matrix, total) in states_dict.items():
        transitions = valid_udp_transitions(matrix)
        for new_matrix, action in transitions:
            new_idx = matrix_to_index(new_matrix)
            if idx != new_idx:
                # Parse action info from action label (e.g., "udp#0" -> msg_id=0)
                action_info = {}
                if action.startswith("udp#"):
                    msg_id = action.split("#")[1]
                    action_info = {'msg_id': msg_id}
                elif action.startswith("tcp#"):
                    parts = action.split("#")
                    if len(parts) >= 3:
                        msg_id = parts[1] 
                        vehicle_id = parts[2]
                        action_info = {'msg_id': msg_id, 'vehicle_id': vehicle_id}
                
                G.add_edge(idx, new_idx, label=action, action_info=action_info)
    
    return G

def plot_weighted_graph(G, show_weights=True):
    """
    Plot the weighted graph with optional weight display.
    
    Parameters:
    G: NetworkX DiGraph - The graph to plot
    show_weights: bool - Whether to display edge weights
    """
    # Layered layout
    layers = {}
    for node, data in G.nodes(data=True):
        layer = data['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)

    pos = {}
    x_gap = 2
    y_gap = 1.5
    for i, layer in enumerate(sorted(layers.keys(), reverse=True)):
        nodes = layers[layer]
        for j, node in enumerate(nodes):
            pos[node] = (i * x_gap, -j * y_gap)

    # Plot
    plt.figure(figsize=(16, 10))
    
    # Draw nodes and basic edges
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw(
        G,
        pos,
        labels=node_labels,
        node_color='lightblue',
        node_size=2500,
        font_size=10,
        with_labels=True,
        connectionstyle='arc3,rad=0.2',
        arrows=True,
        edgecolors='black'
    )

    # Draw edge labels (action + weight if available)
    edge_labels = {}
    for source, target, data in G.edges(data=True):
        label = data.get('label', '')
        if show_weights and 'weight' in data:
            weight = data['weight']
            if weight == float('inf'):
                weight_str = "∞"
            else:
                weight_str = f"{weight:.2f}"
            label = f"{label}\nw={weight_str}"
        edge_labels[(source, target)] = label
    
    nx.draw_networkx_edge_labels(
        G, 
        pos, 
        edge_labels=edge_labels, 
        font_size=10, 
        label_pos=0.5,
        rotate=False,
        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8)
    )

    title = f"Weighted TCP/UDP Transitions Graph (p_udp={p_udp}, p_tcp={p_tcp}, {G.number_of_edges()} edges)"
    if show_weights:
        # Calculate sum of all weights
        total_weight = sum(data.get('weight', 0) for _, _, data in G.edges(data=True) if data.get('weight') != float('inf'))
        inf_count = sum(1 for _, _, data in G.edges(data=True) if data.get('weight') == float('inf'))
        if inf_count > 0:
            title += f" (sum weights: {total_weight:.2f} + {inf_count}∞)"
        else:
            title += f" (sum weights: {total_weight:.2f})"
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_edge_weights(G):
    """
    Get all edge weights as a dictionary.
    
    Parameters:
    G: NetworkX DiGraph - The weighted graph
    
    Returns:
    dict - Dictionary mapping (source, target) tuples to weights
    """
    return {(source, target): data.get('weight', None) 
            for source, target, data in G.edges(data=True)}

def update_weights(G, new_weight_function):
    """
    Update all edge weights using a new weight function.
    
    Parameters:
    G: NetworkX DiGraph - The graph to update
    new_weight_function: callable - New function to calculate weights
    
    Returns:
    NetworkX DiGraph - The graph with updated weights
    """
    return assign_edge_weights(G, None, new_weight_function)

def find_shortest_paths_to_destination(G):
    """
    Find shortest paths from all nodes to the destination (all-zero state).
    
    Parameters:
    G: NetworkX DiGraph - The weighted graph
    
    Returns:
    dict - Dictionary with shortest distances and paths to destination
    """
    # Find the all-zero state (destination)
    destination = None
    for node, (matrix, _) in states_dict.items():
        if np.sum(matrix) == 0:  # All zeros
            destination = node
            break
    
    if destination is None:
        print("Warning: No all-zero state found!")
        return None
    
    # Calculate shortest paths using Dijkstra's algorithm
    try:
        # Use shortest_path_length and shortest_path with weight parameter
        distances = {}
        paths = {}
        
        for source in G.nodes():
            try:
                distance = nx.shortest_path_length(G, source, destination, weight='weight')
                path = nx.shortest_path(G, source, destination, weight='weight')
                distances[source] = distance
                paths[source] = path
            except nx.NetworkXNoPath:
                # No path from this source to destination
                continue
                
    except Exception as e:
        print(f"Warning: Error calculating shortest paths: {e}")
        distances = {}
        paths = {}
    
    return {
        'destination': destination,
        'distances': distances,
        'paths': paths
    }

def get_all_ones_state():
    """
    Find the all-ones state node index.
    
    Returns:
    int - Node index of the all-ones state, or None if not found
    """
    for node, (matrix, _) in states_dict.items():
        if np.sum(matrix) == matrix.size:  # All ones
            return node
    return None

def plot_weighted_graph_with_path(G, highlight_path=None, show_weights=True):
    """
    Plot the weighted graph with optional path highlighting.
    
    Parameters:
    G: NetworkX DiGraph - The graph to plot
    highlight_path: list - List of nodes representing the path to highlight
    show_weights: bool - Whether to display edge weights
    """
    # Layered layout
    layers = {}
    for node, data in G.nodes(data=True):
        layer = data['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)

    pos = {}
    x_gap = 2
    y_gap = 1.5
    for i, layer in enumerate(sorted(layers.keys(), reverse=True)):
        nodes = layers[layer]
        for j, node in enumerate(nodes):
            pos[node] = (i * x_gap, -j * y_gap)

    # Plot
    plt.figure(figsize=(16, 10))
    
    # Draw all edges first with TCP/UDP coloring
    edge_colors = []
    edge_widths = []
    
    for source, target, data in G.edges(data=True):
        if highlight_path and len(highlight_path) > 1:
            # Check if this edge is part of the highlighted path
            is_highlighted = False
            for i in range(len(highlight_path) - 1):
                if source == highlight_path[i] and target == highlight_path[i + 1]:
                    is_highlighted = True
                    break
            
            if is_highlighted:
                # Color based on action type: blue for TCP (0), yellow for UDP (1)
                action_type = data.get('action_type', 1)  # Default to UDP
                if action_type == 0:
                    edge_colors.append('blue')  # TCP action
                else:
                    edge_colors.append('yellow')  # UDP action
                edge_widths.append(4)
            else:
                edge_colors.append('gray')
                edge_widths.append(1)
        else:
            edge_colors.append('gray')
            edge_widths.append(1)
    
    # Draw edges with colors and widths
    nx.draw_networkx_edges(
        G, pos, 
        edge_color=edge_colors,
        width=edge_widths,
        connectionstyle='arc3,rad=0.2',
        arrows=True,
        alpha=0.8
    )
    
    # Draw nodes with special colors for start, end, and path nodes
    node_colors = []
    node_sizes = []
    
    all_ones = get_all_ones_state()
    destination = None
    for node, (matrix, _) in states_dict.items():
        if np.sum(matrix) == 0:
            destination = node
            break
    
    for node in G.nodes():
        if node == all_ones:
            node_colors.append('lightgreen')  # Start node
            node_sizes.append(3500)
        elif node == destination:
            node_colors.append('lightcoral')  # Destination node
            node_sizes.append(3500)
        elif highlight_path and node in highlight_path:
            node_colors.append('gold')  # Path nodes (highlighted)
            node_sizes.append(3000)
        else:
            node_colors.append('lightblue')  # Regular nodes
            node_sizes.append(2500)
    
    # Draw nodes
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors='black'
    )
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    # Draw edge labels (action + weight if available)
    edge_labels = {}
    for source, target, data in G.edges(data=True):
        label = data.get('label', '')
        if show_weights and 'weight' in data:
            weight = data['weight']
            if weight == float('inf'):
                weight_str = "∞"
            else:
                weight_str = f"{weight:.2f}"
            label = f"{label}\nw={weight_str}"
        edge_labels[(source, target)] = label
    
    nx.draw_networkx_edge_labels(
        G, 
        pos, 
        edge_labels=edge_labels, 
        font_size=9, 
        label_pos=0.5,
        rotate=False,
        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8)
    )

    title = f"Weighted TCP/UDP Transitions Graph (p_udp={p_udp}, p_tcp={p_tcp})"
    if highlight_path:
        path_length = len(highlight_path) - 1 if highlight_path else 0  # Number of edges in path
        title += f" (Shortest Path: {path_length} edges)"
    if show_weights:
        # Calculate sum of weights in the shortest path
        if highlight_path and len(highlight_path) > 1:
            path_weight_sum = 0
            path_inf_count = 0
            for i in range(len(highlight_path) - 1):
                source = highlight_path[i]
                target = highlight_path[i + 1]
                if G.has_edge(source, target):
                    weight = G[source][target].get('weight', 0)
                    if weight == float('inf'):
                        path_inf_count += 1
                    else:
                        path_weight_sum += weight
            
            if path_inf_count > 0:
                title += f" (path sum: {path_weight_sum:.2f} + {path_inf_count}∞)"
            else:
                title += f" (path sum: {path_weight_sum:.2f})"
        else:
            # Calculate sum of all weights if no path
            total_weight = sum(data.get('weight', 0) for _, _, data in G.edges(data=True) if data.get('weight') != float('inf'))
            inf_count = sum(1 for _, _, data in G.edges(data=True) if data.get('weight') == float('inf'))
            if inf_count > 0:
                title += f" (sum weights: {total_weight:.2f} + {inf_count}∞)"
            else:
                title += f" (sum weights: {total_weight:.2f})"
    plt.title(title, fontsize=12)
    
    # Add legend if path is highlighted
    if highlight_path:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='lightgreen', label='Start State (All 1s)'),
            Patch(facecolor='lightcoral', label='End State (All 0s)'),
            Patch(facecolor='gold', label='Path States'),
            Patch(facecolor='lightblue', label='Other States'),
            Line2D([0], [0], color='blue', linewidth=3, label='TCP Actions'),
            Line2D([0], [0], color='yellow', linewidth=3, label='UDP Actions')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_reachable_states(initial_state_matrix):
    """
    Get all states reachable from the initial state by flipping 1's to 0's.
    
    Parameters:
    initial_state_matrix: numpy array - The initial state matrix
    
    Returns:
    set - Set of reachable state indices
    """
    reachable = set()
    initial_idx = matrix_to_index(initial_state_matrix)
    reachable.add(initial_idx)
    
    # Use BFS to find all reachable states
    queue = [initial_state_matrix]
    visited = {tuple(initial_state_matrix.flatten())}
    
    while queue:
        current_matrix = queue.pop(0)
        current_idx = matrix_to_index(current_matrix)
        
        # Get all valid UDP transitions from current state
        transitions = valid_udp_transitions(current_matrix)
        for new_matrix, _ in transitions:
            new_tuple = tuple(new_matrix.flatten())
            if new_tuple not in visited:
                visited.add(new_tuple)
                new_idx = matrix_to_index(new_matrix)
                reachable.add(new_idx)
                queue.append(new_matrix)
    
    return reachable

def create_restricted_graph(initial_state_matrix):
    """
    Create a restricted graph containing only states reachable from the initial state.
    
    Parameters:
    initial_state_matrix: numpy array - The initial state matrix
    
    Returns:
    NetworkX DiGraph - The restricted weighted graph
    """
    # Get reachable states
    reachable_states = get_reachable_states(initial_state_matrix)
    
    # Create restricted states dictionary
    restricted_states_dict = {idx: states_dict[idx] for idx in reachable_states}
    
    # Create restricted graph
    G_restricted = nx.DiGraph()
    
    # Add nodes for reachable states only
    for idx, (matrix, total) in restricted_states_dict.items():
        label = matrix_to_str(matrix)
        G_restricted.add_node(idx, label=label, layer=total)
    
    # Add edges between reachable states
    for idx in reachable_states:
        matrix, _ = restricted_states_dict[idx]
        transitions = valid_udp_transitions(matrix)
        for new_matrix, action in transitions:
            new_idx = matrix_to_index(new_matrix)
            if new_idx in reachable_states and idx != new_idx:
                # Parse action info from action label (e.g., "udp#0" -> msg_id=0)
                action_info = {}
                if action.startswith("udp#"):
                    msg_id = action.split("#")[1]
                    action_info = {'msg_id': msg_id}
                elif action.startswith("tcp#"):
                    parts = action.split("#")
                    if len(parts) >= 3:
                        msg_id = parts[1] 
                        vehicle_id = parts[2]
                        action_info = {'msg_id': msg_id, 'vehicle_id': vehicle_id}
                
                G_restricted.add_edge(idx, new_idx, label=action, action_info=action_info)
    
    return G_restricted, restricted_states_dict

def interactive_graph_analysis():
    """
    Interactive graph analysis: show full graph, let user select initial state,
    then show restricted graph with shortest path analysis.
    """
    global states_dict
    
    # Create and show the full graph first
    print("=== FULL GRAPH ANALYSIS ===")
    full_graph = create_weighted_graph()
    full_weighted_graph = assign_edge_weights(full_graph, states_dict)
    
    print(f"Full graph contains {full_weighted_graph.number_of_nodes()} states and {full_weighted_graph.number_of_edges()} edges")
    print("Displaying full graph...")
    plot_weighted_graph(full_weighted_graph, show_weights=True)
    
    # Show all available states for user selection
    print("\n=== STATE SELECTION ===")
    print("Available states (node_id: matrix representation):")
    states_list = []
    for idx, (matrix, total) in states_dict.items():
        matrix_str = matrix_to_str(matrix).replace('\n', ' | ')
        print(f"State {idx}: {matrix_str} (sum={total})")
        states_list.append(idx)
    
    # Get user input for initial state selection
    while True:
        try:
            user_input = input(f"\nSelect initial state (enter node ID from 0 to {max(states_list)}): ")
            selected_state_idx = int(user_input)
            if selected_state_idx in states_list:
                break
            else:
                print(f"Invalid state ID. Please choose from {min(states_list)} to {max(states_list)}")
        except ValueError:
            print("Please enter a valid integer.")
    
    selected_matrix = states_dict[selected_state_idx][0]
    print(f"\nSelected initial state {selected_state_idx}:")
    print(matrix_to_str(selected_matrix))
    
    # Create restricted graph from selected initial state
    print("\n=== RESTRICTED GRAPH ANALYSIS ===")
    restricted_graph, restricted_states_dict = create_restricted_graph(selected_matrix)
    
    # Update states_dict temporarily for weight calculations
    original_states_dict = states_dict.copy()
    states_dict = restricted_states_dict
    
    try:
        # Assign weights to restricted graph
        restricted_weighted_graph = assign_edge_weights(restricted_graph, restricted_states_dict)
        
        print(f"Restricted graph contains {restricted_weighted_graph.number_of_nodes()} states and {restricted_weighted_graph.number_of_edges()} edges")
        print("States reachable from initial state:")
        for idx in sorted(restricted_states_dict.keys()):
            matrix, total = restricted_states_dict[idx]
            matrix_str = matrix_to_str(matrix).replace('\n', ' | ')
            print(f"  State {idx}: {matrix_str} (sum={total})")
        
        # Find shortest paths in restricted graph
        shortest_paths_info = find_shortest_paths_to_destination(restricted_weighted_graph)
        
        if shortest_paths_info:
            destination = shortest_paths_info['destination']
            distances = shortest_paths_info['distances']
            paths = shortest_paths_info['paths']
            
            # Check if destination is reachable from selected initial state
            if selected_state_idx in paths:
                shortest_path = paths[selected_state_idx]
                shortest_distance = distances[selected_state_idx]
                
                print(f"\nShortest path from selected state {selected_state_idx} to destination {destination}:")
                print(f"Path: {shortest_path}")
                print(f"Distance: {shortest_distance}")
                
                # Show detailed path analysis
                print(f"\nDetailed path analysis:")
                total_path_weight = 0
                for i in range(len(shortest_path) - 1):
                    source = shortest_path[i]
                    target = shortest_path[i + 1]
                    source_matrix = restricted_states_dict[source][0]
                    target_matrix = restricted_states_dict[target][0]
                    
                    edge_data = restricted_weighted_graph[source][target]
                    weight = edge_data.get('weight', 0)
                    action_type = edge_data.get('action_type', 1)
                    action_str = "TCP" if action_type == 0 else "UDP"
                    
                    delta = count_flips(source_matrix, target_matrix)
                    weight_str = "∞" if weight == float('inf') else f"{weight:.4f}"
                    
                    print(f"  Step {i+1}: State {source} -> {target} (δ={delta}, weight={weight_str}, action={action_str})")
                    
                    if weight != float('inf'):
                        total_path_weight += weight
                
                print(f"Total path weight: {total_path_weight:.4f}")
                
                # Plot the restricted graph with highlighted shortest path
                print("\nDisplaying restricted graph with shortest path highlighted...")
                plot_weighted_graph_with_path(restricted_weighted_graph, highlight_path=shortest_path, show_weights=True)
                
            else:
                print(f"\nNo path found from selected state {selected_state_idx} to all-zeros destination!")
                if destination in restricted_states_dict:
                    print("All-zeros state is reachable but no path found (this shouldn't happen)")
                else:
                    print("All-zeros state is not reachable from the selected initial state")
                
                # Plot without highlighting
                print("Displaying restricted graph without path highlighting...")
                plot_weighted_graph(restricted_weighted_graph, show_weights=True)
        else:
            print("No shortest path analysis available for restricted graph")
            plot_weighted_graph(restricted_weighted_graph, show_weights=True)
            
    finally:
        # Restore original states_dict
        states_dict = original_states_dict

def generate_solution_filename(n_messages, n_vehicles, p_tcp, p_udp):
    """
    Generate filename for solution based on parameters (standardized base format).
    
    Parameters:
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles
    p_tcp: float - TCP probability
    p_udp: float - UDP probability
    
    Returns:
    str - Filename for the solution
    """
    return f"m_{n_messages}_v_{n_vehicles}_TCP_{p_tcp}_UDP_{p_udp}.csv"

def ensure_brain_directory():
    """
    Ensure Brain/Graph directory exists.
    
    Returns:
    str - Path to Brain/Graph directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    brain_dir = os.path.join(current_dir, "Brain")
    graph_dir = os.path.join(brain_dir, "Graph")
    
    os.makedirs(graph_dir, exist_ok=True)
    return graph_dir

def solution_exists(n_messages, n_vehicles, p_tcp, p_udp):
    """
    Check if solution file already exists.
    
    Parameters:
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles
    p_tcp: float - TCP probability
    p_udp: float - UDP probability
    
    Returns:
    tuple: (bool, str) - (exists, filepath)
    """
    graph_dir = ensure_brain_directory()
    filename = generate_solution_filename(n_messages, n_vehicles, p_tcp, p_udp)
    filepath = os.path.join(graph_dir, filename)
    return os.path.exists(filepath), filepath

def solve_and_save_shortest_paths(n_messages, n_vehicles, p_tcp, p_udp):
    """
    Solve shortest path problem for given parameters and save to CSV.
    
    Parameters:
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles
    p_tcp: float - TCP probability
    p_udp: float - UDP probability
    
    Returns:
    str - Path to saved solution file
    """
    global matrix_shape
    
    # Check if solution already exists
    exists, filepath = solution_exists(n_messages, n_vehicles, p_tcp, p_udp)
    if exists:
        print(f"Solution already exists: {filepath}")
        return filepath
    
    print(f"Generating solution for m={n_messages}, v={n_vehicles}, p_tcp={p_tcp}, p_udp={p_udp}")
    
    # Store original parameters
    original_n_messages = globals().get('n_messages', 2)
    original_n_vehicles = globals().get('n_vehicles', 3)
    original_matrix_shape = globals().get('matrix_shape', (2, 3))
    original_p_tcp = globals().get('p_tcp', 0.95)
    original_p_udp = globals().get('p_udp', 0.9)
    
    try:
        # Update global parameters
        globals()['n_messages'] = n_messages
        globals()['n_vehicles'] = n_vehicles
        globals()['matrix_shape'] = (n_messages, n_vehicles)
        globals()['p_tcp'] = p_tcp
        globals()['p_udp'] = p_udp
        
        # Create and solve the graph
        print("Creating weighted graph...")
        graph = create_weighted_graph()
        weighted_graph = assign_edge_weights(graph, states_dict)
        
        print(f"Graph created with {weighted_graph.number_of_nodes()} nodes and {weighted_graph.number_of_edges()} edges")
        
        # Find shortest paths from all states to destination
        print("Finding shortest paths...")
        shortest_paths_info = find_shortest_paths_to_destination(weighted_graph)
        
        if not shortest_paths_info:
            raise ValueError("Could not find shortest paths")
        
        destination = shortest_paths_info['destination']
        distances = shortest_paths_info['distances']
        paths = shortest_paths_info['paths']
        
        # Prepare solution data
        solution_data = []
        
        for state_id in sorted(states_dict.keys()):
            matrix, total = states_dict[state_id]
            matrix_str = '|'.join(''.join(map(str, row)) for row in matrix)  # Standardized format
            
            if state_id in paths and len(paths[state_id]) > 1:
                # Get next state and action
                path = paths[state_id]
                next_state = path[1]  # Next state in shortest path
                
                # Get action information
                if weighted_graph.has_edge(state_id, next_state):
                    edge_data = weighted_graph[state_id][next_state]
                    action_type = edge_data.get('action_type', 1)
                    action_info = edge_data.get('action_info', {})
                    weight = edge_data.get('weight', 0)
                    
                    # Format action in standardized format
                    if action_type == 0:  # TCP
                        msg_id = action_info.get('msg_id', 0)
                        vehicle_id = action_info.get('vehicle_id', 0)
                        action_str = f"TCP_m_{msg_id}_{vehicle_id}"
                    else:  # UDP
                        msg_id = action_info.get('msg_id', 0)
                        action_str = f"UDP_m_{msg_id}"
                else:
                    action_str = "UNKNOWN"
                    weight = float('inf')
                
                distance = distances.get(state_id, float('inf'))
                
                solution_data.append({
                    'state_id': state_id,
                    'state_matrix': matrix_str,
                    'action': action_str,
                    'next_state': next_state,
                    'edge_weight': weight,
                    'total_distance': distance
                })
            else:
                # This is the destination or unreachable state
                if state_id == destination:
                    solution_data.append({
                        'state_id': state_id,
                        'state_matrix': matrix_str,
                        'action': "DESTINATION",
                        'next_state': state_id,  # Stay at destination
                        'edge_weight': 0,
                        'total_distance': 0
                    })
                else:
                    # Unreachable state
                    solution_data.append({
                        'state_id': state_id,
                        'state_matrix': matrix_str,
                        'action': "UNREACHABLE",
                        'next_state': -1,  # No next state
                        'edge_weight': float('inf'),
                        'total_distance': float('inf')
                    })
        
        # Save to CSV
        print(f"Saving solution to {filepath}")
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['state_id', 'state_matrix', 'action', 'next_state', 'edge_weight', 'total_distance']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write solution data
            for row in solution_data:
                writer.writerow(row)
        
        print(f"Solution saved successfully to {filepath}")
        print(f"Total states: {len(solution_data)}")
        print(f"Reachable states: {len([s for s in solution_data if s['action'] not in ['UNREACHABLE']])}")
        print(f"Destination state: {destination}")
        
        return filepath
        
    finally:
        # Restore original parameters
        globals()['n_messages'] = original_n_messages
        globals()['n_vehicles'] = original_n_vehicles
        globals()['matrix_shape'] = original_matrix_shape
        globals()['p_tcp'] = original_p_tcp
        globals()['p_udp'] = original_p_udp

def load_solution(n_messages, n_vehicles, p_tcp, p_udp):
    """
    Load solution from CSV file.
    
    Parameters:
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles
    p_tcp: float - TCP probability
    p_udp: float - UDP probability
    
    Returns:
    dict - Solution data with state_id as keys
    """
    exists, filepath = solution_exists(n_messages, n_vehicles, p_tcp, p_udp)
    if not exists:
        return None
    
    solution = {}
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            state_id = int(row['state_id'])
            solution[state_id] = {
                'state_matrix': row['state_matrix'],
                'action': row['action'],
                'next_state': int(row['next_state']) if row['next_state'] != '-1' else -1,
                'edge_weight': float(row['edge_weight']) if row['edge_weight'] != 'inf' else float('inf'),
                'total_distance': float(row['total_distance']) if row['total_distance'] != 'inf' else float('inf')
            }
    
    return solution

def get_or_generate_solution(n_messages, n_vehicles, p_tcp, p_udp):
    """
    Get solution from file if exists, otherwise generate and save it.
    
    Parameters:
    n_messages: int - Number of messages
    n_vehicles: int - Number of vehicles
    p_tcp: float - TCP probability
    p_udp: float - UDP probability
    
    Returns:
    dict - Solution data with state_id as keys
    """
    # First try to load existing solution
    solution = load_solution(n_messages, n_vehicles, p_tcp, p_udp)
    if solution is not None:
        print(f"Loaded existing solution for m={n_messages}, v={n_vehicles}, p_tcp={p_tcp}, p_udp={p_udp}")
        return solution
    
    # Generate new solution if not found
    print(f"No existing solution found, generating new one...")
    solve_and_save_shortest_paths(n_messages, n_vehicles, p_tcp, p_udp)
    
    # Load the newly generated solution
    solution = load_solution(n_messages, n_vehicles, p_tcp, p_udp)
    if solution is None:
        raise ValueError("Failed to generate or load solution")
    
    return solution

if __name__ == "__main__":
    # Example usage: generate solution for default parameters
    print("Graph solution module - generating example solution...")
    solution = get_or_generate_solution(n_messages, n_vehicles, p_tcp, p_udp)
    print(f"Solution generated/loaded with {len(solution)} states")
