import numpy as np
from mdptoolbox import mdp
import itertools
import json
from collections import defaultdict
from itertools import product
from scipy.sparse import csr_matrix
import pandas as pd

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

num_messages = 3
num_clients = 2
num_states = 2 ** (num_messages * num_clients)

num_actions_TCP = num_messages * num_clients
num_actions_UDP = num_messages
num_actions = num_actions_TCP + num_actions_UDP

P = [csr_matrix((num_states, num_states)) for _ in range(num_actions)]
R = [csr_matrix((num_states, num_states)) for _ in range(num_actions)]

#P = np.zeros((num_actions, num_states, num_states))
#R = np.zeros((num_states, num_actions))

ind_TCP = np.arange(num_actions_TCP)
ind_UDP = np.arange(num_actions_TCP, num_actions)

discount_factor = 0.9
tcp_success = 0.8
udp_success = 0.6

nodes = []
edge_dict = defaultdict(set)

for current_state_ind in range(num_states):
    current_state =  index_to_matrix(current_state_ind, (num_messages, num_clients))
    nodes.append({"id": f"s{current_state_ind}", "matrix": current_state.tolist()})     ###################

    for action in range(num_actions):

        if action < num_actions_TCP:    # TCP action
            is_tcp = 1
            msg_id, vehicle_id = get_message_vehicle_ids(action, num_clients)
            p_msg_success = tcp_success

        else:                           # UDP action
            is_tcp = 0
            msg_id = action - num_actions_TCP
            vehicle_id = -1
            p_msg_success = udp_success

        next_states = get_all_reachable_states(current_state, msg_id, is_tcp, vehicle_id)

        for next_state in next_states:  # all possible states to reach, including the current state
            next_state_ind = matrix_to_index(next_state)
            num_received_msg, transition_p = transition_prob(current_state, next_state, msg_id, p_msg_success, is_tcp, vehicle_id)
            P[action][current_state_ind, next_state_ind] = transition_p
            R[action][current_state_ind, next_state_ind] = num_received_msg
"""
links = []
for (src, tgt), labells in edge_dict.items():
    links.append({
        "source": f"s{src}",
        "target": f"s{tgt}",
        "label": ", ".join(sorted(labels))
    })

d3_graph = {
    "nodes": nodes,
    "links": links
}

# Extract node and link data from the d3_graph variable
nodes_df = pd.DataFrame(d3_graph["nodes"])
links_df = pd.DataFrame(d3_graph["links"])
print(links_df)

# Save to CSV files
nodes_path = "/mdp_nodes.csv"
links_path = "/mdp_edges.csv"
nodes_path, links_path
nodes_df.to_csv(nodes_path, index=False)
links_df.to_csv(links_path, index=False)


"""


mdp_solver = mdp.ValueIteration(P, R, discount_factor)
mdp_solver.run()

print("Optimal Policy:")
for i in range(num_states):
    state_matrix = index_to_matrix(i, (num_messages, num_clients))
    action = mdp_solver.policy[i]
    if action < num_actions_TCP:
        r, c = divmod(action, num_clients)
        description = f"TCP: msg {r} to client {c}"
    else:
        r = action - num_actions_TCP
        description = f"UDP: broadcast msg {r}"
    print(f"State {i}:\n{state_matrix} -> {description}")
    print("_______________________________")
