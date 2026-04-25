import networkx as nx
import numpy as np
import scipy
from collections import OrderedDict
def calculate_node_coefficient(G, trajectory):
    """
    Calculate the coefficient for each node in the trajectory graph.

    Parameters:
    G (networkx.DiGraph): The directed graph representing the state space.
    trajectory (tuple): The trajectory for which the coefficient is calculated.
    Returns:
    float: coefficient of the node.
    """

    coefficient = 1.0
    trajectory = list(trajectory)
    for i in range(len(trajectory)-1):
        edge_weight = G[trajectory[i]][trajectory[i+1]]['weight']
        coefficient *= edge_weight
    coefficient *= G[trajectory[-1]][trajectory[0]]['weight']
    return coefficient
    

def is_valid_trajectory(trajectory):
    """
    Checks if a given trajectory is valid based on predefined criteria.

    Parameters:
    trajectory (tuple): The trajectory to be validated.

    Returns:
    bool: True if the trajectory is valid, False otherwise.
    """
    trajectory = list(trajectory)
    if len(trajectory) != len(set(trajectory)):
        return False
    return True

def construct_trajectory_graph(G, trajectory=None, trajectory_graph = None):
    """
    Constructs a trajectory graph from the boundary analytics results.
    Keep in mind that weights are log(weights) to accurately calculate F_{i,j}.

    Parameters:
    G (networkx.DiGraph): The directed graph representing the state space.

    Returns:
    networkx.DiGraph: The constructed trajectory graph.
    """

    if trajectory_graph is None:
        trajectory_graph = nx.DiGraph()
    if trajectory is None:
        trajectory = (0)
        trajectory_graph.add_node(trajectory)
    neighbors = [i for i in G.neighbors(trajectory[-1])]
    for i in neighbors:
        test_trajectory = tuple(list(trajectory) + [i])
        if is_valid_trajectory(test_trajectory):
            trajectory_graph.add_edge(trajectory, test_trajectory, weight=G[trajectory[-1]][test_trajectory[-1]]['weight'])
            trajectory_graph = construct_trajectory_graph(G, test_trajectory, trajectory_graph)
    return trajectory_graph

def get_cycle_weight(G, cycle):
    """
    Calculates the weight of a given cycle in the trajectory graph.

    Parameters:
    G (networkx.DiGraph): The directed graph representing the trajectory.
    cycle (list): A list of nodes representing the cycle.

    Returns:
    float: The weight of the cycle.
    """
    trajectory_graph = construct_trajectory_graph(G)
    node_coefficient = calculate_node_coefficient(G, cycle)

    trajectory_nodes = list(trajectory_graph.nodes)
    pi_cycle = 0
    for node in trajectory_nodes:
        if cycle_in_node_trajectory(cycle, node):
            pi_cycle += MTT(trajectory_graph, trajectory_graph[node][cycle[0]])
    return pi_cycle * node_coefficient

def cycle_in_node_trajectory(cycle, node_trajectory):
    """
    Checks if a given cycle is part of a node's trajectory.

    Parameters:
    cycle (list): A list of nodes representing the cycle.
    node_trajectory (tuple): The trajectory associated with the node.

    Returns:
    bool: True if the cycle is part of the node's trajectory, False otherwise.
    """
    cycle = list(OrderedDict.fromkeys(cycle))
    node_trajectory = list(node_trajectory)
    cycle_length = len(cycle)
    for i in range(len(node_trajectory)):
        match = True
        for j in range(cycle_length):
            if node_trajectory[(i + j) % len(node_trajectory)] != cycle[j]:
                match = False
                break
        if match:
            return True
    return False

def MTT(trajectory_graph, root_node):
    root_node_index = list(trajectory_graph.nodes).index(root_node)
    laplacian = np.transpose(nx.laplacian_matrix(trajectory_graph.reverse(copy=False)))
    laplacian_reduced = np.delete(np.delete(laplacian.toarray(), root_node_index, axis=0), root_node_index, axis=1)
    num = np.linalg.det(laplacian_reduced)
    denom = 0.0
    for i in range(laplacian_reduced.shape[0]):
        laplacian_reduced = np.delete(np.delete(laplacian.toarray(), root_node_index, axis=0), root_node_index, axis=1)
        denom += np.linalg.det(np.delete(np.delete(laplacian_reduced, i, axis=0), i, axis=1))
    return num / denom

# def get_ingroup_nodes(G, start_node, end_node):
#     """
#     Retrieves nodes bordering a single edge perturbation in the state space.

#     Parameters:
#     G (networkx.DiGraph): The directed graph representing the trajectory.
#     start_node (str): The identifier for the start node of the edge perturbation.
#     end_node (str): The identifier for the end node of the edge perturbation.

#     Returns:
#     list: A list of nodes belonging to the specified ingroup.
#     """
#     pass

# def get_max_path_length(G):
#     """
#     Determines the maximum path length in the trajectory graph.

#     Parameters:
#     G (networkx.DiGraph): The directed graph representing the trajectory.

#     Returns:
#     int: The maximum path length found in the graph.
#     """
#     pass

# def cycle_discrimination_index(G):
#     """
#     Computes the cycle discrimination index for the trajectory graph.

#     Parameters:
#     G (networkx.DiGraph): The directed graph representing state space.

#     Returns:
#     float: The cycle discrimination index.
#     """
#     pass