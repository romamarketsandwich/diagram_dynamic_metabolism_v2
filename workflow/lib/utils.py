
import mpmath
import networkx as nx


def max_rate_six_states(alpha_32, alpha_21, alpha_54, alpha_65,  alpha_14, alpha_41, dG_1, dG_2, gamma, epsilon_1, epsilon_2, k_bT, **params):
    '''
    Get the maximum rate at steady-state under the Sabatier principle. Expression derived in Mathematica.
    Args:
        alpha_32 (float): rate constant for the transition from state 3 to state 2.
        alpha_21 (float): rate constant for the transition from state 2 to state 1.
        alpha_54 (float): rate constant for the transition from state 5 to state 4.
        alpha_65 (float): rate constant for the transition from state 6 to state 5.
        alpha_14 (float): rate constant for the transition from state 1 to state 4.
        alpha_41 (float): rate constant for the transition from state 4 to state 1.
        dG_1 (float): free energy difference between state 1 and state 2.
        dG_2 (float): free energy difference between state 2 and state 3.
        gamma (float): rate constant for the transition from state 2 to state 5.
        epsilon_1 (float): energy offset for node layer 1.
        epsilon_2 (float): energy offset for node layer 2.
        k_bT (float): thermal energy (Boltzmann constant times temperature).
    Returns:
        float: maximum rate at steady-state under the Sabatier principle.
    '''
    return gamma * (alpha_32 * mpmath.exp((dG_1 + dG_2)/k_bT) * mpmath.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))) \
    / (gamma * (2 * mpmath.exp((dG_1 + dG_2) / (2 * k_bT)) + mpmath.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))) \
    + alpha_32 * (2 * mpmath.exp((dG_1 + dG_2) / (2 * k_bT)) + mpmath.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma)) \
    + mpmath.exp((dG_1 + dG_2) / k_bT) * mpmath.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))))

def osc_midpoint_six_states(alpha_32, alpha_21, alpha_54, alpha_65, alpha_14, alpha_41, dG_1, dG_2, gamma, epsilon_1, epsilon_2, k_bT, **params):
    '''
    Get the oscillation midpoint for the six-state model.
    Args:
        alpha_32 (float): rate constant for the transition from state 3 to state 2.
        alpha_21 (float): rate constant for the transition from state 2 to state 1.
        alpha_54 (float): rate constant for the transition from state 5 to state 4.
        alpha_65 (float): rate constant for the transition from state 6 to state 5.
        alpha_14 (float): rate constant for the transition from state 1 to state 4.
        alpha_41 (float): rate constant for the transition from state 4 to state 1.
        dG_1 (float): free energy difference between state 1 and state 2.
        dG_2 (float): free energy difference between state 2 and state 3.
        gamma (float): rate constant for the transition from state 2 to state 5.
        epsilon_1 (float): energy offset for node layer 1.
        epsilon_2 (float): energy offset for node layer 2.
        k_bT (float): thermal energy (Boltzmann constant times temperature).
    Returns:
        float: oscillation midpoint for the six-state model.
    '''
    return -1/2 * (-dG_1 + dG_2 - k_bT * mpmath.log((alpha_21 * alpha_32 + alpha_21 * gamma) / (alpha_32 * gamma)))




def map_graph_weights_six_states(alpha_32, alpha_21, alpha_54, alpha_65, \
                                 alpha_14, alpha_41, dG_1, dG_2, gamma, epsilon_1, epsilon_2, k_bT, **params):
    '''
    Map the weights of the graph for the six-state model.
    Args:
        alpha_32 (float): rate constant for the transition from state 3 to state 2.
        alpha_21 (float): rate constant for the transition from state 2 to state 1.
        alpha_54 (float): rate constant for the transition from state 5 to state 4.
        alpha_65 (float): rate constant for the transition from state 6 to state 5.
        alpha_14 (float): rate constant for the transition from state 1 to state 4.
        alpha_41 (float): rate constant for the transition from state 4 to state 1.
        dG_1 (float): free energy difference between state 1 and state 2.
        dG_2 (float): free energy difference between state 2 and state 3.
        gamma (float): rate constant for the transition from state 2 to state 5.
        epsilon_1 (float): energy offset for node layer 1.
        epsilon_2 (float): energy offset for node layer 2.
        k_bT (float): thermal energy (Boltzmann constant times temperature).
    Returns:
        dict: dictionary mapping edges to their weights for the six-state model.
    '''
    alpha_12 = alpha_21 * mpmath.exp((dG_1 + epsilon_1) / k_bT)
    alpha_23 = alpha_32 * mpmath.exp((dG_2 - epsilon_1)/ k_bT)


    alpha_45 = alpha_54 * mpmath.exp((dG_1 + epsilon_2) / k_bT)
    alpha_56 = alpha_65 * mpmath.exp((dG_2 - epsilon_2) / k_bT)



    alpha_25 = alpha_14
    alpha_52 = alpha_41
    alpha_36 = alpha_14
    alpha_63 = alpha_41
    d = {        
        (0, 1) : alpha_12,
        (1, 0) : alpha_21,
        (1, 2) : alpha_23,
        (2, 1) : alpha_32,
        (3, 4) : alpha_45,
        (4, 3) : alpha_54,
        (4, 5) : alpha_56,
        (5, 4) : alpha_65,
        (0, 3) : alpha_14,
        (3, 0) : alpha_41,
        (1, 4) : alpha_25,
        (4, 1) : alpha_52,
        (2, 5) : alpha_36,
        (5, 2) : alpha_63,
        (2, 0) : gamma,
        (5, 3) : gamma,
    }

    return d

def map_graph_weights_eight_states(alpha_32, alpha_21, alpha_54, alpha_65, alpha_43, alpha_76, alpha_87,
                                 alpha_14, alpha_41, dG_1, dG_2, dG_3, gamma, epsilon_1, epsilon_2, k_bT, **params):
    '''
    Map the weights of the graph for the eight-state model.
    Args:
        alpha_32 (float): rate constant for the transition from state 3 to state 2.
        alpha_21 (float): rate constant for the transition from state 2 to state 1.
        alpha_54 (float): rate constant for the transition from state 5 to state 4.
        alpha_65 (float): rate constant for the transition from state 6 to state 5.
        alpha_43 (float): rate constant for the transition from state 4 to state 3.
        alpha_76 (float): rate constant for the transition from state 7 to state 6.
        alpha_87 (float): rate constant for the transition from state 8 to state 7.
        alpha_14 (float): rate constant for the transition from state 1 to state 4.
        alpha_41 (float): rate constant for the transition from state 4 to state 1.
        dG_1 (float): free energy difference between state 1 and state 2.
        dG_2 (float): free energy difference between state 2 and state 3.
        dG_3 (float): free energy difference between state 3 and state 4.
        gamma (float): rate constant for the transition from state 2 to state 5.
        epsilon_1 (float): energy offset for node layer 1.
        epsilon_2 (float): energy offset for node layer 2.
        k_bT (float): thermal energy (Boltzmann constant times temperature).
    Returns:
        dict: dictionary mapping edges to their weights for the eight-state model.
    '''
    alpha_12 = alpha_21 * np.exp((dG_1 + epsilon_1) / k_bT)
    alpha_23 = alpha_32 * np.exp((dG_2)/ k_bT)
    alpha_34 = alpha_43 * np.exp((dG_3 - epsilon_1)/ k_bT)


    alpha_56 = alpha_65 * np.exp((dG_1 + epsilon_2) / k_bT)
    alpha_67 = alpha_76 * np.exp((dG_2) / k_bT)
    alpha_78 = alpha_87 * np.exp((dG_2 - epsilon_2) / k_bT)

    alpha_15 = alpha_14
    alpha_51 = alpha_41
    alpha_26 = alpha_14
    alpha_62 = alpha_41
    alpha_37 = alpha_14
    alpha_73 = alpha_41
    alpha_48 = alpha_14
    alpha_84 = alpha_41

    d = {        
        (0, 1) : alpha_12,
        (1, 0) : alpha_21,
        (1, 2) : alpha_23,
        (2, 1) : alpha_32,
        (2, 3) : alpha_34,
        (3, 2) : alpha_43,
        (4, 5) : alpha_56,
        (5, 4) : alpha_65,
        (5, 6) : alpha_67,
        (6, 5) : alpha_76,
        (6, 7) : alpha_78,
        (7, 6) : alpha_87,
        (0, 4) : alpha_15,
        (4, 0) : alpha_51,
        (1, 5) : alpha_26,
        (5, 1) : alpha_62,
        (6, 2) : alpha_73,
        (2, 6) : alpha_37,
        (3, 7) : alpha_48,
        (7, 3) : alpha_84,
        (3, 0) : gamma,
        (7, 4) : gamma,
    }
    return d



def map_graph_weights_ten_states(alpha_32, alpha_21, alpha_54, alpha_65, alpha_32_w, alpha_21_w, alpha_54_w, alpha_65_w,\
                                    alpha_14, alpha_41, dG_1, dG_2, dG_1_w, dG_2_w, gamma, epsilon_1, epsilon_2, k_bT, stochastic_resetting_rate=0, **params):
    '''
    Map the weights of the graph for the ten-state model.
    Args:
        alpha_32 (float): rate constant for the transition from state 3 to state 2.
        alpha_21 (float): rate constant for the transition from state 2 to state
        alpha_54 (float): rate constant for the transition from state 5 to state 4.
        alpha_65 (float): rate constant for the transition from state 6 to state
        alpha_32_w (float): rate constant for the transition from state 3 to state 2 in the "w" layer.
        alpha_21_w (float): rate constant for the transition from state 2 to state
        alpha_54_w (float): rate constant for the transition from state 5 to state 4 in the "w" layer.
        alpha_65_w (float): rate constant for the transition from state 6 to state
        alpha_14 (float): rate constant for the transition from state 1 to state
        alpha_41 (float): rate constant for the transition from state 4 to state 1.
        dG_1 (float): free energy difference between state 1 and state 2
        dG_2 (float): free energy difference between state 2 and state 3.
        dG_1_w (float): free energy difference between state 1 and state
        dG_2_w (float): free energy difference between state 2 and state 3 in the "w" layer.
        gamma (float): rate constant for the transition from state 2 to state 5.
        epsilon_1 (float): energy offset for node layer 1.
        epsilon_2 (float): energy offset for node layer 2.
        k_bT (float): thermal energy (Boltzmann constant times temperature).
    Returns:
        dict: dictionary mapping edges to their weights for the ten-state model.
    '''
    alpha_12 = alpha_21 * np.exp((dG_1 + epsilon_1) / k_bT)
    alpha_23 = alpha_32 * np.exp((dG_2 - epsilon_1)/ k_bT)
    alpha_45 = alpha_54 * np.exp((dG_1 + epsilon_2) / k_bT)
    alpha_56 = alpha_65 * np.exp((dG_2 - epsilon_2) / k_bT)
    alpha_12_w = alpha_21_w * np.exp((dG_1_w + epsilon_1) / k_bT)
    alpha_23_w = alpha_32_w * np.exp((dG_2_w - epsilon_1)/ k_bT)
    alpha_45_w = alpha_54_w * np.exp((dG_1_w + epsilon_2) / k_bT)
    alpha_56_w = alpha_65_w * np.exp((dG_2_w - epsilon_2) / k_bT)

    alpha_25 = alpha_14
    alpha_52 = alpha_41
    alpha_36 = alpha_14
    alpha_63 = alpha_41
    alpha_25_w = alpha_14
    alpha_52_w = alpha_41
    alpha_36_w = alpha_14
    alpha_63_w = alpha_41

    d = {
        (0, 1) : alpha_32_w,
        (1, 0) : alpha_23_w,
        (1, 2) : alpha_21_w + stochastic_resetting_rate,
        (2, 1) : alpha_12_w,
        (2, 3) : alpha_12,
        (3, 2) : alpha_21 + stochastic_resetting_rate,
        (3, 4) : alpha_23,
        (4, 3) : alpha_32,
        (5, 6) : alpha_65_w,
        (6, 5) : alpha_56_w,
        (6, 7) : alpha_54_w + stochastic_resetting_rate,
        (7, 6) : alpha_45_w,
        (7, 8) : alpha_45,
        (8, 7) : alpha_54 + stochastic_resetting_rate,
        (8, 9) : alpha_56,
        (9, 8) : alpha_65,
        (0, 5) : alpha_36_w,
        (5, 0) : alpha_63_w,
        (1, 6) : alpha_25_w,
        (6, 1) : alpha_52_w,
        (7, 2) : alpha_41,
        (2, 7) : alpha_14,
        (3, 8) : alpha_25,
        (8, 3) : alpha_52,
        (4, 9) : alpha_36,
        (9, 4) : alpha_63,
        (0, 2) : gamma,
        (4, 2) : gamma,
        (5, 7) : gamma,
        (9, 7) : gamma,
    }
    return d



def generate_eight_state_graph():
    '''
    Generate an eight-state graph with nodes 0-7 and edges connecting them in a specific pattern.
    Returns:
        nx.Graph: An undirected graph representing the eight-state model.
    '''
    G = nx.Graph()
    G.add_nodes_from([3, 2, 1, 0, 4, 5, 6, 7])
    G.add_edges_from([(0, 1), (1, 2), (2,3), (0,3), (4,5), (5,6), (6,7), (4,7), (0,4), (1,5), (2,6), (3,7)])
    return G

def generate_ten_state_graph():
    '''
    Generate a ten-state graph with nodes 0-9 and edges connecting them in a specific pattern.
    Returns:
        nx.Graph: An undirected graph representing the ten-state model.
    '''
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    G.add_edges_from([(0, 1), (1, 2), (0,2), (2, 3), (3, 4), (4,2), (5, 6), (6, 7), (5, 7), (7, 8), (8, 9), (9, 7), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)])
    return G

def generate_six_state_graph():
    '''
    Generate a six-state graph with nodes 0-5 and edges connecting them in a specific pattern.
    Returns:
        nx.Graph: An undirected graph representing the six-state model.
    '''
    G = nx.Graph()
    G.add_nodes_from([2, 1, 0, 3, 4, 5])
    G.add_edges_from([(0, 1), (1, 2), (0,2), (3,4), (4,5), (3,5), (0,3), (1,4), (2,5)])
    return G

def generate_eight_state_fundamental_cycles():
    '''
    Generate the fundamental cycles for an eight-state graph.
    Returns:
        tuple: A tuple containing a list of fundamental cycles and a list of fundamental edges.
    '''
    t1 = [0,1,2,3]
    t2 = [4,5,6,7]
    t3 = [0,4,5,6,7,3]
    t4 = [4,0,1,2,3,7]
    t5 = [4,0,1,5,6,7]
    t6 = [0,4,5,1,2,3]
    t7 = [0,4,5,6,2,3]
    t8 = [4,0,1,2,6,7]
    t9 = [0,1,5,6,2,3]
    t10 = [4,5,1,2,6,7]
    t11 = [0,4,5,1,2,6,7,3]
    t12 = [4,0,1,5,6,2,3,7]
    t13 = [0,1,5,6,7,3]
    t14 = [4,5,1,2,3,7]
    t15 = [4,5,6,2,3,7]
    t16 = [0,1,2,6,7,3]

    tf1 = [0,4]
    tf2 = [1,5]
    tf3 = [2,6]
    tf4 = [3,7]

    return [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16], [tf1,tf2,tf3,tf4]
