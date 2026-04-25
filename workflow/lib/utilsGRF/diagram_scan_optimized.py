import numpy as np
from itertools import product
from multiprocessing import Pool
import pickle
import mpmath
import networkx as nx

def spanTrees(trs, Edg, all_span_trees, k):
    if k == 0:
        all_span_trees.extend(product(*trs))#Expand parallels edges using the Cartesian product
    
    for i in range(k):
        if Edg[k][i] == []: continue
        trs.append(Edg[k][i])
        Edg[i] = [Edg[i][j] + Edg[k][j] for j in range(i)]#Contraction 
        spanTrees(trs, Edg, all_span_trees, k-1)
        trs.pop()
        [Edg[i][j].pop() for j in range(i) for _ in range(len(Edg[k][j]))]#Deletion

#Helper function
def Spanning_Trees_Generator(g, hch = False):
    if not nx.is_connected(g):
        print("Only for connected graphs")
        return 0
    if nx.is_directed(g): nx.to_undirected(g)
    n = g.number_of_nodes()
    edgs = list(g.edges)
    Edg = [[[] for _ in range(n)] for _ in range(n)]
    mx = len(edgs)
    edgNum = dict() #dictionary designed to store labeled edges where the keys are integer labels for the edges, and the values are tuples representing the ordinary edge definitions (in, out)

    for ed in edgs:
        i, j = sorted(ed)
        Edg[j][i] = [mx]
        edgNum[mx] = ed
        mx -= 1
    all_span_trees = []
    spanTrees([], Edg, all_span_trees, n-1)
  
    if hch:  
        return all_span_trees #Generate only the labeled edges (Keys) 
    else:
        return [nx.Graph(edgNum[k] for k in element) for element in all_span_trees] #Generate spanning trees as NetworkX graphics objects
    
def root_tree_weights(spanning_tree, weights, nodes=None):
    #assumes at least one valid root in the spanning tree implementing one-way fluxes. 
    if nodes is None:
        nodes = [n for n in spanning_tree]
    
    """ 
    Key insight: if the tree contains a one-way edge, then nothing "upstream" of the one-way edge can be a root.
    Also, all flows "upstream" of one-way edge must flow through the one-way edge.
    
    Else, there would exist at least one edge in spanning_tree that would connect "head" and "tail" subgraphs,
    suggesting a cycle. This is a contradiction.

    Therefore, we can recursively treat tree weight calculation by:
        1. removing one-way edges, if possible
        2. calculating "tail" tree weights, assuming root at one-way edge "tail".
        3. tree weight rooted at node i in "head" = "tail" tree weight * one-way edge weight * "head" tree weight .
        4. Sum all tree weights over all i in the "head".
    """
    spanning_tree = spanning_tree.copy()
    prefactor = 1
    if ((3,5) in spanning_tree.edges or (5,3) in spanning_tree.edges) and ((0,2) in spanning_tree.edges or (2,0) in spanning_tree.edges):
        spanning_tree.remove_edges_from([(3,5), (5,3), (0,2), (2,0)])
        nodes1 = nx.node_connected_component(spanning_tree, 3)
        nodes2 = nx.node_connected_component(spanning_tree, 0)
        if len(nodes1) > len(nodes2):
            nodes = nodes2
        else:
            nodes = nodes1
        prefactor = prefactor * root_tree_weights(spanning_tree, weights, nodes=[5]) * weights.get((5, 3)) * root_tree_weights(spanning_tree, weights, nodes=[2]) * weights.get((2, 0))
    elif (3,5) in spanning_tree.edges or (5,3) in spanning_tree.edges:
        spanning_tree.remove_edges_from([(3,5), (5,3)])
        nodes = nx.node_connected_component(spanning_tree, 3)
        prefactor = prefactor * root_tree_weights(spanning_tree, weights, nodes=[5]) * weights.get((5, 3))
    elif (0,2) in spanning_tree.edges or (2,0) in spanning_tree.edges:
        spanning_tree.remove_edges_from([(0,2), (2,0)])
        nodes = nx.node_connected_component(spanning_tree, 0)
        prefactor = prefactor * root_tree_weights(spanning_tree, weights, nodes=[2]) * weights.get((2, 0))
    total_weight = 0
    for root in sorted(nodes):
        # Perform BFS to get the tree structure rooted at 'root'
        BFS_list = []
        visited = set()
        queue = [root]
        while queue:
            current = queue.pop(0)
            visited.add(current)
            for neighbor in spanning_tree.neighbors(current):
                if neighbor not in visited and (current, neighbor) in spanning_tree.edges:
                    BFS_list.append((neighbor, current))  # Store as (child, parent)
                    queue.append(neighbor)
        # Calculate the weight of the rooted tree
        tree_weight = np.prod([weights.get((u, v)) for u, v in BFS_list]) * prefactor
        # Store or process the tree weight as needed
        total_weight += tree_weight

    return total_weight 

def map_graph_weights(alpha_32=1, alpha_21=1, alpha_65=1, alpha_54=1, alpha_14=1, alpha_41=1, dG_1=2, dG_2=2, gamma=10, epsilon_1=0, epsilon_2=0, k_bT=1):

    alpha_32, alpha_21, alpha_65, alpha_54, alpha_14, alpha_41 = mpmath.mpf(alpha_32), mpmath.mpf(alpha_21), mpmath.mpf(alpha_65), mpmath.mpf(alpha_54), mpmath.mpf(alpha_14), mpmath.mpf(alpha_41)
    dG_1, dG_2, gamma, epsilon_1, epsilon_2, k_bT = mpmath.mpf(dG_1), mpmath.mpf(dG_2), mpmath.mpf(gamma), mpmath.mpf(epsilon_1), mpmath.mpf(epsilon_2), mpmath.mpf(k_bT)
    alpha_12 = alpha_21 * mpmath.exp((dG_1 + epsilon_1) / k_bT)
    alpha_23 = alpha_32 * mpmath.exp((dG_2 - epsilon_1)/ k_bT)

    alpha_54 = alpha_21
    alpha_65 = alpha_32
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

def dynamic_steady_state_diagram(epsilon_1, epsilon_2, alpha_21, alpha_32, gamma_1, gamma_2, alpha_14, alpha_41, dG_1=2, dG_2=2, k_bT=1, spanTreesGraph= None):
    '''
    Calculates the turnover rate for a 2-pathway, 6-state catalytic cycle with two
    driven steps (at epsilon_1 and epsilon_2), each with its own driving force
    (dG). Each driven step has its own "completion" rate (gamma_1 and gamma_2).
    All other rate constants are set to 1, except for the reverses of the driven steps,
    which are determined by detailed balance. Transitions between the two pathways are 
    given by (alpha_14, alpha_41), which represent the relative timescales of breaking
    steady-state.'''

    epsilon_1, epsilon_2, alpha_21, alpha_32, gamma_1, gamma_2, alpha_14, alpha_41, dG_1, dG_2, k_bT = mpmath.mpf(epsilon_1), mpmath.mpf(epsilon_2), mpmath.mpf(alpha_21), mpmath.mpf(alpha_32), mpmath.mpf(gamma_1), mpmath.mpf(gamma_2), mpmath.mpf(alpha_14), mpmath.mpf(alpha_41), mpmath.mpf(dG_1), mpmath.mpf(dG_2), mpmath.mpf(k_bT)
    
    alpha_54 = alpha_21
    alpha_65 = alpha_32

    alpha_23 = alpha_32 * mpmath.exp((dG_2 - epsilon_1) / k_bT)
    alpha_12 = alpha_21 * mpmath.exp((dG_1 + epsilon_1) / k_bT)

    alpha_56 = alpha_65 * mpmath.exp((dG_2 - epsilon_2) / k_bT)
    alpha_45 = alpha_54 * mpmath.exp((dG_1 + epsilon_2) / k_bT)



    alpha_25 = alpha_14
    alpha_52 = alpha_41
    alpha_36 = alpha_14
    alpha_63 = alpha_41

    sigma_1 = alpha_63 * alpha_32 + alpha_36 * alpha_65 + alpha_36 * gamma_2 + \
        alpha_63 * gamma_1 + gamma_1 * gamma_2 + alpha_65 * alpha_32 + \
        alpha_32 * gamma_2 + alpha_65 * gamma_1
    sigma_2 = alpha_41 * alpha_12 + alpha_14 * alpha_45 + alpha_12 * alpha_45
    sigma_3 = 1
    sigma_4 = alpha_14 * alpha_25 * alpha_36 + alpha_21 * alpha_14 * alpha_36 + \
        alpha_14 * alpha_23 * alpha_36 + alpha_12 * alpha_23 * alpha_36 + \
        alpha_32 * alpha_21 * alpha_14 + alpha_12 * alpha_32 * alpha_25 + \
        gamma_1 * alpha_21 * alpha_14 + gamma_1 * alpha_12 * alpha_25 + \
        gamma_1 * alpha_14 * alpha_25 + alpha_23 * gamma_1 * alpha_14 + \
        alpha_12 * alpha_25 * alpha_36 + alpha_32 * alpha_25 * alpha_14
    sigma_5 = alpha_41 * alpha_52 * alpha_63 + alpha_54 * alpha_41 * alpha_63 + \
        alpha_41 * alpha_56 * alpha_63 + alpha_45 * alpha_56 * alpha_63 \
        + alpha_45 * alpha_65 * alpha_52 + alpha_65 * alpha_54 * alpha_41 \
        + gamma_2 * alpha_41 * alpha_54 + gamma_2 * alpha_45 * alpha_52 + \
        gamma_2 * alpha_41 * alpha_52 + alpha_56 * gamma_2 * alpha_41 + \
        alpha_45 * alpha_52 * alpha_63 + alpha_65 * alpha_52 * alpha_41
    sigma_6 = alpha_25 + alpha_23 + alpha_21
    sigma_7 = alpha_52 + alpha_56 + alpha_54
    sigma_8 = alpha_36 + alpha_32 + gamma_1
    sigma_9 = alpha_63 + alpha_65 + gamma_2
    sigma_10 = alpha_45 + alpha_41
    sigma_11 = alpha_14 + alpha_12

    sigma_f1 = alpha_21 * alpha_54 * alpha_32 * alpha_63 \
        + alpha_21 * alpha_32 * alpha_63 * alpha_56 \
        + alpha_23 * alpha_36 * alpha_65 * alpha_54 \
        + alpha_21 * alpha_36 * alpha_65 * alpha_54 \
        + alpha_32 * alpha_21 * alpha_65 * alpha_54 \
        + alpha_21 * alpha_52 * alpha_32 * alpha_63 \
        + alpha_54 * alpha_25 * alpha_65 * alpha_36 \
        + alpha_21 * alpha_52 * alpha_65 * alpha_36 \
        + alpha_54 * alpha_25 * alpha_32 * alpha_63 \
        + alpha_54 * alpha_25 * alpha_65 * alpha_32 \
        + alpha_21 * alpha_32 * alpha_52 * alpha_65 \
        + alpha_21 * alpha_54 * alpha_36 * gamma_2 \
        + alpha_21 * alpha_54 * gamma_1 * alpha_63 \
        + alpha_21 * alpha_54 * gamma_1 * gamma_2 \
        + alpha_21 * alpha_54 * alpha_65 * gamma_1 \
        + alpha_21 * alpha_54 * alpha_32 * gamma_2 \
        + alpha_21 * alpha_52 * gamma_1 * gamma_2 \
        + alpha_21 * alpha_52 * alpha_65 * gamma_1 \
        + alpha_54 * alpha_25 * gamma_1 * gamma_2 \
        + alpha_54 * alpha_25 * alpha_32 * gamma_2 \
        + alpha_54 * alpha_25 * alpha_65 * gamma_1 \
        + alpha_54 * alpha_25 * alpha_63 * gamma_1 \
        + alpha_54 * alpha_25 * alpha_36 * gamma_2 \
        + alpha_21 * alpha_52 * gamma_1 * alpha_63 \
        + alpha_21 * alpha_52 * gamma_2 * alpha_36 \
        + alpha_21 * alpha_52 * alpha_32 * gamma_2
    
    sigma_f2 = alpha_12 * alpha_32 * alpha_45 * alpha_65 \
    + alpha_41 * alpha_12 * alpha_32 * alpha_65 \
    + alpha_14 * alpha_45 * alpha_32 * alpha_65 \
    + alpha_12 * alpha_45 * alpha_63 * alpha_32 \
    + alpha_12 * alpha_45 * alpha_65 * alpha_36 \
    + alpha_14 * alpha_45 * alpha_65 * alpha_36 \
    + alpha_41 * alpha_12 * alpha_36 * alpha_65 \
    + alpha_41 * alpha_12 * alpha_32 * alpha_63 \
    + alpha_14 * alpha_45 * alpha_32 * alpha_63 \
    + alpha_12 * alpha_45 * gamma_1 * gamma_2 \
    + alpha_14 * alpha_45 * gamma_1 * gamma_2 \
    + alpha_41 * alpha_12 * gamma_1 * gamma_2 \
    + alpha_12 * alpha_45 * alpha_32 * gamma_2 \
    + alpha_14 * alpha_45 * alpha_32 * gamma_2 \
    + alpha_12 * alpha_32 * alpha_41 * gamma_2 \
    + alpha_12 * alpha_45 * alpha_36 * gamma_2 \
    + alpha_14 * alpha_45 * alpha_36 * gamma_2 \
    + alpha_41 * alpha_12 * alpha_36 * gamma_2 \
    + alpha_12 * alpha_45 * alpha_63 * gamma_1 \
    + alpha_12 * alpha_41 * alpha_63 * gamma_1 \
    + alpha_14 * alpha_45 * alpha_63 * gamma_1 \
    + alpha_12 * alpha_45 * alpha_65 * gamma_1 \
    + alpha_41 * alpha_12 * alpha_65 * gamma_1 \
    + alpha_14 * alpha_45 * alpha_65 * gamma_1 

    sigma_f3 = alpha_41 * alpha_12 * alpha_23 * alpha_56 \
    + alpha_54 * alpha_41 * alpha_12 * alpha_23 \
    + alpha_21 * alpha_14 * alpha_45 * alpha_56 \
    + alpha_14 * alpha_45 * alpha_56 * alpha_23 \
    + alpha_12 * alpha_23 * alpha_45 * alpha_56 \
    + alpha_41 * alpha_12 * alpha_23 * alpha_52 \
    + alpha_14 * alpha_45 * alpha_56 * alpha_25 \
    + alpha_14 * alpha_45 * alpha_52 * alpha_23 \
    + alpha_41 * alpha_12 * alpha_25 * alpha_56 \
    + alpha_12 * alpha_25 * alpha_45 * alpha_56 \
    + alpha_12 * alpha_23 * alpha_52 * alpha_45    

    # 3 futile cycles (no conversion)

    Pi_1_plus = alpha_14 * alpha_45 * alpha_52 * alpha_21
    Pi_1_minus = alpha_41 * alpha_54 * alpha_25 * alpha_12

    Pi_2_plus = alpha_25 * alpha_56 * alpha_63 * alpha_32
    Pi_2_minus = alpha_52 * alpha_65 * alpha_36 * alpha_23

    Pi_3_plus = alpha_14 * alpha_45 * alpha_56 * alpha_63 * alpha_32 * alpha_21
    Pi_3_minus = alpha_41 * alpha_54 * alpha_65 * alpha_36 * alpha_23 * alpha_12

    # completion at eps2, eps1

    Pi_4_minus = alpha_45 * alpha_56 * gamma_2

    Pi_5_plus = alpha_12 * alpha_23 * gamma_1

    # completion at eps1, eps2, breaking steady-state, but no net benefit

    Pi_6_plus = alpha_14 * alpha_45 * alpha_56 * alpha_63 * gamma_1

    Pi_7_minus = alpha_41 * alpha_12 * alpha_23 * alpha_36 * gamma_2

    # exactly what you don't want by breaking steady-state

    Pi_8_minus = alpha_41 * alpha_12 * alpha_25 * alpha_56 * gamma_2

    # exactly what you want by breaking steady-state

    Pi_9_plus = alpha_14 * alpha_45 * alpha_52 * alpha_23 * gamma_1

    # exactly what you don't want by breaking steady-state

    Pi_10_plus = alpha_12 * alpha_25 * alpha_56 * alpha_63 * gamma_1

    # exactly what you want by breaking steady-state

    Pi_11_minus = alpha_45 * alpha_52 * alpha_23 * alpha_36 * gamma_2

    # fluxes from i -> ii -> i
    Pi_f1 = alpha_14 * alpha_41
    Pi_f2 = alpha_25 * alpha_52
    Pi_f3 = alpha_36 * alpha_63

    J_1 = sigma_1 * (Pi_1_plus - Pi_1_minus)
    J_2 = sigma_2 * (Pi_2_plus - Pi_2_minus)
    J_3 = sigma_3 * (Pi_3_plus - Pi_3_minus) 
    J_4 = sigma_4 * ( - Pi_4_minus) 
    J_5 = sigma_5 * (Pi_5_plus) 
    J_6 = sigma_6 * (Pi_6_plus)
    J_7 = sigma_7 * ( - Pi_7_minus)
    J_8 = sigma_8 * ( - Pi_8_minus)
    J_9 = sigma_9 * (Pi_9_plus) 
    J_10 = sigma_10 * (Pi_10_plus) 
    J_11 = sigma_11 * ( - Pi_11_minus)
    J_f1 = sigma_f1 * Pi_f1
    J_f2 = sigma_f2 * Pi_f2
    J_f3 = sigma_f3 * Pi_f3

    # Calculation of partition function
    

    weights = map_graph_weights(alpha_14=alpha_14, alpha_41=alpha_41, epsilon_1=epsilon_1, epsilon_2=epsilon_2, gamma=gamma_1, dG_1=dG_1, dG_2=dG_2)
    tree_weights = [root_tree_weights(spanTreesGraph[i], weights, nodes=None) for i in range(len(spanTreesGraph))]
    sigma = np.sum(tree_weights)   
        
    J_1 = J_1 / sigma
    J_2 = J_2 / sigma
    J_3 = J_3 / sigma
    J_4 = J_4 / sigma
    J_5 = J_5 / sigma
    J_6 = J_6 / sigma
    J_7 = J_7 / sigma

    J_8 = J_8 / sigma
    J_9 = J_9 / sigma
    J_10 = J_10 / sigma
    J_11 = J_11 / sigma
    J_f1 = J_f1 / sigma
    J_f2 = J_f2 / sigma
    J_f3 = J_f3 / sigma

    return J_1, J_2, J_3, J_4, J_5, J_6, J_7, J_8, J_9, J_10, J_11, J_f1, J_f2, J_f3



n_param = 3
n_v = 20

arguments = []
J_turnover_list = []
J_futile_list = []
max_ss_turnover_list = []

 # Calculation of partition function
G = nx.Graph()
G.add_nodes_from([2, 1, 0, 3, 4, 5])
G.add_edges_from([(0, 1), (1, 2), (0,2), (3,4), (4,5), (3,5), (0,3), (1,4), (2,5)])
spanTreesGraph = Spanning_Trees_Generator(G)

spanTreesGraph[14] = np.nan
spanTreesGraph[17] = np.nan
spanTreesGraph[27] = np.nan
spanTreesGraph[30] = np.nan
spanTreesGraph[33] = np.nan
spanTreesGraph[58] = np.nan
spanTreesGraph[61] = np.nan
spanTreesGraph[70] = np.nan
spanTreesGraph[73] = np.nan
spanTreesGraph = [g for g in spanTreesGraph if type(g) != float]
print('calculated')

# gamma = 20000
# epsilon_list = np.linspace(-10, 10, 100)
# J_turnover_total_list = []
# for epsilon in epsilon_list:
#         alpha_14_ss = 1e11
#         alpha_41_ss = 1e11
#         J_1, J_2, J_3, J_4, J_5, J_6, J_7, J_8, J_9, J_10, J_11, J_f1, J_f2, J_f3 = dynamic_steady_state_diagram(
#                 epsilon, epsilon, 1, 1, gamma, gamma, alpha_14_ss, alpha_41_ss, dG_1=2, dG_2=2, k_bT=1, spanTreesGraph=spanTreesGraph)
        
#         J_turnover_total = np.abs(J_4) + np.abs(J_5) + np.abs(J_6) + np.abs(J_7) + np.abs(J_8) + np.abs(J_9) + np.abs(J_10) + np.abs(J_11)
#         J_turnover_total_list.append(J_turnover_total)
# J_turnover_total_list = np.array(J_turnover_total_list)
# idx = int(np.where(J_turnover_total_list == np.max(J_turnover_total_list))[0])
# print(idx, J_turnover_total_list[idx], epsilon_list[idx])
# oscMidpoint = epsilon_list[idx]
# J_max_ss = J_turnover_total_list[idx]

def rate_fn(param):
    [alpha_41, alpha_14, amplitude] = param
    # alpha_14 = alpha_41
    # amplitude = 3
    amplitude = np.log10(amplitude)
    gamma = 1e5
    alpha_21 = 1
    alpha_32 = 1
    
    dG_1 = 2
    dG_2 = 2
    # alpha_21 = 1
    # alpha_32 = 1
    k_bT = 1
    oscMidpoint = -1/2 * (-dG_1 + dG_2 - k_bT * np.log((alpha_21 * alpha_32 + alpha_21 * gamma) / (alpha_32 * gamma)))
    
    J_max_ss = gamma * (alpha_32 * np.exp((dG_1 + dG_2)/k_bT) * np.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))) \
    / (gamma * (2 * np.exp((dG_1 + dG_2) / (2 * k_bT)) + np.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))) \
    + alpha_32 * (2 * np.exp((dG_1 + dG_2) / (2 * k_bT)) + np.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma)) \
    + np.exp((dG_1 + dG_2) / k_bT) * np.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))))

    # epsilon_list = np.linspace(-10, 10, 100)
    # J_turnover_total_list = []
    # for epsilon in epsilon_list:
    #     alpha_14_ss = 1e11
    #     alpha_41_ss = 1e11
    #     J_1, J_2, J_3, J_4, J_5, J_6, J_7, J_8, J_9, J_10, J_11, J_f1, J_f2, J_f3 = dynamic_steady_state_diagram(
    #             epsilon, epsilon, alpha_21, alpha_32, gamma, gamma, alpha_14_ss, alpha_41_ss, dG_1=dG_1, dG_2=dG_2, k_bT=1, spanTreesGraph=spanTreesGraph)
        
    #     J_turnover_total = np.abs(J_4) + np.abs(J_5) + np.abs(J_6) + np.abs(J_7) + np.abs(J_8) + np.abs(J_9) + np.abs(J_10) + np.abs(J_11)
    #     J_turnover_total_list.append(J_turnover_total)

    # print('calculated max.')
        
    # J_turnover_total_list = np.array(J_turnover_total_list)
    # idx = int(np.where(J_turnover_total_list == np.max(J_turnover_total_list))[0])
    # print(idx, J_turnover_total_list[idx], epsilon_list[idx])
    # oscMidpoint = epsilon_list[idx]
    # oscMidpoint = -0.10101010101010033
    epsilon_1 = oscMidpoint + amplitude
    epsilon_2 = oscMidpoint - amplitude
    J_1, J_2, J_3, J_4, J_5, J_6, J_7, J_8, J_9, J_10, J_11, J_f1, J_f2, J_f3 = dynamic_steady_state_diagram(
        epsilon_1, epsilon_2, alpha_21, alpha_32, gamma, gamma, alpha_14, alpha_41, dG_1=dG_1, dG_2=dG_2, k_bT=1, spanTreesGraph=spanTreesGraph)
    J_turnover_total = np.abs(J_4) + np.abs(J_5) + np.abs(J_6) + np.abs(J_7) + np.abs(J_8) + np.abs(J_9) + np.abs(J_10) + np.abs(J_11)
    # J_futile = 2 * np.abs(J_1) + 2 * np.abs(J_2) + 2 * np.abs(J_3) + 2 * np.abs(J_6) + 2 * np.abs(J_7) + 2 * np.abs(J_8) + 2 * np.abs(J_9) \
    #     + 2 * np.abs(J_10) + 2 * np.abs(J_11)#+ np.abs(J_f1) + np.abs(J_f2) + np.abs(J_f3)
    J_futile = np.abs(J_f1) + np.abs(J_f2) + np.abs(J_f3)
    # print(J_turnover_total / np.max(J_turnover_total_list), J_futile/J_turnover_total)
    return J_turnover_total / J_max_ss, J_futile / J_turnover_total
    # return J_turnover_total/J_max_ss, J_turnover_total/(J_turnover_total + J_futile)

def epsilon_heat_engine(k_b, F, params):
    epsilon_1, epsilon_2, alpha_21, alpha_32, gamma_1, gamma_2, dG_1, dG_2 = params
    
    dynamic_steady_state_diagram(epsilon_1, epsilon_2, alpha_21, alpha_32, gamma_1, gamma_2, alpha_14, alpha_41, dG_1=2, dG_2=2, k_bT=1, spanTreesGraph= None)
    return epsilon_1 - epsilon_2



# for alpha_21 in np.logspace(-3, 3, n_param):
#     for alpha_32 in np.logspace(-3, 3, n_param):
#         # for alpha_54 in np.logspace(-2, 2, n_param):
#         #     for alpha_65 in np.logspace(-2, 2, n_param):
#         for alpha_14 in np.logspace(-2, 2, n_param):
#             for alpha_41 in np.logspace(-2, 2, n_param):
#                 for dG_1 in np.linspace(0,2,n_param):
#                     for dG_2 in np.linspace(0,2,n_param):
#                         for amplitude in np.linspace(0, 2, n_param):
#                             for gamma in np.linspace(0.01, 100, n_v):
#                                 # alpha_54 = alpha_21
#                                 # alpha_65 = alpha_32
#                                 param = [amplitude, alpha_21, alpha_32, alpha_41, alpha_14, dG_1, dG_2, amplitude, gamma]
#                                 arguments.append([param])

# if __name__ == '__main__':
#     with Pool(4) as p:
#         rates = p.starmap(rate_fn, arguments)

# # Convert the lists into Numpy arrays
# rates_abs_ls = np.array(rates)
# J_turnover_total = rates_abs_ls[:, 0]
# J_futile = rates_abs_ls[:, 1]
# max_ss_turnover = rates_abs_ls[:, 2]

# # Number of significant digits kept for plotting data
# J_turnover_abs_ls_sparse = []
# J_futile_abs_ls_sparse = []
# J_turnover_ss_abs_ls_sparse = []
# n_round = 7

# for i in range(len(J_turnover_total)):
#     J_turnover_abs_ls_sparse.append(np.round(J_turnover_total[i],n_round))
#     J_futile_abs_ls_sparse.append(np.round(J_futile[i],n_round))
#     J_turnover_ss_abs_ls_sparse.append(np.round(max_ss_turnover[i],n_round))

# with open('FigS6_output_data_dynamic_turnover.p', 'wb') as f:
    pickle.dump([J_turnover_abs_ls_sparse, J_futile_abs_ls_sparse, J_turnover_ss_abs_ls_sparse], f)