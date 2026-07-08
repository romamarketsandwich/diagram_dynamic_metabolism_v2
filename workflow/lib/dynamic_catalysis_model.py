import numpy as np
import mpmath
import networkx as nx
import copy
from itertools import product

def root_tree_weights(spanning_tree, weights, absorption_edge_1=None, absorption_edge_2=None, nodes=None, allNodes=True):
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

    allNodes is a boolean describing if wwe should get all valid roots of the tree (MTT) or just one node (cycle MTT).
    """
    spanning_tree = spanning_tree.copy()
    prefactor = 1
    if absorption_edge_1 is not None:
        reversed_absorption_edge_1 = absorption_edge_1[::-1]
    if absorption_edge_2 is not None:
        reversed_absorption_edge_2 = absorption_edge_2[::-1]
    if absorption_edge_1 is not None and absorption_edge_2 is not None:
        if (absorption_edge_1 in spanning_tree.edges or reversed_absorption_edge_1 in spanning_tree.edges) and (absorption_edge_2 in spanning_tree.edges or reversed_absorption_edge_2 in spanning_tree.edges):
            spanning_tree.remove_edges_from([absorption_edge_1, reversed_absorption_edge_1, absorption_edge_2, reversed_absorption_edge_2])
            nodes1 = nx.node_connected_component(spanning_tree, absorption_edge_1[1])
            nodes2 = nx.node_connected_component(spanning_tree, absorption_edge_2[1])
            if len(nodes1) > len(nodes2):
                nodes = nodes2 if allNodes else nodes
            else:
                nodes = nodes1 if allNodes else nodes
            prefactor = prefactor * root_tree_weights(spanning_tree, weights, absorption_edge_1=absorption_edge_1, absorption_edge_2=absorption_edge_2, nodes=[absorption_edge_1[0]], allNodes=allNodes) * \
                weights.get(absorption_edge_1) * \
                root_tree_weights(spanning_tree, weights, absorption_edge_1=absorption_edge_1, absorption_edge_2=absorption_edge_2, nodes=[absorption_edge_2[0]], allNodes=allNodes) * \
                weights.get(absorption_edge_2)
        elif absorption_edge_1 in spanning_tree.edges or reversed_absorption_edge_1 in spanning_tree.edges:
            spanning_tree.remove_edges_from([absorption_edge_1, reversed_absorption_edge_1])
            nodes = nx.node_connected_component(spanning_tree, absorption_edge_1[1]) if allNodes else nodes
            prefactor = prefactor * root_tree_weights(spanning_tree, weights, absorption_edge_1=absorption_edge_1, absorption_edge_2=absorption_edge_2, nodes=[absorption_edge_1[0]], allNodes=allNodes) * weights.get(absorption_edge_1)
        elif absorption_edge_2 in spanning_tree.edges or reversed_absorption_edge_2 in spanning_tree.edges:
            spanning_tree.remove_edges_from([absorption_edge_2, reversed_absorption_edge_2])
            nodes = nx.node_connected_component(spanning_tree, absorption_edge_2[1]) if allNodes else nodes
            prefactor = prefactor * root_tree_weights(spanning_tree, weights, absorption_edge_1=absorption_edge_1, absorption_edge_2=absorption_edge_2, nodes=[absorption_edge_2[0]], allNodes=allNodes) * weights.get(absorption_edge_2)
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

def map_graph_weights_six_states(alpha_32, alpha_21, alpha_65, alpha_54,\
                        alpha_14, alpha_41, dG_1, dG_2,\
                        gamma, epsilon_1, epsilon_2, k_bT):
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

def contract_node(node_1, node_2, graph, graph_weights, absorption_edge_1, absorption_edge_2):
    '''
    contract a graph by the specified edge. Adjust the graph_weights dict to re-map the node indices to the contracted graph.
    node1: node to retain index number
    node2: node to contract into node1
    graph: undirected networkx graph showing connectivity (including absorption arrows). 
    graph_weights: dict of weights assigned to edges (use function map_graph_weights)
    '''
    new_graph = nx.contracted_nodes(graph, node_1, node_2, self_loops=False, copy=True, store_contraction_as=None)

    # need to make sure that any edges emanating from node_2 now emanate from node_1
    # also, need to make sure that any edges inflowing into node_2 now inflow into node_1.
    # finally, edges going from node_2 -> node_1 or vice versa are now irrelevant (in cycle coefficient)
    new_graph_weights = copy.deepcopy(graph_weights)
    for key in graph_weights:
        #check for node_2 <-> node_1 -- this is part of the cycle.
        if key == (node_1, node_2) or key == (node_2, node_1):
            new_graph_weights.pop(key, None)
        # check for outflow -- no need to adjust weights here, since a valid spanning tree has no outflowing nodes from the root.
        elif key[0] == node_2:
            new_graph_weights[(node_1, key[1])] = graph_weights[(node_1, key[1])] + graph_weights[key] if (node_1, key[1]) in graph_weights else graph_weights[key]
            new_graph_weights.pop(key, None)
        # check for inflow
        elif key[1] == node_2:
            new_graph_weights[(key[0], node_1)] = graph_weights[(key[0], node_1)] + graph_weights[key] if (key[0], node_1) in graph_weights else graph_weights[key]
            new_graph_weights.pop(key, None)

    #verify that the index of the start or end of the absorption arrow hasn't changed, else change to node_1.
    # of course, if the absorption arrow is a self-directed edge then nothing needs to be done, as our Markov Graph doesn't allow for self-directed edges.
    if absorption_edge_1 is not None and absorption_edge_2 is not None:
        if node_2 == absorption_edge_1[0]:
            absorption_edge_1 = (node_1, absorption_edge_1[1])
        if node_2 == absorption_edge_1[1]:
            absorption_edge_1 = (absorption_edge_1[0], node_1)
        if node_2 == absorption_edge_2[0]:
            absorption_edge_2 = (node_1, absorption_edge_2[1])
        if node_2 == absorption_edge_2[1]:
            absorption_edge_2 = (absorption_edge_2[0], node_1)

    # finally, if we have contracted the graph such that an absorption arrow now has a reverse arrow, then it makes no sense to keep treating the absorption arrow like one.
    # we 'neuter' the absorption requirement by making it a self-directed edge.
    # if absorption_edge_1[::-1] in new_graph_weights.keys():
    #     absorption_edge_1 = (absorption_edge_1[0], absorption_edge_1[0])
    # if absorption_edge_2[::-1] in new_graph_weights.keys():
    #     absorption_edge_2 = (absorption_edge_2[0], absorption_edge_1[0])

    # print(set(graph_weights.keys()).difference(new_graph_weights.keys()))
    # print(set(new_graph_weights.keys()).difference(graph_weights.keys()))

    return new_graph_weights, new_graph, absorption_edge_1, absorption_edge_2

def is_valid_root(spanning_tree, root_node, absorption_edge_1, absorption_edge_2):
    '''
    given a spanning tree and a root, return if the given root is a valid spanning tree.

    root would not be a valid root of a spanning tree if it is upstream of an absorption arrow.
    '''
    temp1 = copy.deepcopy(spanning_tree)
    temp2 = copy.deepcopy(spanning_tree)
    reversed_absorption_edge_1 = None
    reversed_absorption_edge_2 = None
    if absorption_edge_1 is None and absorption_edge_2 is None:
        return True
    if absorption_edge_1 is not None:
        reversed_absorption_edge_1 = absorption_edge_1[::-1]
    if absorption_edge_2 is not None:
        reversed_absorption_edge_2 = absorption_edge_2[::-1]
    # case 1: both absorption edges in tree, so verify that node_1 is in the intersection of (downstream tree of G - absorption edge 2) and (downstream tree of G - absorption edge 1)
    if absorption_edge_1 in temp1.edges and absorption_edge_2 in temp1.edges:
        temp1.remove_edges_from([absorption_edge_1, reversed_absorption_edge_1])
        temp2.remove_edges_from([absorption_edge_2, reversed_absorption_edge_2])
        nodes1 = nx.node_connected_component(temp1, absorption_edge_1[1])
        nodes2 = nx.node_connected_component(temp2, absorption_edge_2[1])
        if root_node not in nodes1 or root_node not in nodes2:
            return False
    # case 2/3: only one absorption edge, so just check that node_1 is downstream.
    elif absorption_edge_1 in temp1.edges:
        temp1.remove_edges_from([absorption_edge_1, reversed_absorption_edge_1])
        nodes = nx.node_connected_component(temp1, absorption_edge_1[1])
        if root_node not in nodes:
            return False
    elif absorption_edge_2 in temp1.edges:
        temp1.remove_edges_from([absorption_edge_2, reversed_absorption_edge_2])
        nodes = nx.node_connected_component(temp1, absorption_edge_2[1])
        if root_node not in nodes:
            return False
    return True

def cycle_inflows(cycle, graph, graph_weights, absorption_edge_1=None, absorption_edge_2=None):
    '''
    Calculate Sigma, the net inflow into the cycle. 

    cycle: iterable of node indices with no repeated nodes. The assumption is that for indices 1, 2, ... n,
        the cycle will proceed through nodes 1, 2, ..., n, 1, ...
        Make sure that cycle has at least two nodes
    graph: undirected networkx graph showing connectivity (including absorption arrows). 
    graph_weights: dict of weights assigned to edges (use function map_graph_weights

    '''
    assert len(cycle) >= 2    
    # contract cycle into a new node, named the first node of the cycle.
    node_1 = cycle[0]

    new_graph = copy.deepcopy(graph)
    new_graph_weights = copy.deepcopy(graph_weights)
    if node_1 != cycle[1]:
        for i in range(1, len(cycle)):
            new_graph_weights, new_graph, absorption_edge_1, absorption_edge_2 = contract_node(node_1, cycle[i], new_graph, new_graph_weights, absorption_edge_1, absorption_edge_2)

    # get spanning trees of new graph
    
    span_trees_graph = [i for i in nx.SpanningTreeIterator(new_graph)]
    # check if node_1 is a valid root of each spanning tree, if not then exclude that spanning tree.
    for index, tree in enumerate(span_trees_graph):
        if not is_valid_root(tree, node_1, absorption_edge_1=absorption_edge_1, absorption_edge_2=absorption_edge_2):
            span_trees_graph[index] = np.nan
    span_trees_graph = [g for g in span_trees_graph if type(g) != float]
    Sigma = 0
    for spanning_tree in span_trees_graph:
        # print(root_tree_weights(spanning_tree, new_graph_weights, absorption_edge_1=absorption_edge_1, absorption_edge_2=absorption_edge_2, nodes=[node_1]))
        Sigma += root_tree_weights(spanning_tree, new_graph_weights, absorption_edge_1=absorption_edge_1, absorption_edge_2=absorption_edge_2, nodes=[node_1], allNodes=False)
    return Sigma
    # return span_trees_graph

def cycle_coefficient(cycle, graph_weights):
    '''
    Calculate kappa, the cycle coefficient for a general n-state cycle
    
    cycle: iterable of node indices with no repeated nodes. The assumption is that for indices 1, 2, ... n,
        the cycle will proceed through nodes 1, 2, ..., n, 1, ...
        Make sure that cycle has at least two nodes.

    graph_weights: dict of weights assigned to edges (use function map_graph_weights)
    '''
    assert len(cycle) >= 2


    new_cycle = [i for i in cycle] + [cycle[0]]

    cycle_coefficient = 1
    for i in range(len(cycle)):
        edge = (new_cycle[i], new_cycle[i+1])
        edge_weight = graph_weights[edge]
        cycle_coefficient *= edge_weight
    return cycle_coefficient

def generate_spanning_trees(graph, absorption_edge_1=None, absorption_edge_2=None):
    '''
    generate all spanning trees for a given graph.
    Note that this is not the same as what's being performed in cycle_inflows. In cycle_inflows, we generate all valid spanning trees given a root node (a subset of this function's output).
    Here, we generate all valid spanning trees given _any_ root node. That is, if there is one or more valid roots of the tree, its spanning tree will be generated.
    Passing this set of spanning trees to root_tree_weights allows for the generation of the partition function (denominator of the MTT).
    '''
    span_trees_graph = [i for i in nx.SpanningTreeIterator(graph)]
    for tree_index, tree in enumerate(span_trees_graph):
        valid_nodes = []
        for node in tree.nodes:
            if is_valid_root(tree, node, absorption_edge_1, absorption_edge_2):
                valid_nodes.append(node)
        if len(valid_nodes) == 0:
            span_trees_graph[tree_index] = np.nan
    return [g for g in span_trees_graph if type(g) != float]

def max_rate_six_states(alpha_32, alpha_21, alpha_65, alpha_54,\
                        alpha_14, alpha_41, dG_1, dG_2,\
                        gamma, epsilon_1, epsilon_2, k_bT):
    '''
    get the maximum rate at steady-state under the Sabatier principle. Expression derived in Mathematica.
    '''
    return gamma * (alpha_32 * mpmath.exp((dG_1 + dG_2)/k_bT) * mpmath.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))) \
    / (gamma * (2 * mpmath.exp((dG_1 + dG_2) / (2 * k_bT)) + mpmath.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))) \
    + alpha_32 * (2 * mpmath.exp((dG_1 + dG_2) / (2 * k_bT)) + mpmath.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma)) \
    + mpmath.exp((dG_1 + dG_2) / k_bT) * mpmath.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))))

def osc_midpoint_six_states(alpha_32, alpha_21, alpha_65, alpha_54,\
                        alpha_14, alpha_41, dG_1, dG_2,\
                        gamma, epsilon_1, epsilon_2, k_bT):
    return -1/2 * (-dG_1 + dG_2 - k_bT * mpmath.log((alpha_21 * alpha_32 + alpha_21 * gamma) / (alpha_32 * gamma)))




def cycle_flux_fn(turnover_cycles, futile_cycles, reaction_graph, absorption_edge_1=None, absorption_edge_2=None, max_ss_rate_fn=None, weight_mapping_fn=None, **params):
    '''
    calculate the cycle flux given two sets of biochemical cycles (called turnover_cycles, futile_cycles here).
    The output (total flux attributable to turnover_cycles, total flux attributable to futile_cycles) will be used as input to BoundaryExplorer.
    
    turnover_cycles: list of list of nodes. each constituent list of nodes represents one biochemical cycle attributable to turnover.
    futile_cycles: list of list of nodes. each constituent list of nodes represents one biochemical cycle attributable to futile cycling.
    reaction_graph: the dynamic reaction network to be subjected to the cyclic MTT.
    graph_weights: weights of reaction_graph (generated with map_graph_weights)
    absorption_edge_1: 2-tuple of the directed absorption edge representing turnover
    absorption_edge_2: 2-tuple of the other directed absorption edge representing turnover.
    '''
    graph_weights = weight_mapping_fn(**params)
    max_steady_turnover = max_ss_rate_fn(**params)
    spanning_trees = generate_spanning_trees(reaction_graph, absorption_edge_1=absorption_edge_1, absorption_edge_2=absorption_edge_2)
    tree_weights = [root_tree_weights(spanning_trees[i], graph_weights, absorption_edge_1=absorption_edge_1, absorption_edge_2=absorption_edge_2, nodes=None, allNodes=True) for i in range(len(spanning_trees))]
    partition_function= np.sum(tree_weights) 

    turnover_flux = 0
    futile_flux = 0
    for cycle in turnover_cycles:
        cycle_flux = (cycle_coefficient(cycle, graph_weights) * cycle_inflows(cycle, reaction_graph, graph_weights, absorption_edge_1, absorption_edge_2)) / partition_function
        turnover_flux += cycle_flux
    for cycle in futile_cycles:
        cycle_flux = (cycle_coefficient(cycle, graph_weights) * cycle_inflows(cycle, reaction_graph, graph_weights, absorption_edge_1, absorption_edge_2)) / partition_function
        futile_flux += cycle_flux
    v_bar = turnover_flux / max_steady_turnover
    q_bar = futile_flux / turnover_flux
    # q_bar = futile_flux / max_steady_turnover
    return v_bar, q_bar


def map_graph_weights_eight_states(alpha_32, alpha_21, alpha_43, alpha_65, alpha_76, alpha_87,\
        alpha_15, alpha_51, dG_1, dG_2, dG_3,\
        gamma, epsilon_1, epsilon_2, k_bT):
    alpha_12 = alpha_21 * mpmath.exp((dG_1 + epsilon_1) / k_bT)
    alpha_23 = alpha_32 * mpmath.exp((dG_2)/ k_bT)
    alpha_34 = alpha_43 * mpmath.exp((dG_3 - epsilon_1)/ k_bT)


    alpha_56 = alpha_65 * mpmath.exp((dG_1 + epsilon_2) / k_bT)
    alpha_67 = alpha_76 * mpmath.exp((dG_2) / k_bT)
    alpha_78 = alpha_87 * mpmath.exp((dG_2 - epsilon_2) / k_bT)


    alpha_26 = alpha_15
    alpha_62 = alpha_51
    alpha_37 = alpha_15
    alpha_73 = alpha_51
    alpha_48 = alpha_15
    alpha_84 = alpha_51

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


def osc_midpoint_eight_states(alpha_32, alpha_21, alpha_43, dG_1, dG_2, dG_3, gamma, k_bT):
    return 1/2 * (dG_1 - dG_3 + \
                k_bT * np.log(-alpha_21 * alpha_32 * alpha_43 - \
                alpha_21 * alpha_32  * alpha_43 * np.exp(dG_2/k_bT) -\
                alpha_21  * alpha_32  * gamma - alpha_21 * alpha_32 * np.exp(dG_2/k_bT) * gamma)- \
                k_bT * np.log(alpha_21 * alpha_43 * gamma + alpha_32 * alpha_43 *np.exp(dG_2/k_bT) * gamma))

def max_rate_eight_states(alpha_32, alpha_21, alpha_43, alpha_65, alpha_54,\
                        alpha_14, alpha_41, dG_1, dG_2,dG_3,\
                        gamma, epsilon_1, epsilon_2, k_bT):
    return (alpha_21 * alpha_32 * alpha_43 * np.exp((3 * dG_1 + 2 * dG_2 + 3 * dG_3)/(2 * k_bT)) *\
    gamma * np.sqrt(alpha_43 * (alpha_21 + alpha_32 * np.exp(dG_2/k_bT)) * gamma) *\
    np.sqrt(-alpha_21 * alpha_32 (1 + np.exp(dG_2/k_bT)) * \
    (alpha_43 + gamma)))/(alpha_32**2 * alpha_43**2 * \
    np.exp((2 * (dG_2 + dG_3))/k_bT) * gamma**2 - \
    alpha_21**2 * (-alpha_43**2 * np.exp((2 * dG_3)/k_bT) \
    * gamma**2 + alpha_32**2 * np.exp((2 * dG_1)/k_bT)\
    * (1 + np.exp(dG_2/k_bT))**2 * (alpha_43 + gamma)**2) \
    + alpha_21 * np.exp(dG_3/(2 * k_bT)) * (alpha_43 * \
    np.exp((3 * dG_1 + 2 * dG_3)/(2 * k_bT)) * gamma * \
    np.sqrt(alpha_43 * (alpha_21 + alpha_32 *np.exp(dG_2/k_bT))\
    * gamma) * np.sqrt(-alpha_21 * alpha_32 * (1 + np.exp(dG_2/k_bT)) \
    * (alpha_43 + gamma)) + alpha_32 *  (2 * alpha_43**2 *np.exp((2 * dG_2 + 3 * dG_3)/(2 * k_bT)) \
    * gamma**2 + alpha_43 *np.exp(dG_1/(2 * k_bT)) * (1 + np.exp((dG_1 + dG_2 + dG_3)/k_bT)) \
    * np.sqrt(alpha_43 * (alpha_21 + alpha_32 *np.exp(dG_2/k_bT)) * gamma) * \
    np.sqrt(-alpha_21 * alpha_32 * (1 + np.exp(dG_2/k_bT)) * (alpha_43 + gamma)) + \
    np.exp(dG_1/(2 * k_bT)) * gamma * np.sqrt(alpha_43 * (alpha_21 + alpha_32 * \
    np.exp(dG_2/k_bT)) * gamma) *np.sqrt(-alpha_21 * alpha_32 * (1 + np.exp(dG_2/k_bT)) * (alpha_43 + gamma)))))

def generate_eight_state_graph():
    G = nx.Graph()
    G.add_nodes_from([3, 2, 1, 0, 4, 5, 6, 7])
    G.add_edges_from([(0, 1), (1, 2), (2,3), (0,3), (4,5), (5,6), (6,7), (4,7), (0,4), (1,5), (2,6), (3,7)])
    return G

def generate_eight_state_fundamental_cycles():
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