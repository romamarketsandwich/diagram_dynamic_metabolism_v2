import numpy as np
import networkx as nx
import copy
from itertools import product
from utilsGRF import BoundaryFinder as BF
import json
import sys
import importlib
import datetime
import os
from pathlib import Path
import tqdm

def calculate_tree_weights(subgraph, weights, absorption_edge_list=None, root_nodes=None, all_nodes=True):
    '''
    Calculate the weight of a tree rooted at a specific node in a subgraph.
    If multiple nodes are provided, then the weight of the tree rooted at each node is calculated and summed.
    Args:
        subgraph (nx.Graph): networkx graph representing the spanning tree of interest.
        weights (dict): dictionary mapping edges to their weights.
        absorption_edge_list (list): list of one-way edges in the subgraph.
        root_nodes (list): list of nodes to consider as roots.
    
    Returns: 
        float: total weight of the tree rooted at the specified nodes.
    '''
    #assumes at least one valid root in the spanning tree implementing one-way fluxes. 
    if root_nodes is None:
        root_nodes = [n for n in subgraph]

    
    """ 
    Key insight: if the tree contains a one-way edge, then nothing "upstream" of the one-way edge can be a root.
    Also, all flows "upstream" of one-way edge must flow through the one-way edge.
    
    Else, there would exist at least one edge in subgraph that would connect "head" and "tail" subgraphs,
    suggesting a cycle. This is a contradiction.

    Therefore, we can recursively treat tree weight calculation by:
        1. removing one-way edges, if possible
        2. calculating "tail" tree weights, assuming root at one-way edge "tail".
        3. tree weight rooted at node i in "head" = "tail" tree weight * one-way edge weight * "head" tree weight .
        4. Sum all tree weights over all i in the "head".

    all_nodes is a boolean describing if we should get all valid roots of the tree (MTT), True or just one node (cycle MTT), False.
    """

    """
    absorption_edge_set is a list of one-way edges. If None, then we assume no one-way edges in the spanning tree.
    
    """
    subgraph = subgraph.copy()
    prefactor = 1
    if absorption_edge_list is not None:
        reversed_absorption_edges = [edge[::-1] for edge in absorption_edge_list]
    if absorption_edge_list is not None:
        if any(edge in subgraph.edges or edge in reversed_absorption_edges for edge in absorption_edge_list):
            # get the edges in the spanning tree
            absorption_tree_edges = [edge for edge in absorption_edge_list if edge in subgraph.edges or edge[::-1] in subgraph.edges]
            subgraph.remove_edges_from([edge for edge in absorption_edge_list + reversed_absorption_edges])
            node_set = [nx.node_connected_component(subgraph, absorption_head) for absorption_head in [edge[1] for edge in absorption_edge_list]]
            # get member of node_set with least number of nodes
            if all_nodes:
                sorted_node_set = sorted(node_set, key=lambda x: len(x))
                root_nodes = sorted_node_set[0]
            for absorption_edge in absorption_tree_edges:
                prefactor = prefactor * calculate_tree_weights(subgraph, weights, absorption_edge_list=absorption_edge_list, root_nodes=[absorption_edge[0]], all_nodes=all_nodes) * \
                weights.get(absorption_edge)
    total_weight = 0
    for root in sorted(root_nodes):
        # Perform BFS to get the tree structure rooted at 'root'
        BFS_list = []
        visited = set()
        queue = [root]
        while queue:
            current = queue.pop(0)
            visited.add(current)
            for neighbor in subgraph.neighbors(current):
                if neighbor not in visited and (current, neighbor) in subgraph.edges:
                    BFS_list.append((neighbor, current))  # Store as (child, parent)
                    queue.append(neighbor)
        # Calculate the weight of the rooted tree
        tree_weight = np.prod([weights.get((u, v)) for u, v in BFS_list]) * prefactor
        # Store or process the tree weight as needed
        total_weight += tree_weight
    return total_weight 



def contract_node(node_1, node_2, graph, graph_weights, absorption_edge_list=None):
    '''
    contract a graph by the specified edge. Adjust the graph_weights dict to re-map the node indices to the contracted graph.
    Args:
        node1 (int): node label to retain index number
        node2 (int): node label to contract into node1
        graph (nx.Graph): undirected networkx graph showing connectivity (including absorption arrows). 
        graph_weights (dict): dict of weights assigned to edges (use function map_graph_weights)
        absorption_edge_list (list): list of absorption edges in the graph.

    Returns: 
        new_graph_weights (dict): dict of weights assigned to edges in the contracted graph.
        new_graph (nx.Graph): contracted networkx graph.
        new_absorption_edge_list (list): list of absorption edges in the contracted graph.
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
    new_absorption_edge_list = []
    if absorption_edge_list is not None:
        for absorption_edge in absorption_edge_list:
            if node_2 == absorption_edge[0]:
                absorption_edge = (node_1, absorption_edge[1])
            if node_2 == absorption_edge[1]:
                absorption_edge = (absorption_edge[0], node_1)
            new_absorption_edge_list.append(absorption_edge)

    return new_graph_weights, new_graph, new_absorption_edge_list

def is_valid_root(spanning_tree, root_node, absorption_edge_list):
    '''
    given a spanning tree and a root, return if the given root is a valid spanning tree.
    root would not be a valid root of a spanning tree if it is upstream of an absorption arrow.

    Args:
        spanning_tree (nx.Graph): undirected networkx graph showing connectivity (including absorption arrows). 
        root_node (int): node label to check if it is a valid root of the spanning tree.
        absorption_edge_list (list): list of absorption edges in the graph.
    Returns:
        bool: True if root_node is a valid root of the spanning tree, False otherwise.
    '''
    if absorption_edge_list is None or len(absorption_edge_list) == 0:
        return True
    reversed_absorption_edges = [edge[::-1] for edge in absorption_edge_list]
    # case 1: both absorption edges in tree, so verify that node_1 is in the intersection of (downstream tree of G - absorption edge i) for all i in absorption_edge_list. If not, then node_1 is upstream of at least one absorption edge and is not a valid root.
    absorption_tree_edges = [edge for edge in absorption_edge_list if edge in spanning_tree.edges or edge[::-1] in spanning_tree.edges]
    trees = [copy.deepcopy(spanning_tree) for _ in range(len(absorption_tree_edges))]
    for i, tree in enumerate(trees):
        tree.remove_edges_from([absorption_tree_edges[i], absorption_tree_edges[i][::-1]])
    nodes = [nx.node_connected_component(trees[i], absorption_tree_edges[i][1]) for i in range(len(absorption_tree_edges))]
    if any(root_node not in node_set for node_set in nodes):
        return False
    return True

def cycle_inflows(cycle, graph, graph_weights, absorption_edge_list=None):
    '''
    Calculate Sigma, the net inflow into the cycle. 

    Args:
        cycle (iterable): iterable of node indices with no repeated nodes.
        graph (nx.Graph): undirected networkx graph showing connectivity (including absorption arrows). 
        graph_weights (dict): dict of weights assigned to edges (use function map_graph_weights).
        absorption_edge_list (list): list of absorption edges in the graph.
    Returns:
        float: Sigma, the net inflow into the cycle.
    '''
    '''
    The assumption is that for indices 1, 2, ... n, the cycle will proceed through nodes 1, 2, ..., n, 1, ...
        Make sure that cycle has at least two nodes.
    '''
    assert len(cycle) >= 2    
    # contract cycle into a new node, named the first node of the cycle.
    node_1 = cycle[0]
    new_graph = copy.deepcopy(graph)
    new_graph_weights = copy.deepcopy(graph_weights)
    if node_1 != cycle[1]:
        for i in range(1, len(cycle)):
            new_graph_weights, new_graph, absorption_edge_list = contract_node(node_1, cycle[i], new_graph, new_graph_weights, absorption_edge_list)
    G = nx.DiGraph()
    for (u,v), attrs in new_graph_weights.items():
        G.add_edge(u, v, weight=np.float64(attrs))

    if not G.nodes:
        return 1

    Sigma = nx.number_of_spanning_trees(G.reverse(copy=False), weight='weight', root=node_1)
    return Sigma
    # return span_trees_graph

def cycle_coefficient(cycle, graph_weights):
    '''
    Calculate kappa, the cycle coefficient for a general n-state cycle
    Args:
        cycle (iterable): iterable of node indices with no repeated nodes. The assumption is that for indices 1, 2, ... n,
            the cycle will proceed through nodes 1, 2, ..., n, 1, ...
            Make sure that cycle has at least two nodes.

        graph_weights (dict): dict of weights assigned to edges (use function map_graph_weights)
    Returns:
        float: kappa, the cycle coefficient for the cycle.
    '''
    assert len(cycle) >= 2

    new_cycle = [i for i in cycle] + [cycle[0]]

    cycle_coefficient = 1
    for i in range(len(cycle)):
        edge = (new_cycle[i], new_cycle[i+1])
        edge_weight = graph_weights[edge]
        cycle_coefficient *= edge_weight
    return cycle_coefficient

def generate_spanning_trees(graph, absorption_edge_list=None):
    '''
    generate all spanning trees for a given graph.
    Note that this is not the same as what's being performed in cycle_inflows. In cycle_inflows, we generate all valid spanning trees given a root node (a subset of this function's output).
    Here, we generate all valid spanning trees given _any_ root node. That is, if there is one or more valid roots of the tree, its spanning tree will be generated.
    Passing this set of spanning trees to root_tree_weights allows for the generation of the partition function (denominator of the MTT).
    
    Args:
        graph (nx.Graph): undirected networkx graph showing connectivity (including absorption arrows).
        absorption_edge_list (list, optional): list of edges that represent absorption processes.
    Returns:
        list: list of spanning trees (nx.Graph) for the given graph.
    
    '''
    span_trees_graph = [i for i in nx.SpanningTreeIterator(graph)]
    for tree_index, tree in enumerate(span_trees_graph):
        valid_nodes = []
        for node in tree.nodes:
            if is_valid_root(tree, node, absorption_edge_list=absorption_edge_list):
                valid_nodes.append(node)
        if len(valid_nodes) == 0:
            span_trees_graph[tree_index] = np.nan
    return [g for g in span_trees_graph if type(g) != float]



def cycle_flux_fn(turnover_cycles, futile_cycles, reaction_graph, absorption_edge_list=None, max_ss_rate_fn=None, weight_mapping_fn=None, **params):
    '''
    calculate the cycle flux given two sets of biochemical cycles (called turnover_cycles, futile_cycles here).
    The output (total flux attributable to turnover_cycles, total flux attributable to futile_cycles) will be used as input to BoundaryExplorer.
    
    Args:
        turnover_cycles (list of list of int): list of list of nodes. each constituent list of nodes represents one biochemical cycle attributable to turnover.
        futile_cycles (list of list of int): list of list of nodes. each constituent list of nodes represents one biochemical cycle attributable to futile cycling.
        reaction_graph (nx.Graph): the undirected reaction network to be subjected to the cyclic MTT.
        weight_mapping_fn (function): function to map the reaction graph to a set of weights.
        absorption_edge_list (list, optional): list of directed absorption edges representing turnover.
    Returns:
        tuple: (v_bar, q_bar) where v_bar is the normalized turnover flux and q_bar is the normalized futile flux.
    '''
    graph_weights = weight_mapping_fn(**params)
    max_steady_turnover = max_ss_rate_fn(**params)
    # spanning_trees = generate_spanning_trees(reaction_graph, absorption_edge_list=absorption_edge_list)
    # tree_weights = [root_tree_weights(spanning_trees[i], graph_weights, absorption_edge_list=absorption_edge_list, nodes=None, allNodes=True) for i in range(len(spanning_trees))]
    #
    G = nx.DiGraph()
    for (u,v), attrs in graph_weights.items():
        G.add_edge(u, v, weight=np.float64(attrs))
    partition_function= np.sum([nx.number_of_spanning_trees(G.reverse(copy=False), weight='weight', root=root) for root in G.nodes()]) 

    turnover_flux_ls = np.zeros(len(turnover_cycles))
    futile_flux_ls = np.zeros(len(futile_cycles))

    for i, cycle in enumerate(turnover_cycles):
        cycle_flux = (cycle_coefficient(cycle, graph_weights) * cycle_inflows(cycle, reaction_graph, graph_weights, absorption_edge_list=absorption_edge_list)) / partition_function
        turnover_flux_ls[i] = cycle_flux
    for i, cycle in enumerate(futile_cycles):
        cycle_flux = (cycle_coefficient(cycle, graph_weights) * cycle_inflows(cycle, reaction_graph, graph_weights, absorption_edge_list=absorption_edge_list)) / partition_function
        futile_flux_ls[i] = cycle_flux
    v_bar = np.sum(turnover_flux_ls) / max_steady_turnover
    # q_bar = np.sum(turnover_flux_ls) / np.sum(futile_flux_ls)
    q_bar = np.sum(futile_flux_ls) / np.sum(turnover_flux_ls)
    return v_bar, q_bar



def load_simulation_settings(config_file_path):
    '''
    Load simulation settings from a JSON file.
    Args:
        config_file_path (str): The path to the JSON configuration file.
    Returns:
        dict: A dictionary containing the simulation settings.
    '''
    with open(config_file_path, "r") as f:
        return json.load(f)


def build_rate_fn_kwargs(simulation_settings):
    '''
    Build a dictionary of keyword arguments for the rate function based on the simulation settings.
    Args:
        simulation_settings (dict): A dictionary containing the simulation settings.
    Returns:
        dict: A dictionary of keyword arguments for the rate function.'''
    return {
        "cycles_1": simulation_settings.get("cycles_1", None),
        "cycles_2": simulation_settings.get("cycles_2", None),
        "absorption_edge_list": simulation_settings.get("absorption_edge_list", None),
        "max_ss_rate_fn": simulation_settings.get("max_ss_rate_fn", None),
        "weight_mapping_fn": simulation_settings.get("weight_mapping_fn", None),
        "osc_midpoint_fn": simulation_settings.get("osc_midpoint_fn", None),
        "graph_fn": simulation_settings.get("graph_fn", None),
        "alpha_32": simulation_settings.get("alpha_32", None),
        "alpha_21": simulation_settings.get("alpha_21", None),
        "alpha_54": simulation_settings.get("alpha_54", None),
        "alpha_65": simulation_settings.get("alpha_65", None),
        "alpha_21_w": simulation_settings.get("alpha_21_w", None),
        "alpha_32_w": simulation_settings.get("alpha_32_w", None),
        "alpha_43": simulation_settings.get("alpha_43", None),
        "alpha_43_w": simulation_settings.get("alpha_43_w", None),
        "alpha_54_w": simulation_settings.get("alpha_54_w", None),
        "alpha_65_w": simulation_settings.get("alpha_65_w", None),
        "alpha_87": simulation_settings.get("alpha_87", None),
        "alpha_76": simulation_settings.get("alpha_76", None),
        "alpha_87_w": simulation_settings.get("alpha_87_w", None),
        "alpha_76_w": simulation_settings.get("alpha_76_w", None),
        "dG_1": simulation_settings.get("dG_1", None),
        "dG_2": simulation_settings.get("dG_2", None),
        "dG_1_w": simulation_settings.get("dG_1_w", None),
        "dG_2_w": simulation_settings.get("dG_2_w", None),
        "dG_3": simulation_settings.get("dG_3", None),
        "dG_3_w": simulation_settings.get("dG_3_w", None),
        "gamma": simulation_settings.get("gamma", None),
        "k_bT": simulation_settings.get("k_bT", None),
    }


def make_rate_fn(hyperparams):
    '''
    Create a rate function based on the provided hyperparameters.
    Args:
        hyperparams (dict): A dictionary containing the hyperparameters for the rate function.
    Returns:
        function: A rate function that takes a list of parameters and returns the rate.
    '''
    weight_mapping_fn = hyperparams["weight_mapping_fn"]
    max_ss_rate_fn = hyperparams["max_ss_rate_fn"]
    osc_midpoint_fn = hyperparams["osc_midpoint_fn"]
    graph_fn = hyperparams["graph_fn"]

    def rate_fn(params):
        if len(params) == 3:
            alpha_14, alpha_41, amplitude = params
            amplitude = np.log10(amplitude)
            osc_midpoint = osc_midpoint_fn(
                alpha_14=None,
                alpha_41=None,
                epsilon_1=None,
                epsilon_2=None,
                **hyperparams,
            )
            epsilon_1 = osc_midpoint + amplitude
            epsilon_2 = osc_midpoint - amplitude
        elif len(params) == 4:
            alpha_14, alpha_41, epsilon_1, epsilon_2 = params
        else:
            raise ValueError("params must have length 3 or 4")

        return cycle_flux_fn(
            hyperparams["cycles_1"],
            hyperparams["cycles_2"],
            graph_fn(),
            absorption_edge_list=hyperparams["absorption_edge_list"],
            max_ss_rate_fn=max_ss_rate_fn,
            weight_mapping_fn=weight_mapping_fn,
            osc_midpoint_fn=osc_midpoint_fn,
            alpha_14=alpha_14,
            alpha_41=alpha_41,
            alpha_32=hyperparams["alpha_32"],
            alpha_21=hyperparams["alpha_21"],
            alpha_54=hyperparams["alpha_54"],
            alpha_65=hyperparams["alpha_65"],
            alpha_76=hyperparams["alpha_76"],
            alpha_87=hyperparams["alpha_87"],
            alpha_21_w=hyperparams["alpha_21_w"],
            alpha_32_w=hyperparams["alpha_32_w"],
            alpha_43=hyperparams["alpha_43"],
            alpha_43_w=hyperparams["alpha_43_w"],
            alpha_54_w=hyperparams["alpha_54_w"],
            alpha_65_w=hyperparams["alpha_65_w"],
            alpha_76_w=hyperparams["alpha_76_w"],
            alpha_87_w=hyperparams["alpha_87_w"],
            dG_1=hyperparams["dG_1"],
            dG_2=hyperparams["dG_2"],
            dG_1_w=hyperparams["dG_1_w"],
            dG_2_w=hyperparams["dG_2_w"],
            dG_3=hyperparams["dG_3"],
            dG_3_w=hyperparams["dG_3_w"],
            gamma=hyperparams["gamma"],
            epsilon_1=epsilon_1,
            epsilon_2=epsilon_2,
            k_bT=hyperparams["k_bT"],
        )

    return rate_fn


def run_dycat_model(config_file_path):
    '''
    Run the dynamic catalysis model based on the provided configuration file.
    Args:
        config_file_path (str): The path to the JSON configuration file.
    Returns:
        str: The path to the output folder where results are saved.
    '''
    simulation_settings = load_simulation_settings(config_file_path)

    outfolder = simulation_settings.get("folder_save")
    outfolder_final = f"{outfolder}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    name_save = simulation_settings.get("name_save")
    name = f"{name_save}_last"
    mod = importlib.import_module("utils")
    simulation_settings["weight_mapping_fn"] = getattr(mod, simulation_settings["weight_mapping_fn"])
    simulation_settings["graph_fn"] = getattr(mod, simulation_settings["graph_fn"])
    if simulation_settings.get("max_ss_rate_fn") is None and simulation_settings.get("osc_midpoint_fn") is None:
        def dummy_max_rate_fn(**params):
            return 1
        def dummy_osc_midpoint_fn(**params):
            return 0
        
        simulation_settings["max_ss_rate_fn"] = dummy_max_rate_fn
        simulation_settings["osc_midpoint_fn"] = dummy_osc_midpoint_fn
        dummy_rate_fn = make_rate_fn(build_rate_fn_kwargs(simulation_settings))
        epsilon_list = np.linspace(-10, 10, 100)
        J_tot_list_cycle_r = []
        J_tot_list_cycle_w = []
        for epsilon in epsilon_list:    
            J_tot_r, J_tot_w = dummy_rate_fn([1e-11, 1e-11, epsilon, epsilon])
            J_tot_list_cycle_r.append(J_tot_r)
            J_tot_list_cycle_w.append(J_tot_w)
        max_index = np.argmax(np.array(J_tot_list_cycle_r))
        max_epsilon = epsilon_list[max_index]
        max_rate = J_tot_list_cycle_r[max_index]

        def osc_midpoint_fn(**params):
            return max_epsilon
        def max_rate_fn(**params):
            return max_rate
        simulation_settings["max_ss_rate_fn"] = max_rate_fn
        simulation_settings["osc_midpoint_fn"] = osc_midpoint_fn
    else:
        assert simulation_settings.get("max_ss_rate_fn") is not None, "max_ss_rate_fn must be specified in the config file"
        assert simulation_settings.get("osc_midpoint_fn") is not None, "osc_midpoint_fn must be specified in the config file"
        simulation_settings["max_ss_rate_fn"] = getattr(mod, simulation_settings["max_ss_rate_fn"])
        simulation_settings["osc_midpoint_fn"] = getattr(mod, simulation_settings["osc_midpoint_fn"])

    assert simulation_settings.get("col_ar_min") is not None, "col_ar_min must be specified in the config file"
    assert simulation_settings.get("col_ar_max") is not None, "col_ar_max must be specified in the config file"
    assert simulation_settings.get("row_ar_min") is not None, "row_ar_min must be specified in the config file"
    assert simulation_settings.get("row_ar_max") is not None, "row_ar_max must be specified in the config file"
    assert simulation_settings.get("n_grid") is not None, "n_grid must be specified in the config file"
    assert simulation_settings.get("pars_limit_min") is not None, "pars_limit_min must be specified in the config file"
    assert simulation_settings.get("pars_limit_max") is not None, "pars_limit_max must be specified in the config file"
    assert simulation_settings.get("npars") is not None, "npars must be specified in the config file"
    assert simulation_settings.get("weight_mapping_fn") is not None, "weight_mapping_fn must be specified in the config file"
    # assert simulation_settings.get("max_ss_rate_fn") is not None, "max_ss_rate_fn must be specified in the config file"
    # assert simulation_settings.get("osc_midpoint_fn") is not None, "osc_midpoint_fn must be specified in the config file"
    assert simulation_settings.get("graph_fn") is not None, "graph_fn must be specified in the config file"

    stp_ar = np.linspace(
        simulation_settings.get("col_ar_min", None),
        simulation_settings.get("col_ar_max", None),
        simulation_settings.get("n_grid", None),
    )
    pos_ar = np.linspace(
        simulation_settings.get("row_ar_min", None),
        simulation_settings.get("row_ar_max", None),
        simulation_settings.get("n_grid", None),
    )

    defaultsettings = {
        "pars_limit": [
            simulation_settings.get("pars_limit_min", None),
            simulation_settings.get("pars_limit_max", None),
        ],
        "row_ar": pos_ar,
        "col_ar": stp_ar,
        "seed": simulation_settings.get("seed", None),
        "mat": simulation_settings.get("mat", None),
        "mat_pars": simulation_settings.get("mat_pars", None),
        "npars": simulation_settings.get("npars", None)
    }

    defaultargs = {
        "niters": simulation_settings.get("niters", None),
        "niters_conv": simulation_settings.get("niters_conv_points", None),
        "niters_conv_points": simulation_settings.get("niters_conv_points", None),
        "niters_save": simulation_settings.get("niters_save", None),
        "prob_par": simulation_settings.get("prob_par", None),
        "prob_replace": simulation_settings.get("prob_replace", None),
        "extr_uniform": simulation_settings["extr_uniform"],
        "L_project": simulation_settings["L_project"],
        "plotting": simulation_settings["plotting"],
        "verbose": simulation_settings["verbose"],
        "dofirstmutate": simulation_settings["dofirstmutate"],
        "dopullcentroids": simulation_settings["dopullcentroids"],
        "dopulltangents": simulation_settings["dopulltangents"],
        "folder_save": outfolder_final,
        "name_save": name_save,
    }

    


    simulation_settings["npars"] = simulation_settings.get("npars", None) 
    os.makedirs(outfolder_final, exist_ok=True)

    settings = defaultsettings.copy()
    settings["compute_x_y_f"] = make_rate_fn(build_rate_fn_kwargs(simulation_settings))

    BE = BF.BoundaryExplorer(**settings)
    BE.get_initial_points(10)
    BE.extend_boundary(**defaultargs)
    np.save(os.path.join(outfolder_final, f"mat_{name}.npy"), BE.mat)
    np.save(os.path.join(outfolder_final, f"mat_pars_{name}.npy"), BE.mat_pars)



    

    cycles = simulation_settings["cycles_1"] + simulation_settings["cycles_2"]
    sub_cycle_fluxes = np.zeros((len(cycles), BE.mat_pars.shape[0] * BE.mat_pars.shape[1]))
    print(f"Calculating sub-cycle fluxes for {BE.mat_pars.shape[0] * BE.mat_pars.shape[1]} points... ")

    total_evaluations = (
        len(cycles) * BE.mat_pars.shape[0] * BE.mat_pars.shape[1]
    )

    with tqdm.tqdm(
        total=total_evaluations,
        desc="Calculating subcycle fluxes",
        unit="evaluation",
    ) as progress:
        for k, cycle in enumerate(cycles):
            simulation_settings["cycles_1"] = [cycle]
            simulation_settings["cycles_2"] = []
            subcycle_rate_fn = make_rate_fn(build_rate_fn_kwargs(simulation_settings))

            for i in range(BE.mat_pars.shape[0]):
                for j in range(BE.mat_pars.shape[1]):
                    params = BE.mat_pars[i, j, :]
                    sub_cycle_fluxes[k, i + j] = subcycle_rate_fn(params)[0]
                    progress.update(1)

    np.save(os.path.join(outfolder_final, f"sub_cycle_fluxes_{name}.npy"), sub_cycle_fluxes)
    simulation_settings = load_simulation_settings(config_file_path)
    with open(os.path.join(outfolder_final, f"{name_save}_config.json"), "w") as f:
            json.dump(simulation_settings, f)
    return outfolder_final