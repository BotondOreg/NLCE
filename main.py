# import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from scipy import sparse
from scipy.sparse import linalg
import itertools
import datetime

"""
TODO:
 - Calculate results for every temperature, not just one
 - Separate elements to separate packages
 - Save the clusters and embeddings so that they can be reused
 - Add docstrings to functions
 - Generate requirements and wrap up the whole package
 - Diagonalize the Hamiltonian block diagonally. Spin up and spin down numbers are conserved.
 - Put things into arrays while taking care of lengths and indices.
"""


def get_connected_subgraphs_multiplicities(graph, order):
    nodes = graph.nodes

    unique_subgraphs = []

    for selected_nodes in itertools.combinations(nodes, order):
        current_subgraph = nx.subgraph(graph, selected_nodes)
        if not nx.is_connected(current_subgraph):
            continue
        is_new = True
        for index, (us, multi) in enumerate(unique_subgraphs):
            if nx.algorithms.isomorphism.is_isomorphic(us, current_subgraph):
                is_new = False
                unique_subgraphs[index] = (us, multi+1)
        if is_new:
            unique_subgraphs.append((current_subgraph, 1))

    return unique_subgraphs


def create_clusters(grid_size, max_order):

    global_connectivity_matrix = nx.grid_2d_graph(grid_size, grid_size)

    # Subgraph basis and multiplicities in the original matrix
    lattice_subgraphs = []  # (order, subgraph-index) -> basis subgraph
    lattice_embedding = []  # (order, subgraph-index) -> multiplicity
    subcluster_embedding_in_cluster = []  # (order, subgraph-index, sub-order, subcluster-index) -> sub-multiplicity
    for current_order in range(1, max_order+1):
        # Get subgraphs and their multiplicities in the big matrix
        # The 'basis' for the expansion
        unique_n_sg_m = get_connected_subgraphs_multiplicities(global_connectivity_matrix, current_order)
        lattice_subgraphs.append([sg for (sg, m) in unique_n_sg_m])
        lattice_embedding.append([m for (us, m) in unique_n_sg_m])

        # Get the embeddings of the smaller clusters in the new clusters
        n_order_sg_embedding = []
        for sg in lattice_subgraphs[current_order-1]:
            sg_embedding = []
            # go until (current_order-1) included
            for sub_order in range(1, current_order):
                sub_multiplicity = np.zeros((len(lattice_subgraphs[sub_order-1]),), int)
                unique_subclusters_multiplicities = get_connected_subgraphs_multiplicities(sg, sub_order)
                # Match the subclusters to the basis
                for sc, sc_multiplicity in unique_subclusters_multiplicities:
                    for bg_index, basis_graph in enumerate(lattice_subgraphs[sub_order-1]):
                        if nx.algorithms.isomorphism.is_isomorphic(sc, basis_graph):
                            sub_multiplicity[bg_index] = sc_multiplicity
                            break
                sg_embedding.append(sub_multiplicity)
            n_order_sg_embedding.append(sg_embedding)
        subcluster_embedding_in_cluster.append(n_order_sg_embedding)

    # adjacency matrices
    adjacency_matrices = [[nx.adjacency_matrix(graph, dtype=int).todense() for graph in lattice_subgraphs[order-1]]
                          for order in range(1, max_order+1)]  # (order, subgraph-index) -> adjacency-matrix

    return adjacency_matrices, lattice_embedding, subcluster_embedding_in_cluster


def atom_number(state_number):
    # The number of atoms in a state represented by state_number

    if type(state_number) == int:
        binary = np.binary_repr(state_number)
        return str.count(binary, '1')
    elif type(state_number) == list or type(state_number) == tuple:
        state_number = np.array(state_number)
    elif type(state_number) != np.ndarray:
        print("Unknown input type!")

    flat_input = state_number.flatten()
    number = np.char.count(list(map(np.binary_repr, flat_input)), '1')
    number = number.reshape(state_number.shape)

    return number


def generate_hopping(adjacency, spin='both'):
    # Create a sparse array

    (from_list, to_list) = np.nonzero(adjacency)

    n_sites = adjacency.shape[0]
    matrix_size = 4**n_sites

    hopping = scipy.sparse.csc_matrix((matrix_size, matrix_size))

    for index in range(len(from_list)):
        from_site = from_list[index]
        to_site = to_list[index]

        states_without_ij = np.arange(2**(2*n_sites-2), dtype=int)

        # Spin-up hopping
        if spin=='up' or spin=='both':
            fermionic_sign = np.ones((2 ** (2 * n_sites - 2),))  # Make the hamiltonian floating point
            if from_site < to_site:
                # insert placeholder for from site spin up (2*4**from_site)
                (div, mod) = np.divmod(states_without_ij, 2 * 4 ** from_site)
                empty_states = mod + 4 ** (from_site + 1) * div
                fermionic_sign *= (-1) ** atom_number(div)
                # insert placeholder for to site spin up (2*4**to_site)
                (div, mod) = np.divmod(empty_states, 2 * 4 ** to_site)
                empty_states = mod + 4 ** (to_site + 1) * div
                fermionic_sign *= (-1) ** atom_number(div)
            else:  # Assuming to_site != from_site
                # insert placeholder for to site spin up (2*4**to_site)
                (div, mod) = np.divmod(states_without_ij, 2 * 4 ** to_site)
                empty_states = mod + 4 ** (to_site + 1) * div
                fermionic_sign *= (-1) ** atom_number(div)
                # insert placeholder for from site spin up (2*4**from_site)
                (div, mod) = np.divmod(empty_states, 2 * 4 ** from_site)
                empty_states = mod + 4 ** (from_site + 1) * div
                fermionic_sign *= (-1) ** atom_number(div)
            from_states = empty_states + 2 * 4 ** from_site
            to_states = empty_states + 2 * 4 ** to_site

            hopping += sparse.coo_matrix((fermionic_sign, (from_states, to_states)),
                                         shape=(matrix_size, matrix_size))

        # Spin-down hopping
        if spin == 'dn' or spin == 'both':
            fermionic_sign = np.ones((2 ** (2 * n_sites - 2),))  # Make the hamiltonian floating point
            if from_site < to_site:
                # insert placeholder for from site spin down (4**from_site)
                (div, mod) = np.divmod(states_without_ij, 4 ** from_site)
                empty_states = mod + 2 * 4 ** from_site * div
                fermionic_sign *= (-1) ** atom_number(div)
                # insert placeholder for to site spin up (4**to_site)
                (div, mod) = np.divmod(empty_states, 4 ** to_site)
                empty_states = mod + 2 * 4 ** to_site * div
                fermionic_sign *= (-1) ** atom_number(div)
            else:  # Assuming to_site != from_site
                # insert placeholder for to site spin up (4**to_site)
                (div, mod) = np.divmod(states_without_ij, 4 ** to_site)
                empty_states = mod + 2 * 4 ** to_site * div
                fermionic_sign *= (-1) ** atom_number(div)
                # insert placeholder for from site spin up (4**from_site)
                (div, mod) = np.divmod(empty_states, 4 ** from_site)
                empty_states = mod + 2 * 4 ** from_site * div
                fermionic_sign *= (-1) ** atom_number(div)
            from_states = empty_states + 4 ** (from_site)
            to_states = empty_states + 4 ** (to_site)

            hopping += sparse.coo_matrix((fermionic_sign, (from_states, to_states)),
                                         shape=(matrix_size, matrix_size))

    return hopping


def generate_dh_correlator(adjacency):
    # Create a sparse array

    n_sites = adjacency.shape[0]
    matrix_size = 4 ** n_sites

    correlator = scipy.sparse.csc_matrix((matrix_size, matrix_size))

    for from_site in range(n_sites):
        # Apparently, the dimensionality of adjacency[from_site, :] depends on the numpy version
        (_,to_list) = np.nonzero(np.atleast_2d(adjacency[from_site, :]))

        for to_site in to_list:
            states_without_ij = np.arange(2 ** (2 * (n_sites - 2)), dtype=int)
            if from_site < to_site:
                # insert placeholder for from site
                (div, mod) = np.divmod(states_without_ij, 4 ** from_site)
                empty_states = mod + 4 ** (from_site + 1) * div
                # insert placeholder for to site
                (div, mod) = np.divmod(empty_states, 4 ** to_site)
                empty_states = mod + 4 ** (to_site + 1) * div
            else:  # Assuming to_site != from_site
                # insert placeholder for to site spin up
                (div, mod) = np.divmod(states_without_ij, 4 ** to_site)
                empty_states = mod + 4 ** (to_site + 1) * div
                # insert placeholder for from site spin up (2*4**from_site)
                (div, mod) = np.divmod(empty_states, 4 ** from_site)
                empty_states = mod + 4 ** (from_site + 1) * div

            # doublon-hole
            dh_states = empty_states + 3 * 4 ** from_site
            correlator += sparse.coo_matrix(
                (np.ones((2 ** (2 * (n_sites - 2)),)) / len(to_list), (dh_states, dh_states)),
                shape=(matrix_size, matrix_size))

    return correlator


def generate_mm_correlator(adjacency):
    # Create a sparse array

    n_sites = adjacency.shape[0]
    matrix_size = 4**n_sites

    correlator = scipy.sparse.csc_matrix((matrix_size, matrix_size))

    for from_site in range(n_sites):
        # Apparently, the dimensionality of adjacency[from_site, :] depends on the numpy version
        (_, to_list) = np.nonzero(np.atleast_2d(adjacency[from_site, :]))

        for to_site in to_list:
            states_without_ij = np.arange(2**(2*(n_sites-2)), dtype=int)
            if from_site < to_site:
                # insert placeholder for from site
                (div, mod) = np.divmod(states_without_ij, 4 ** from_site)
                empty_states = mod + 4 ** (from_site + 1) * div
                # insert placeholder for to site
                (div, mod) = np.divmod(empty_states, 4 ** to_site)
                empty_states = mod + 4 ** (to_site + 1) * div
            else: # Assuming to_site != from_site
                # insert placeholder for to site spin up
                (div, mod) = np.divmod(states_without_ij, 4 ** to_site)
                empty_states = mod + 4 ** (to_site + 1) * div
                # insert placeholder for from site spin up (2*4**from_site)
                (div, mod) = np.divmod(empty_states, 4 ** from_site)
                empty_states = mod + 4 ** (from_site + 1) * div

            # up-up
            up_up_states = empty_states + 2 * 4 ** from_site + 2 * 4 ** to_site
            correlator += sparse.coo_matrix((np.ones((2**(2*(n_sites-2)),))/len(to_list), (up_up_states, up_up_states)),
                                    shape=(matrix_size, matrix_size))
            # dn-dn
            dn_dn_states = empty_states + 4 ** from_site + 4 ** to_site
            correlator += sparse.coo_matrix(
                (np.ones((2 ** (2 * (n_sites - 2)),)) / len(to_list), (dn_dn_states, dn_dn_states)),
                shape=(matrix_size, matrix_size))
            # up-dn
            up_dn_states = empty_states + 2 * 4 ** from_site + 4 ** to_site
            correlator += sparse.coo_matrix(
                (-np.ones((2 ** (2 * (n_sites - 2)),)) / len(to_list), (up_dn_states, up_dn_states)),
                shape=(matrix_size, matrix_size))
            # dn-up
            dn_up_states = empty_states + 4 ** from_site + 2 * 4 ** to_site
            correlator += sparse.coo_matrix(
                (-np.ones((2 ** (2 * (n_sites - 2)),)) / len(to_list), (dn_up_states, dn_up_states)),
                shape=(matrix_size, matrix_size))

    return correlator


def generate_doublon(adjacency):
    n_sites = adjacency.shape[0]
    matrix_size = 4 ** n_sites

    doublon = scipy.sparse.csc_matrix((matrix_size, matrix_size))

    states = np.arange(matrix_size)

    for site in range(n_sites):
        # cut the sites above site
        rem = np.remainder(states, 4**(site+1))
        # check if site is occupied (if and only if 4**site and 2*4**site bit are both 1)
        _d = np.floor_divide(rem, 3*4**site)

        doublon += sparse.coo_matrix((_d, (states, states)), shape=(matrix_size, matrix_size))

    return doublon


def generate_density(adjacency, spin='both'):
    n_sites = adjacency.shape[0]
    matrix_size = 4 ** n_sites

    density = sparse.csc_matrix((matrix_size, matrix_size))
    states = np.arange(matrix_size)

    for site in range(n_sites):
        if spin=='up' or spin=='both':
            _n = np.floor_divide(np.remainder(states, 4**(site+1)), 2*4**site)
            density += sparse.coo_matrix((_n, (states, states)), shape=(matrix_size, matrix_size))
        if spin=='dn' or spin=='both':
            _n = np.floor_divide(np.remainder(states, 2*4**site), 4**site)
            density += sparse.coo_matrix((_n, (states, states)), shape=(matrix_size, matrix_size))

    return density


def generate_hamiltonian(adjacency, u_per_t, mu_per_t):
    # Hamiltonian for a graph with N sites is a 2^(2*N)-by-2^(2*N) matrix
    # The index of a state is the sum of the following terms
    # If there is a spin down on site i, that contributes 2^i
    # a spin up contributes 2^(i+1)
    # i goes from 0 to (N-1) included

    # for each edge in the connectivity matrix (say including direction)
    # add the tunneling of each spin up and spin down

    ham = - generate_hopping(adjacency) + u_per_t*generate_doublon(adjacency) - mu_per_t*generate_density(adjacency)

    return ham


def generate_hamiltonian_list(adjacency_matrix_list, u_per_t, mu_per_t):
    return [[generate_hamiltonian(graph, u_per_t, mu_per_t) for graph in graph_list]
                for graph_list in adjacency_matrix_list]


def generate_density_list(adjacency_matrix_list):
    return [[generate_density(graph) for graph in graph_list] for graph_list in adjacency_matrix_list]


def generate_doublon_list(adjacency_matrix_list):
    return [[generate_doublon(graph) for graph in graph_list] for graph_list in adjacency_matrix_list]

def generate_dh_list(adjacency_matrix_list):
    return [[generate_dh_correlator(graph) for graph in graph_list] for graph_list in adjacency_matrix_list]

def generate_mm_list(adjacency_matrix_list):
    return [[generate_mm_correlator(graph) for graph in graph_list] for graph_list in adjacency_matrix_list]


def generate_cluster_expansion(observables, hamiltonians, temperature_per_t):
    beta = 1.0/temperature_per_t

    expectation_value = []

    for order in range(1, len(hamiltonians)+1):
        _exp_val = []
        for cluster_index in range(len(hamiltonians[order-1])):
            _a = observables[order-1][cluster_index]
            _mbh = -beta*hamiltonians[order-1][cluster_index]

            value = np.sum(linalg.expm_multiply(_mbh, _a).diagonal()) / np.sum(linalg.expm(_mbh).diagonal())

            for sub_order in range(1, order):
                for subcluster_index in range(len(subcluster_embedding_in_cluster[order-1][cluster_index][sub_order-1])):
                    value -= subcluster_embedding_in_cluster[order-1][cluster_index][sub_order-1][subcluster_index]\
                        *expectation_value[sub_order-1][subcluster_index]

            _exp_val.append(value)
        expectation_value.append(_exp_val)

    return expectation_value





if __name__ == "__main__":
    print(datetime.datetime.now()) # 1
    ## Global properties
    l_grid = 7
    n_sites = l_grid**2
    max_order = 4

    u_per_t = 4
    mu_per_t = 1
    temperature_per_t = 1.0
    t_list = np.array([0.8])
    print(t_list)
    generate_clusters = True

    ## Run the code
    # Generate clusters
    # if generate_clusters:
    adjacency_matrices, lattice_embedding, subcluster_embedding_in_cluster = create_clusters(l_grid, max_order)
    print(datetime.datetime.now()) # 2
    densities = generate_density_list(adjacency_matrices)
    doublons = generate_doublon_list(adjacency_matrices)
    dh_corrs = generate_dh_list(adjacency_matrices)
    mm_corrs = generate_mm_list(adjacency_matrices)
    hamiltonians = generate_hamiltonian_list(adjacency_matrices, u_per_t, mu_per_t)

    print(datetime.datetime.now()) # 3
    n_list = []
    d_list = []
    dh_list = []
    mm_list = []
    for _t in t_list:
        cluster_expansion = generate_cluster_expansion(doublons, hamiltonians, _t)
        cluster_expansion_with_multiplicity = [[cluster_expansion[order-1][i]*lattice_embedding[order-1][i]
                                                for i in range(len(lattice_embedding[order-1]))]
                                                   for order in range(1, max_order+1)]
        # print([np.sum(cluster_expansion_with_multiplicity[order-1]) for order in range(1, max_order+1)])
        d_list.append(np.sum([np.sum(cluster_expansion_with_multiplicity[order-1]) for order in range(1, max_order+1)]))

        cluster_expansion = generate_cluster_expansion(densities, hamiltonians, _t)
        cluster_expansion_with_multiplicity = [[cluster_expansion[order - 1][i] * lattice_embedding[order - 1][i]
                                                for i in range(len(lattice_embedding[order - 1]))]
                                               for order in range(1, max_order + 1)]
        # # print([np.sum(cluster_expansion_with_multiplicity[order-1]) for order in range(1, max_order+1)])
        n_list.append(
            np.sum([np.sum(cluster_expansion_with_multiplicity[order - 1]) for order in range(1, max_order + 1)]))

        cluster_expansion = generate_cluster_expansion(dh_corrs, hamiltonians, _t)
        cluster_expansion_with_multiplicity = [[cluster_expansion[order - 1][i] * lattice_embedding[order - 1][i]
                                                for i in range(len(lattice_embedding[order - 1]))]
                                               for order in range(1, max_order + 1)]
        # print([np.sum(cluster_expansion_with_multiplicity[order-1]) for order in range(1, max_order+1)])
        dh_list.append(
            np.sum([np.sum(cluster_expansion_with_multiplicity[order - 1]) for order in range(1, max_order + 1)]))

        cluster_expansion = generate_cluster_expansion(mm_corrs, hamiltonians, _t)
        cluster_expansion_with_multiplicity = [[cluster_expansion[order - 1][i] * lattice_embedding[order - 1][i]
                                                for i in range(len(lattice_embedding[order - 1]))]
                                               for order in range(1, max_order + 1)]
        # print([np.sum(cluster_expansion_with_multiplicity[order-1]) for order in range(1, max_order+1)])
        mm_list.append(
            np.sum([np.sum(cluster_expansion_with_multiplicity[order - 1]) for order in range(1, max_order + 1)]))

    d_list = np.array(d_list)/n_sites
    n_list = np.array([n_list]) / n_sites
    dh_list = np.array(dh_list) / n_sites
    mm_list = np.array(mm_list) / n_sites

    np.savetxt('result.csv', (t_list, n_list, d_list, dh_list, mm_list), delimiter=',')
    print(datetime.datetime.now()) # 4
    # plt.plot(mu_list, n_list, 'o-')
    # plt.xlabel('mu / t')
    # plt.ylabel('n')
    # plt.title('U / t = {:g}; T / t = {:g}'.format(u_per_t, temperature_per_t))
    # plt.show()




