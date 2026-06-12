# This file adapts Torque Clustering code from Cognet-74/TorqueClusteringPy:
# https://github.com/Cognet-74/TorqueClusteringPy
# Upstream license: CC BY-NC-SA 4.0. See THIRD_PARTY_NOTICES.md.

from typing import Tuple, Union, Optional, List, Dict, Any
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import networkx as nx
import matplotlib.pyplot as plt
from utils.ps2psdist import ps2psdist
from utils.Updateljmat_weighted import Updateljmat_weighted
from utils.uniqueZ import uniqueZ
from utils.Nab_dec import Nab_dec
from utils.Nab_dec_quantile import Nab_dec_quantile
from utils.Final_label import Final_label

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

def TorqueClustering_weighted(
    ALL_DM: Union[np.ndarray, scipy.sparse.spmatrix],
    node_masses: np.ndarray,
    K: int = 0,
    isnoise: bool = False,
    isfig: bool = False,
    matlab_compatibility: bool = True,
    use_std_adjustment: bool = True,
    adjustment_factor: float = 0.5,
    cluster_selection: str = "paper_eq4",
    num_band: float = 0.25,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Multi-layer Torque Clustering on node graph (e.g., CAplus/IDAT nodes) with
    community masses defined by node_masses.

    This is a variant of the original TorqueClustering which:
      - assumes ALL_DM is a node-to-node distance matrix
      - computes community mass as sum(node_masses[idx]) over member indices
      - keeps the multi-layer merging behavior of the original algorithm

    Args:
        ALL_DM: Distance matrix between nodes (n x n).
        node_masses: 1D array of per-node masses (e.g., CountNode or cluster_counts_).
        K: Desired number of clusters (0 for automatic selection).
        isnoise: Enable noise detection.
        isfig: Plot decision graph.
        matlab_compatibility: Mimic MATLAB numeric behavior.
        use_std_adjustment, adjustment_factor: Passed to Nab_dec when cluster_selection="gap".
        cluster_selection: "paper_eq4", "gap", or "quantile".
        verbose: Print layer-level progress when True.

    Returns:
        Idx, Idx_with_noise, cutnum, cutlink_ori, p, firstlayer_loc_onsortp,
        mass, R, cutlinkpower_all_np, diagnostics
    """
    if matlab_compatibility:
        old_settings = np.seterr(all='ignore')

    if ALL_DM is None:
        raise ValueError('Distance Matrix is required.')

    if not scipy.sparse.issparse(ALL_DM) and not isinstance(ALL_DM, np.ndarray):
        raise ValueError('Distance matrix must be a NumPy array or scipy sparse matrix')

    if ALL_DM.shape[0] != ALL_DM.shape[1]:
        raise ValueError('Distance matrix must be square')

    if isinstance(ALL_DM, np.ndarray) and ALL_DM.dtype != np.float64:
        ALL_DM = np.float64(ALL_DM)

    diagnostics = {
        'parameters': {
            'K': K,
            'isnoise': isnoise,
            'use_std_adjustment': use_std_adjustment,
            'adjustment_factor': adjustment_factor,
            'matlab_compatibility': matlab_compatibility,
            'cluster_selection': cluster_selection,
            'node_masses_used': node_masses is not None,
        },
        'input_matrix': {
            'shape': ALL_DM.shape,
            'is_sparse': scipy.sparse.issparse(ALL_DM),
            'dtype': str(ALL_DM.dtype)
        }
    }

    is_input_sparse = scipy.sparse.issparse(ALL_DM)

    if not is_input_sparse:
        ALL_DM_sparse = scipy.sparse.csr_matrix(ALL_DM)
    else:
        ALL_DM_sparse = ALL_DM.tocsr()

    datanum = np.int64(ALL_DM_sparse.shape[0])
    cutlinkpower_all = []
    link_adjacency_matrix = scipy.sparse.lil_matrix((datanum, datanum), dtype=np.float64)
    dataloc = np.arange(datanum, dtype=np.int64)
    community = [[dataloc[i]] for i in range(datanum)]

    inter_community_distance_matrix = ALL_DM_sparse.copy()
    community_num = np.int64(datanum)

    graph_connectivity_matrix = scipy.sparse.lil_matrix((community_num, community_num), dtype=np.float64)

    neighbor_community_indices = [None] * community_num

    for i in range(community_num):
        row = ALL_DM_sparse[i].toarray().flatten()
        row = row.astype(np.float64)
        row[i] = np.inf

        min_indices = np.argsort(row, kind='mergesort')
        min_idx = np.int64(min_indices[0])

        graph_connectivity_matrix[i, min_idx] = 1
        graph_connectivity_matrix[min_idx, i] = 1
        neighbor_community_indices[i] = min_idx

    graph_connectivity_matrix = graph_connectivity_matrix.tocsr()

    SG = nx.from_scipy_sparse_array(graph_connectivity_matrix)
    components = list(nx.connected_components(SG))
    components.sort(key=lambda c: min(c))

    BINS = np.zeros(datanum, dtype=np.int64)
    for i, component in enumerate(components):
        for node in component:
            BINS[node] = i

    current_cluster_count = len(np.unique(BINS))
    if verbose:
        print(f'The number of clusters in this layer is: {current_cluster_count}')

    link_adjacency_matrix, cutlinkpower = Updateljmat_weighted(
        link_adjacency_matrix, neighbor_community_indices,
        community, inter_community_distance_matrix,
        graph_connectivity_matrix, ALL_DM_sparse,
        node_masses=node_masses,
    )

    if cutlinkpower is not None and cutlinkpower.size > 0:
        cutlinkpower = np.float64(cutlinkpower)
        if len(cutlinkpower.shape) == 1:
            cutlinkpower = cutlinkpower.reshape(1, -1)

    cutlinkpower, link_adjacency_matrix = uniqueZ(cutlinkpower, link_adjacency_matrix)

    firstlayer_conn_num = 0
    if cutlinkpower is not None and cutlinkpower.size > 0:
        firstlayer_conn_num = np.int64(cutlinkpower.shape[0])
        cutlinkpower_all.append(cutlinkpower)

    previous_unique_bins = 0
    max_iterations = datanum * 2
    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            if verbose:
                print("Warning: Maximum iterations reached. Breaking loop.")
            break

        Idx_tmp = BINS.copy()
        uni_Idx = np.unique(Idx_tmp)
        num_uni_Idx = np.int64(len(uni_Idx))

        community_new = [None] * num_uni_Idx
        for i in range(num_uni_Idx):
            uniloc = (uni_Idx[i] == Idx_tmp)
            current_community = []
            indices = np.where(uniloc)[0]
            for idx in indices:
                current_community.extend(community[idx])
            community_new[i] = current_community

        community = community_new
        community_num = np.int64(len(community))

        inter_community_distance_matrix = scipy.sparse.lil_matrix((community_num, community_num), dtype=np.float64)

        for i in range(community_num):
            for j in range(community_num):
                if i != j:
                    dist = np.float64(ps2psdist(community[i], community[j], ALL_DM_sparse))
                    inter_community_distance_matrix[i, j] = dist

        inter_community_distance_matrix = inter_community_distance_matrix.tocsr()
        
        if node_masses is not None:
            community_masses = np.array(
                [np.sum(node_masses[np.asarray(comm, dtype=int)]) for comm in community],
                dtype=np.float64
            )
        else:
            # If node masses are not provided, use the number of nodes as mass (original behavior)
            community_masses = np.array(
                [len(comm) for comm in community],
                dtype=np.float64
            )

        graph_connectivity_matrix = scipy.sparse.lil_matrix((community_num, community_num), dtype=np.float64)
        neighbor_community_indices = [None] * community_num

        for i in range(community_num):
            row = inter_community_distance_matrix[i].toarray().flatten()
            row = row.astype(np.float64)
            row[i] = np.inf

            sorted_indices = np.argsort(row, kind='mergesort')

            for j in sorted_indices:
                if j != i:
                    graph_connectivity_matrix[i, j] = 1
                    graph_connectivity_matrix[j, i] = 1
                    neighbor_community_indices[i] = j
                    break

        graph_connectivity_matrix = graph_connectivity_matrix.tocsr()

        SG = nx.from_scipy_sparse_array(graph_connectivity_matrix)
        components = list(nx.connected_components(SG))
        components.sort(key=lambda c: min(c))

        BINS = np.zeros(community_num, dtype=np.int64)
        for i, component in enumerate(components):
            for node in component:
                BINS[node] = i

        current_cluster_count = len(np.unique(BINS))
        if verbose:
            print(f'The number of clusters in this layer is: {current_cluster_count}')

        link_adjacency_matrix, cutlinkpower = Updateljmat_weighted(
            link_adjacency_matrix, neighbor_community_indices,
            community, inter_community_distance_matrix,
            graph_connectivity_matrix, ALL_DM_sparse,
            node_masses=node_masses,
        )

        if cutlinkpower is not None and cutlinkpower.size > 0:
            cutlinkpower = np.float64(cutlinkpower)
            if len(cutlinkpower.shape) == 1:
                cutlinkpower = cutlinkpower.reshape(1, -1)

        cutlinkpower, link_adjacency_matrix = uniqueZ(cutlinkpower, link_adjacency_matrix)

        if cutlinkpower is not None and cutlinkpower.size > 0:
            cutlinkpower_all.append(cutlinkpower)

        unique_bins = np.unique(BINS)

        if len(unique_bins) == 1 or len(unique_bins) == previous_unique_bins:
            break

        previous_unique_bins = len(unique_bins)

    if cutlinkpower_all:
        cutlinkpower_all_np = np.vstack([cp for cp in cutlinkpower_all if cp.size > 0])
        cutlinkpower_all_np = np.float64(cutlinkpower_all_np)
    else:
        cutlinkpower_all_np = np.array([], dtype=np.float64)
        if matlab_compatibility:
            np.seterr(**old_settings)
        return np.zeros(datanum, dtype=np.int64), np.array([], dtype=np.int64), 0, np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64), diagnostics

    mass = np.float64(cutlinkpower_all_np[:, 4] * cutlinkpower_all_np[:, 5])
    R = np.float64(cutlinkpower_all_np[:, 6]**2)
    p = np.float64(mass * R)
    R_mass = np.float64(R / mass)
    """
    if isfig:
        plt.figure(figsize=(10, 12))
        plt.subplot(2, 1, 1)
        plt.plot(R, mass, 'o', markersize=5, markerfacecolor='k', markeredgecolor='k')
        plt.title('Decision Graph', fontsize=15)
        plt.xlabel('R (Distance Squared)')
        plt.ylabel('Mass')
        plt.grid(True)
    """

    order_torque = np.argsort(p, kind='mergesort')[::-1]
    order_2 = np.argsort(order_torque, kind='mergesort')

    if firstlayer_conn_num > 0:
        firstlayer_loc_onsortp = order_2[:firstlayer_conn_num]
    else:
        firstlayer_loc_onsortp = np.array([], dtype=np.int64)

    indices_to_cut_main: List[int] = []

    if K == 0:
        if cluster_selection == "paper_eq4":
            mass_mean = np.mean(mass, dtype=np.float64)
            R_mean = np.mean(R, dtype=np.float64)

            abnormal_mask = (mass >= mass_mean) & (R >= R_mean)
            abnormal_indices = np.where(np.atleast_1d(abnormal_mask))[0]

            indices_to_cut_main = [int(idx) for idx in abnormal_indices]
            cutnum = np.int64(len(indices_to_cut_main))
        elif cluster_selection == "gap":
            NAB, _, _ = Nab_dec(p, mass, R, firstlayer_loc_onsortp, use_std_adjustment, adjustment_factor)
            if len(NAB) == 0:
                raise ValueError("Nab_dec returned empty result while cluster_selection='gap'.")
            cutnum = np.int64(max(1, NAB[0]))
            cutnum_int = int(min(cutnum, len(order_torque)))
            indices_to_cut_main = [int(order_torque[i]) for i in range(cutnum_int)]
        elif cluster_selection=="quantile":
            NAB, _, _ = Nab_dec_quantile(p, mass, R, firstlayer_loc_onsortp, use_std_adjustment, adjustment_factor, mode="torque_band",torque_band=num_band)
            if len(NAB) == 0:
                raise ValueError("Nab_dec returned empty result while cluster_selection='gap'.")
            cutnum = np.int64(max(1, NAB[0]))
            cutnum_int = int(min(cutnum, len(order_torque)))
            indices_to_cut_main = [int(order_torque[i]) for i in range(cutnum_int)]
        else:
            raise ValueError(f"Unknown cluster_selection mode: {cluster_selection}")
    else:
        cutnum = np.int64(max(1, K - 1))
        cutnum_int = int(min(cutnum, len(order_torque)))
        indices_to_cut_main = [int(order_torque[i]) for i in range(cutnum_int)]

    if not indices_to_cut_main:
        cutlink1 = np.empty((0, cutlinkpower_all_np.shape[1]), dtype=np.float64)
    else:
        cutlink1 = cutlinkpower_all_np[indices_to_cut_main, :].copy()

        # Visualize torque values in descending order
    if isfig and p.size > 0:
        # Sort torques in descending order
        # order_torque stores the original indices after sorting by descending torque
        sorted_p = p[order_torque]
        edge_rank = np.arange(1, len(sorted_p) + 1)  # Ranks 1, 2, 3, ...

        plt.figure(figsize=(8, 4))
        # Plot torque values for all edges
        plt.plot(edge_rank, sorted_p, '-o', markersize=4, label='all edges (sorted by torque)')

        if len(indices_to_cut_main) > 0:
            cut_edges = np.array(indices_to_cut_main, dtype=int)

            # Find the rank of each cut edge in the torque-sorted order
            rank_of_cut_edges = []
            for eidx in cut_edges:
                pos = np.where(order_torque == eidx)[0]
                if pos.size > 0:
                    rank_of_cut_edges.append(pos[0] + 1)  # 1-based rank

            rank_of_cut_edges = np.array(rank_of_cut_edges, dtype=int)

            if rank_of_cut_edges.size > 0:
                # Torque values at the selected ranks
                cut_p_values = sorted_p[rank_of_cut_edges - 1]

                # Highlight cut edges with red circles
                plt.scatter(rank_of_cut_edges, cut_p_values,
                            facecolors='none', edgecolors='r',
                            s=60, linewidths=1.5,
                            label='cut edges')

                # Mark the threshold corresponding to the cut region
                # Draw a horizontal line at the minimum torque among the cut edges
                torque_threshold = np.min(cut_p_values)
                plt.axhline(y=torque_threshold,
                            linestyle='--', linewidth=1.2,
                            label=f'min torque of cut edges = {torque_threshold:.3g}')

        # Add a reference line for the current cluster_selection mode when available
        if K == 0 and cluster_selection == "gap" and len(sorted_p) > 0:
            # In gap mode, cut the top cutnum edges and draw a vertical line at that rank
            plt.axvline(x=int(cutnum), linestyle=':', linewidth=1.2,
                        label=f'cutnum = {int(cutnum)}')

        if K == 0 and cluster_selection == "paper_eq4":
            # In paper_eq4 mode, edges exceeding the mean mass and mean R are treated as abnormal merges
            mass_mean = np.mean(mass, dtype=np.float64)
            R_mean    = np.mean(R,    dtype=np.float64)
            abnormal_mask = (mass >= mass_mean) & (R >= R_mean)

            abnormal_on_sorted = abnormal_mask[order_torque]
            abnormal_ranks = edge_rank[abnormal_on_sorted]
            abnormal_p     = sorted_p[abnormal_on_sorted]

            if abnormal_ranks.size > 0:
                plt.scatter(abnormal_ranks, abnormal_p,
                            facecolors='none', edgecolors='g',
                            s=60, linewidths=1.5,
                            label='abnormal (mass>=mean & R>=mean)')

        plt.xlabel('Edge rank (1 = largest torque)')
        plt.ylabel('Torque value p')
        plt.title('Torque vs Edge rank (cut edges highlighted)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    

    cutlink_ori = cutlink1.copy()
    if cutlink1.size > 0:
        cutlink1 = np.delete(cutlink1, [0, 1, 4, 5, 6], axis=1)

    Idx_with_noise = np.array([], dtype=np.int64)
    if isnoise:
        R_mean_noise = np.mean(R, dtype=np.float64)
        mass_mean_noise = np.mean(mass, dtype=np.float64)
        R_mass_mean = np.mean(R_mass, dtype=np.float64)

        epsilon = 1e-10

        noise_loc_indices = np.intersect1d(
            np.intersect1d(
                np.where(R >= R_mean_noise - epsilon)[0],
                np.where(mass <= mass_mean_noise + epsilon)[0]
            ),
            np.where(R_mass >= R_mass_mean - epsilon)[0]
        )

        indices_to_cut_noise_base = indices_to_cut_main
        noise_indices_python = [int(idx) for idx in noise_loc_indices]

        all_indices = set(indices_to_cut_noise_base).union(set(noise_indices_python))
        all_indices = list(all_indices)

        if not all_indices:
            cutlink2 = np.empty((0, cutlinkpower_all_np.shape[1]), dtype=np.float64)
        else:
            cutlink2 = cutlinkpower_all_np[all_indices, :].copy()
        if cutlink2.size > 0:
            cutlink2 = np.delete(cutlink2, [0, 1, 4, 5, 6], axis=1)

    if not isinstance(link_adjacency_matrix, scipy.sparse.csr_matrix):
        link_adjacency_matrix = link_adjacency_matrix.tocsr()

    ljmat1 = link_adjacency_matrix.copy()

    link_adjacency_matrix = link_adjacency_matrix.tolil()

    updates = []
    cutlinknum1 = np.int64(cutlink1.shape[0])
    for i in range(cutlinknum1):
        row_index = np.int64(cutlink1[i, 0])
        col_index = np.int64(cutlink1[i, 1])
        updates.append((row_index, col_index))
        updates.append((col_index, row_index))

    for r, c in updates:
        link_adjacency_matrix[r, c] = 0

    link_adjacency_matrix = link_adjacency_matrix.tocsr()

    ljmat_G = nx.from_scipy_sparse_array(link_adjacency_matrix)
    components = list(nx.connected_components(ljmat_G))
    components.sort(key=lambda c: min(c))

    labels1 = np.zeros(datanum, dtype=np.int64)
    for i, component in enumerate(components):
        for node in component:
            labels1[node] = i

    Idx = labels1.copy()

    if isnoise:
        ljmat1 = ljmat1.tolil()

        updates = []
        cutlinknum2 = np.int64(cutlink2.shape[0])  # type: ignore[name-defined]
        for i in range(cutlinknum2):
            row_index = np.int64(cutlink2[i, 0])  # type: ignore[name-defined]
            col_index = np.int64(cutlink2[i, 1])  # type: ignore[name-defined]
            updates.append((row_index, col_index))
            updates.append((col_index, row_index))

        for r, c in updates:
            ljmat1[r, c] = 0

        ljmat1 = ljmat1.tocsr()

        ljmat1_G = nx.from_scipy_sparse_array(ljmat1)
        components = list(nx.connected_components(ljmat1_G))
        components.sort(key=lambda c: min(c))

        labels2 = np.zeros(datanum, dtype=np.int64)
        for i, component in enumerate(components):
            for node in component:
                labels2[node] = i

        Idx_with_noise = Final_label(labels1, labels2)

    """
    if isfig:
        plt.subplot(2, 1, 2)

        uniqueLabels = np.unique(Idx)
        numClusters = len(uniqueLabels)

        colors = plt.cm.hsv(np.linspace(0, 1, numClusters))

        cluster_to_points = {}
        for i, cluster_id in enumerate(uniqueLabels):
            cluster_to_points[cluster_id] = np.where(Idx == cluster_id)[0]

        for i, cluster_id in enumerate(uniqueLabels):
            cluster_points = cluster_to_points[cluster_id]

            connection_mask = np.zeros(cutlinkpower_all_np.shape[0], dtype=bool)
            for point in cluster_points:
                connection_mask |= (cutlinkpower_all_np[:, 0] == point) | (cutlinkpower_all_np[:, 1] == point)

            connection_indices = np.where(connection_mask)[0]

            if len(connection_indices) > 0:
                plt.plot(R[connection_indices], mass[connection_indices], 'o', markersize=5,
                         markerfacecolor=colors[i], markeredgecolor=colors[i])

        plt.title('Clusters in Decision Graph', fontsize=15)
        plt.xlabel('D (Distance)')
        plt.ylabel('M (Mass)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    """

    if matlab_compatibility:
        np.seterr(**old_settings)

    return Idx, Idx_with_noise, int(cutnum), cutlink_ori, p, firstlayer_loc_onsortp, mass, R, cutlinkpower_all_np, diagnostics
