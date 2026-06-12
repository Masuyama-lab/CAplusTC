# This file adapts code from Cognet-74/TorqueClusteringPy:
# https://github.com/Cognet-74/TorqueClusteringPy
# Upstream license: CC BY-NC-SA 4.0. See THIRD_PARTY_NOTICES.md.

import numpy as np
import logging
import scipy.sparse as sp
from utils.mindisttwinsloc import mindisttwinsloc

# Set up logger
logger = logging.getLogger(__name__)


def Updateljmat_weighted(
    old_ljmat,
    neiborloc,
    community,
    commu_DM,
    G,
    ALL_DM,
    node_masses=None,
):
    """
    Update the connectivity matrix (ljmat) and record cut link power information
    with optional community mass based on per-node masses.

    Args:
        old_ljmat: Current link adjacency matrix (sparse or dense).
        neiborloc: List of neighbor indices for each community.
        community: List of communities, each community is a list of point/node indices.
        commu_DM: Inter-community distance matrix.
        G: Graph connectivity matrix for current layer.
        ALL_DM: Original distance matrix (sparse).
        node_masses: Optional 1D array of per-node masses (e.g., CountNode of IDAT nodes).
                     If provided, community mass is sum(node_masses[idx] for idx in community[i]).
                     If None, community mass falls back to len(community[i]) (original behavior).

    Returns:
        new_ljmat: Updated link adjacency matrix.
        cutlinkpower: (n_links, 7) array with:
            [0] - min index in community i
            [1] - min index in community j
            [2] - linkloc1 (point index in community i)
            [3] - linkloc2 (point index in community j)
            [4] - mass of community i
            [5] - mass of community j
            [6] - inter-community distance (commu_DM[i, j])
    """
    old_ljmat_is_sparse = sp.issparse(old_ljmat)
    G_is_sparse = sp.issparse(G)

    if not isinstance(community, list):
        logger.warning("community should be a list - unexpected behavior may occur")
    if not isinstance(neiborloc, list):
        logger.warning("neiborloc should be a list - unexpected behavior may occur")

    community_num = len(community)
    logger.debug(f"Processing {community_num} communities")

    # MATLAB compatible empty check
    def is_matlab_empty(n):
        """Match MATLAB's emptiness check."""
        return n is None or (isinstance(n, list) and len(n) == 0)

    # Helper: compute mass of a community
    def community_mass(idx):
        members = community[idx]
        if node_masses is None:
            return float(len(members))
        return float(sum(node_masses[m] for m in members))

    pd = len(community[0])

    if pd > 1:
        cutlinknum = sum(1 for n in neiborloc if not is_matlab_empty(n))
        cutlinkpower = np.zeros((cutlinknum, 7), dtype=np.float64)
        th = 0

        for i in range(community_num):
            if not is_matlab_empty(neiborloc[i]):
                neighbor_idx = neiborloc[i]

                linkloc1, linkloc2 = mindisttwinsloc(
                    community[i], community[neighbor_idx], ALL_DM
                )

                xx = min(community[i])
                yy = min(community[neighbor_idx])

                old_ljmat[linkloc1, linkloc2] = 1
                old_ljmat[linkloc2, linkloc1] = 1

                mass_i = community_mass(i)
                mass_j = community_mass(neighbor_idx)

                cutlinkpower[th, 0] = xx
                cutlinkpower[th, 1] = yy
                cutlinkpower[th, 2] = linkloc1
                cutlinkpower[th, 3] = linkloc2
                cutlinkpower[th, 4] = mass_i
                cutlinkpower[th, 5] = mass_j
                cutlinkpower[th, 6] = commu_DM[i, neighbor_idx]

                th += 1

    elif pd == 1:
        cutlinkpower = np.zeros((community_num, 7), dtype=np.float64)
        th = 0

        for i in range(community_num):
            linkloc1 = community[i][0]
            neighbor_idx = neiborloc[i]
            linkloc2 = community[neighbor_idx][0]

            mass_i = community_mass(i)
            mass_j = community_mass(neighbor_idx)

            cutlinkpower[th, 0] = linkloc1
            cutlinkpower[th, 1] = linkloc2
            cutlinkpower[th, 2] = linkloc1
            cutlinkpower[th, 3] = linkloc2
            cutlinkpower[th, 4] = mass_i
            cutlinkpower[th, 5] = mass_j
            cutlinkpower[th, 6] = commu_DM[i, neighbor_idx]

            th += 1

        old_ljmat = G

    new_ljmat = old_ljmat

    if old_ljmat_is_sparse and not sp.issparse(new_ljmat):
        new_ljmat = sp.csr_matrix(new_ljmat)
    elif not old_ljmat_is_sparse and sp.issparse(new_ljmat):
        new_ljmat = new_ljmat.toarray()

    return new_ljmat, cutlinkpower
