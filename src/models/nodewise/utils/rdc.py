from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from joblib import Parallel, delayed
import numpy as np
from sklearn.cross_decomposition import CCA
from itertools import combinations
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning


MAX_ITER_CCA = 500
PARALLELIZE = False
PARALLELIZATION_THRESHOLD = None

def connected_components(adjacency_matrix: np.ndarray) -> list[set[int]]:
    """Computes all connected components in an undirected graph.

    Computes all connected components in an undirected graph.
    Returns a list of all connected componentsin an undirected graph,specified using an adjaceny matrix.

    Args:
        adjacency_matrix:
            Two-dimensional NumPy array representing the symmetric adjacency matrix of an undirected graph.
            Any non-zero entry will be treated as an edge.

    Returns:
        A list containing sets of integers representing the node ids of the connected components.

    Raises:
        ValueError: Specified adjacency matrix is not symmetric.
    """

    if not np.all(adjacency_matrix == adjacency_matrix.T):
        raise ValueError(
            "Connected components expects input to be an undirected graph, but specified adjacency matrix was not symmetrical."
        )

    # convert adjacency matrix to boolean (any non-zero entries are treated as an edge)
    adjacency_matrix = adjacency_matrix != 0

    # set of all vertices
    vertices = set(range(adjacency_matrix.shape[0]))

    # list of connected components
    ccs = []

    while vertices:

        # perform breadth-first search
        visited = set()
        active_set = {vertices.pop()}

        # while there are previously unvisited vertices
        while active_set:

            source = active_set.pop()
            visited.update([source])

            # get neighbors of node
            neighbors = np.where(adjacency_matrix[source])[0]

            # add all unvisited vertices to active set
            active_set.update(set(neighbors.tolist()).difference(visited))

        # add visited vertices to list of connected components
        ccs.append(visited)

        # remove connected component from set of vertices to explore
        vertices = vertices.difference(visited)

    return ccs


def empirical_cdf(data: np.ndarray) -> np.ndarray:
    """Computes the empirical cummulative distribution function (CDF) for specified input data.

    Returns the values of all input data according to the empirical cummulative distribution function (CDF) computed from said data.

    Args:
        data:
            Two-dimensional NumPy array containing empirical input values. Each row is regarded as a sample and each column as a different feature.
            All columns (i.e., features) are ranked independently. Missing entries (i.e., NaN) are ignored and are not counted towards the empirical
            CDF of the corresponding feature.

    Returns:
        Numpy array containing the empirical CDF values for the specified input.
    """
    # empirical cumulative distribution function (step function that increases by 1/N at each unique data step in order)
    # here: done using scipy's 'rankdata' function (preferred over numpy's argsort due to tie-breaking)

    nan_mask = np.isnan(data)

    # rank data values from min to max
    ecd = rankdata(data, axis=0, method="max").astype(float)

    # set nan values to 0
    ecd[nan_mask] = 0

    # normalize rank values (not counting nan entries) to get ecd values
    n_entries = (~nan_mask).sum(axis=0, keepdims=True)
    n_entries[n_entries == 0] = 1

    # normalize rank values (not counting nan entries) to get ecd values
    ecd /= n_entries

    return ecd

def _rdc_cca(args):
    i, j, rdc_features = args
    cca = CCA(n_components=1, max_iter=MAX_ITER_CCA)
    try:
        X_cca, Y_cca = cca.fit_transform(rdc_features[i], rdc_features[j])
        rdc = np.corrcoef(X_cca.T, Y_cca.T)[0, 1]
    except (RuntimeWarning, UserWarning):
        rdc = 0.0
    return rdc


def randomized_dependency_coefficients(
    data: np.ndarray, k: int = 100, s: float = 1 / 6, phi: callable = np.sin, cpus: int = -2
) -> np.ndarray:
    """Computes the randomized dependency coefficients (RDCs) for a given data set.

    Returns the randomized dependency coefficients (RDCs) computed from a specified data set, as described in (Lopez-Paz et al., 2013): "The Randomized Dependence Coefficient"

    Args:
        data:
            Two-dimensional NumPy array containing the data set. Each row is regarded as a sample and each column as a different feature.
            May not contain any missing (i.e., NaN) entries.
        k:
            Integer specifying the number of random projections to be used.
            Defaults to 100.
        s:
            Floating point value specifying the standard deviation of the Normal distribution to sample the weights for the random projections from.
            Defaults to 1/6.
        phi:
            Callable representing the (non-linear) projection.
            Defaults to 'np.sin'.

    Returns:
        NumPy array containing the computed randomized dependency coefficients.

    Raises:
        ValueError: Invalid inputs.
    """
    # default arguments according to paper
    if np.any(np.isnan(data)):
        raise ValueError("Randomized dependency coefficients cannot be computed for data with missing values.")

    # compute ecd values for data
    ecdf = empirical_cdf(data)

    # bring ecdf values into correct shape and pad with ones (for biases)
    ecdf_features = np.stack([ecdf.T, np.ones(ecdf.T.shape)], axis=-1)

    # compute random weights (and biases) generated from normal distribution
    rand_gaussians = np.random.randn(data.shape[1], 2, k)  # 2 for weight (of size 1) and bias

    # compute linear combinations of ecdf feature using generated weights
    features = np.stack([np.dot(features, weights) for features, weights in zip(ecdf_features, rand_gaussians)])
    features *= s  # multiplying by s is equal to generating random weights from N(0,s)

    # apply non-linearity phi
    features: np.ndarray = phi(features)


    pairwise_combinations = list(combinations(range(data.shape[1]), 2))
    
    import warnings
    warnings.filterwarnings("error")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    rdcs = np.eye(data.shape[1])


    if PARALLELIZE and PARALLELIZATION_THRESHOLD is not None and data.shape[0] >= PARALLELIZATION_THRESHOLD:
        # compute rdcs for all pairs of features
        with ThreadPoolExecutor(max_workers=8) as executor:
        # with ProcessPoolExecutor(max_workers=cpus) as executor:
            for (i, j) in pairwise_combinations:
                future = executor.submit(_rdc_cca, (i, j, features))
                rdcs[i, j] = rdcs[j, i] = future.result()            
    else:
        cca = CCA(n_components=1, max_iter=MAX_ITER_CCA)
        for (i, j) in pairwise_combinations:
            try:
                i_cca, j_cca = cca.fit_transform(features[i], features[j])
                rdcs[j][i] = rdcs[i][j] = np.corrcoef(i_cca.T, j_cca.T)[0, 1]
            except (RuntimeWarning, UserWarning):
                # the cca might estimate constant functions, which in turn have a stdev of 0, hence we cannot compute the Pearson correlation coefficent
                # if this happens, we catch numpy's ("RuntimeWarning: invalid value encountered in divide c /= stdev[:, None] || c /= stdev[None, :]") 
                # and set the pairwise rdc to 0, as we have no correlation
                rdcs[j][i] = rdcs[i][j] = 0       

    warnings.resetwarnings()

    return rdcs


def partition_by_rdc(
    data: np.ndarray,
    threshold: float = 0.3,
    cpus = None,
) -> np.ndarray:
    """Performs partitioning usig randomized dependence coefficients (RDCs) to be used with the LearnSPN algorithm in the ``base`` backend.

    Args:
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        threshold:
            Floating point value specifying the threshold for independence testing between two features.
            Defaults to 0.3.
        preprocessing:
            Optional callable that is called with ``data`` and returns another NumPy array of the same shape.
            Defaults to None.

    Returns:
        One-dimensional NumPy array with the same number of entries as the number of features in ``data``.
        Each integer value indicates the partition the corresponding feature is assigned to.
    """
    # get pairwise rdc values
    rdcs = randomized_dependency_coefficients(data, cpus=cpus)

    # create adjacency matrix of features from thresholded rdcs
    adj_mat = (rdcs >= threshold).astype(int)

    partition_ids = np.zeros(data.shape[1])

    for i, cc in enumerate(connected_components(adj_mat)):
        partition_ids[list(cc)] = i

    return partition_ids
