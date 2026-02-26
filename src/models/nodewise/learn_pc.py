import os
from typing import Optional, Union
import warnings
from collections import deque
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from timeit import default_timer
import logging
import random
from copy import deepcopy

import torch
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score

from src.models.nodewise.utils.clustering import kmeans_sklearn as kmeans_wrapper
from src.models.nodewise.nodes import Node, SumNode, ProductNode, LeafNode, get_nodes_by_type, bfs
from src.models.nodewise.structural_manipulation import prune_pc
from src.models.nodewise.distributions import Gaussian, Categorical, Bernoulli, Poisson
from src.models.nodewise.utils.rdc import partition_by_rdc


PARALLELIZE = True
MAX_NUM_PROCESSES = 40

if PARALLELIZE:
    num_processes = min(MAX_NUM_PROCESSES, max(1, os.cpu_count() - 2))
else:
    num_processes = 1
os.environ["OMP_NUM_THREADS"] = f"{num_processes}"
os.environ["MKL_NUM_THREADS"] = f"{num_processes}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_processes}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{num_processes}"


class Operation(Enum):
    CREATE_LEAF = 1
    SPLIT_COLUMNS = 2
    SPLIT_ROWS = 3
    NAIVE_FACTORIZATION = 4
    REMOVE_UNINFORMATIVE_FEATURES = 5
    CONDITIONING = 6


def get_next_operation(min_instances_slice: int = 100, cluster_univariate: bool = False, remove_uninformative_features: bool = False):
    def _next_operation(data: np.ndarray, scope: list[int], no_clusters: bool = False, no_independencies: bool = False, is_first: bool = False, cluster_first: bool = True):

        minimal_features = len(scope) == 1
        minimal_instances = data.shape[0] <= min_instances_slice

        if minimal_features:
            if minimal_instances or no_clusters:
                return Operation.CREATE_LEAF, None
            else:
                if cluster_univariate:
                    return Operation.SPLIT_ROWS, None
                else:
                    return Operation.CREATE_LEAF, None
                
        uninformative_features_idx = np.var(data[:, 0 : len(scope)], axis=0) == 0
        ncols_zero_variance = np.sum(uninformative_features_idx)
        if ncols_zero_variance > 0:
            if ncols_zero_variance == data.shape[1]:
                return Operation.NAIVE_FACTORIZATION, None
            elif remove_uninformative_features:
                return (
                    Operation.REMOVE_UNINFORMATIVE_FEATURES,
                    np.arange(len(scope))[uninformative_features_idx].tolist(),
                )

        if minimal_instances or (no_clusters and no_independencies):
            return Operation.NAIVE_FACTORIZATION, None
        
        if no_independencies:
            return Operation.SPLIT_ROWS, None
        
        if no_clusters:
            return Operation.SPLIT_COLUMNS, None
        
        if is_first:
            if cluster_first:
                return Operation.SPLIT_ROWS, None
            else:
                return Operation.SPLIT_COLUMNS, None
            
        return Operation.SPLIT_COLUMNS, None
    
    return _next_operation


def default_slicer(data: np.ndarray, cols: list[int], num_cond_cols: Optional[list[int]] = None):
    if num_cond_cols is None:
        if len(cols) == 1:
            return data[:, cols[0]].reshape((-1, 1))
        else:
            return data[:, cols]
    else:
        return np.concatenate((data[:, cols], data[:, -num_cond_cols:]), axis=1)
    

def learn_pc_spflow(data: np.ndarray, distributions_per_scope: list[type[LeafNode]], min_instances: int = 100, k_S: int = 2, k_P: int = 2, rdc_threshold: float = 0.3, epsilon: float = 1e-6, clustering_restarts: int = 10, max_iter: int = 50, number_of_categories: dict[int, int] = None, seed: int = 47, cluster_univariate: bool = False):
    assert data is not None
    assert distributions_per_scope is not None
    assert data.shape[1] == len(distributions_per_scope)
    assert 1 <= min_instances <= data.shape[0]
    assert 0.0 < rdc_threshold <= 1.0

    next_operation = get_next_operation(min_instances_slice=min_instances, cluster_univariate=cluster_univariate)

    initial_scope = list(range(data.shape[1]))
    root = ProductNode(scope=initial_scope, children_circuit=[None])

    tasks = deque()
    next_task: tuple[np.ndarray, Node, int, list[int], bool, bool] = (data, root, 0, initial_scope, False, False)
    tasks.append(next_task)

    while tasks:

        local_data, parent, children_pos, scope, no_clusters, no_independencies = tasks.popleft()

        operation, op_params = next_operation(data=local_data, scope=scope, no_clusters=no_clusters, no_independencies=no_independencies, is_first=(parent is root))

        logging.debug(f"OP: {operation} on slice {local_data.shape} (remaining tasks: {len(tasks)})")

        if operation == Operation.REMOVE_UNINFORMATIVE_FEATURES:
            node = ProductNode(scope=scope, children_circuit=[])
            parent.children_circuit[children_pos] = node

            rest_scope = set(range(len(scope)))
            for col in op_params:
                rest_scope.remove(col)
                node.children_circuit.append(None)
                next_task = (default_slicer(data=local_data, cols=[col], num_cond_cols=None), node, len(node.children_circuit) - 1, [scope[col]], True, True)
                tasks.append(next_task)

            next_final = False

            if len(rest_scope) == 0:
                continue
            elif len(rest_scope) == 1:
                next_final = True

            node.children_circuit.append(None)
            c_pos = len(node.children_circuit) - 1

            rest_cols = list(rest_scope)
            rest_scope = [scope[col] for col in rest_scope]

            next_task = (default_slicer(data=local_data, cols=rest_cols, num_cond_cols=None), node, c_pos, rest_scope, next_final, next_final)

            continue

        elif operation == Operation.SPLIT_ROWS:

            split_start_t = default_timer()
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    clusters = kmeans_wrapper(local_data, n_clusters=k_S, n_init=clustering_restarts, max_iter=max_iter, seed=seed)
                except ConvergenceWarning:
                    clusters = np.zeros((local_data.shape[0],))
            data_slices = split_data_by_clusters(local_data, clusters, scope, rows=True)
            split_end_t = default_timer() 
            logging.debug(f"\t\tfound {len(data_slices)} row clusters (in {split_end_t-split_start_t:.5f}s)")

            if len(data_slices) == 1:
                next_task = (local_data, parent, children_pos, scope, True, False)
                continue

            node = SumNode(scope=scope, children_circuit=[], log_weights=torch.tensor([0.0]))
            parent.children_circuit[children_pos] = node

            weights = []
            for data_slice, scope_slice, proportion in data_slices:
                assert isinstance(scope_slice, list), f"slice must be a list, but was {type(scope_slice)}"

                node.children_circuit.append(None)
                weights.append(proportion)
                next_task = (data_slice, node, len(node.children_circuit) - 1, scope, False, False)
                tasks.append(next_task)

            node.log_weights = torch.nn.Parameter(torch.log(torch.tensor(weights)))

            continue

        elif operation == Operation.SPLIT_COLUMNS:
            split_start_t = default_timer()
            partitions = partition_by_rdc(data=local_data, threshold=rdc_threshold, cpus=num_processes)
            data_slices = split_data_by_clusters(local_data, clusters=partitions, scope=scope, rows=False)
            split_end_t = default_timer()
            logging.debug(f"\t\tfound {len(data_slices)} partitions (in {split_end_t-split_start_t:.5f}s)")

            if len(data_slices) == 1:
                next_task = (local_data, parent, children_pos, scope, False, True)
                tasks.append(next_task)
                continue

            node = ProductNode(scope=scope, children_circuit=[])
            parent.children_circuit[children_pos] = node

            for data_slice, scope_slice, _ in data_slices:
                assert isinstance(scope_slice, list), f"slice must be a list, but was {type(scope_slice)}"

                node.children_circuit.append(None)
                next_task = (data_slice, node, len(node.children_circuit) - 1, scope_slice, False, False)
                tasks.append(next_task)

            continue

        elif operation == Operation.NAIVE_FACTORIZATION:
            node = ProductNode(scope=scope, children_circuit=[])
            parent.children_circuit[children_pos] = node

            local_tasks = []
            local_children_params = []
            split_start_t = default_timer()

            for col in range(len(scope)):
                node.children_circuit.append(None)
                local_tasks.append(len(node.children_circuit) - 1)
                child_data_slice = default_slicer(data=local_data, cols=[col], num_cond_cols=None)
                local_children_params.append((child_data_slice, scope[col], distributions_per_scope[scope[col]], number_of_categories, epsilon))

            # with ProcessPoolExecutor(max_workers=num_processes) as executor:
            #     factor_nodes = list(executor.map(build_univariate_leaf, local_children_params))
            # with Pool(processes=num_processes) as pool:
            #     factor_nodes = pool.starmap(build_univariate_leaf, local_children_params)
            factor_nodes = []
            for lll in local_children_params:
                fn = build_univariate_leaf(*lll)
                factor_nodes.append(fn)

            for child_pos, child in zip(local_tasks, factor_nodes):
                node.children_circuit[child_pos] = child

            split_end_t = default_timer()
            logging.debug(f"\t\tnaive factorization of {len(scope)} columns (in {split_end_t-split_start_t:.5f}s)")

            continue

        elif operation == Operation.CREATE_LEAF:
            leaf_start_t = default_timer()
            assert len(scope) == 1
            scope = scope[0]
            node = build_univariate_leaf(local_data, scope, distributions_per_scope[scope], number_of_categories, epsilon)
            parent.children_circuit[children_pos] = node
            leaf_end_t = default_timer()

            logging.debug(f"\t\tcreated leaf {node.__class__.__name__} with scope={scope} (in {leaf_end_t-leaf_start_t:.5f}s)")

        else:
            raise ValueError("Invalid operation: " + operation)
        

    node: ProductNode = root.children_circuit[0]
    try:
        node.is_valid()
    except Exception as e:
        print(node)
        raise e
    node = prune_pc(node)
    try:
        node.is_valid()
    except Exception as e:
        print(node)
        raise e

    return node
   

def split_data_by_clusters(data: np.ndarray, clusters: list[int], scope: list[int], rows=True):
    unique_clusters = np.unique(clusters)
    result = []

    nscope = np.asarray(scope)

    for uc in unique_clusters:
        if rows:
            local_data = data[clusters == uc, :]
            proportion = local_data.shape[0] / data.shape[0]
            result.append((local_data, scope, proportion))
        else:
            local_data = data[:, clusters == uc].reshape((data.shape[0], -1))
            proportion = local_data.shape[1] / data.shape[1]
            result.append((local_data, nscope[clusters == uc].tolist(), proportion))

    return result


def learn_pc_recursive(data: np.ndarray, distributions_per_scope: list[type[LeafNode]], min_instances: int = 100, k_S: Union[int, tuple[int, int]] = 2, k_P: int = 2, rdc_threshold: float = 0.3, epsilon: float = 1e-6, clustering_restarts: int = 10, max_iter: int = 50, number_of_categories: dict[int, int] = None, seed: int = 47) -> Node:
    
    def build_leaf(data: np.ndarray, scope: int):
        with warnings.catch_warnings():
            leaf = build_univariate_leaf(data=data, scope=scope, distribution=distributions_per_scope[scope], number_of_categories=number_of_categories, epsilon=epsilon)
            return leaf
        
    def build_sum_node(data: np.ndarray, scopes: list[int]):
        if len(scopes) == 1:
            leaf_scope = scopes[0]
            return build_leaf(data=data, scope=leaf_scope)
        
        # elif len(data) <= min_instances:
        #     time_factor_start = default_timer()
        #     children_circuit = [build_leaf(data=data, scope=scope) for scope in scopes]
        #     prod = ProductNode(scope=scopes, children_circuit=children_circuit)
        #     time_factor_end = default_timer()
        #     logging.debug(f"\t\t took {time_factor_end-time_factor_start:.5f}s to factorize {scopes}")
        #     return prod
        
        else:
            # cluster
            time_cluster_start = default_timer()
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                if type(k_S) == int:
                    k_S_range = (k_S, k_S+1)
                else:
                    k_S_range = k_S
                best_n = -1
                best_score = -np.inf
                best_clusters = None
                for n in range(*k_S_range):
                    try:
                        clusters = kmeans_wrapper(data[:, scopes], n_clusters=n, n_init=clustering_restarts, max_iter=max_iter, seed=seed)
                        score = silhouette_score(data[:, scopes], clusters)
                    except ConvergenceWarning:
                        clusters = np.zeros((data.shape[0], ))
                        score = -1
                    if score > best_score:
                        best_n = n
                        best_score = score
                        best_clusters = clusters
                clusters = best_clusters
            time_cluster_end = default_timer()
            logging.debug(f"\tCLUSTERING:\t\t took {time_cluster_end-time_cluster_start:.5f}s to find {best_n} clusters for {scopes} of data-slice {data.shape}")
            
            # create children
            linear_weights = np.array([len(clusters[clusters == k]) / len(data) for k in np.unique(clusters)])
            log_weights = torch.log(torch.tensor(linear_weights))
            children_circuit = [build_product_node(data=data[clusters==k], scopes=scopes) for k in np.unique(clusters)]
            sum = SumNode(scope=scopes, children_circuit=children_circuit, log_weights=log_weights)
            return sum
        
    def build_product_node(data: np.ndarray, scopes: list[int]):
        if len(data) <= min_instances:
            time_factor_start = default_timer()
            children_circuit = [build_leaf(data=data, scope=scope) for scope in scopes]
            prod = ProductNode(scope=scopes, children_circuit=children_circuit)
            time_factor_end = default_timer()
            logging.debug(f"\tFACTORIZATION:\t took {time_factor_end-time_factor_start:.5f}s for {scopes} of data_slice {data.shape}")
            return prod
        
        # partiton
        time_partition_start = default_timer()
        partitions = partition_by_rdc(data=data[:, scopes], threshold=rdc_threshold).astype(int)
        time_partition_end = default_timer()
        logging.debug(f"\tPARTITION:\t\t took {time_partition_end-time_partition_start:.5f}s for {scopes} of data-slice {data.shape}")

        # create children
        children_circuit = [build_sum_node(data=data, scopes=np.array(scopes)[partitions==k].tolist()) for k in np.unique(partitions)]
        prod = ProductNode(scope=scopes, children_circuit=children_circuit)
        return prod
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    scopes = list(range(data.shape[1]))
    time_start = default_timer()
    if data.shape[0] >= min_instances:
        root = build_product_node(data=data, scopes=scopes)
    else:
        root = build_sum_node(data=data, scopes=scopes)
    time_end = default_timer()
    logging.debug(f"\tTOTAL:\t\t\t took {time_end-time_start:.5f}s to construct circuit from data-slice {data.shape}")

    all_nodes = root.get_nodes_by_type()
    sum_nodes = root.get_nodes_by_type(node_type=SumNode)
    product_nodes = root.get_nodes_by_type(node_type=ProductNode)
    leaf_nodes = root.get_nodes_by_type(node_type=LeafNode)
    logging.debug(f"Number of nodes:")
    logging.debug(f"- Total:    {len(all_nodes)}")
    logging.debug(f"- Sums:     {len(sum_nodes)}")
    logging.debug(f"- Product:  {len(product_nodes)}")
    logging.debug(f"- Leaves:   {len(leaf_nodes)}")
    try: 
        root.is_valid()
        logging.debug(f"Is valid: True")
    except:
        logging.debug(f"Is valid: FALSE")
        print("CIRCUIT NOT VALID")
    logging.debug(str(root))

    time_prune_start = default_timer()
    root = prune_pc(root)
    time_prune_end = default_timer()
    logging.debug(f"\t\tPRUNING: took {time_prune_end-time_prune_start:.5f}s")

    all_nodes = root.get_nodes_by_type()
    sum_nodes = root.get_nodes_by_type(node_type=SumNode)
    product_nodes = root.get_nodes_by_type(node_type=ProductNode)
    leaf_nodes = root.get_nodes_by_type(node_type=LeafNode)
    logging.debug(f"Number of nodes:")
    logging.debug(f"- Total:    {len(all_nodes)}")
    logging.debug(f"- Sums:     {len(sum_nodes)}")
    logging.debug(f"- Product:  {len(product_nodes)}")
    logging.debug(f"- Leaves:   {len(leaf_nodes)}")
    try: 
        root.is_valid()
        logging.debug(f"Is valid: True")
    except:
        logging.debug(f"Is valid: FALSE")
        print("CIRCUIT NOT VALID")
    logging.debug(str(root))

    warnings.resetwarnings()

    return root


def build_univariate_leaf(data: np.ndarray, scope: int, distribution: LeafNode, number_of_categories: dict[int, int] = None, epsilon: float = 1e-6) -> LeafNode:
    assert type(scope) is int
    if data.ndim == 1 or data.shape[1] == 1:
        leaf_data = torch.tensor(data, dtype=torch.get_default_dtype())
    else:
        leaf_data = torch.tensor(data[:, scope], dtype=torch.get_default_dtype())

    if distribution == Gaussian:
        mean = torch.mean(leaf_data)
        if mean is None or not torch.isfinite(mean):
            mean = torch.tensor(0.0)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                stdev = torch.std(leaf_data)
            except UserWarning as e:
                stdev = None
        if stdev is None or stdev < epsilon or not torch.isfinite(stdev):
            stdev = torch.tensor(epsilon)
        log_stdev = torch.log(stdev)
        return Gaussian(scope=[scope], mean=mean, log_stdev=log_stdev)
    elif distribution == Categorical:
        if number_of_categories is None:
            raise ValueError("Trying to learn categorical leaf but cat_numbers is None")
        classes = list(range(number_of_categories[scope]))
        p = torch.tensor([len(leaf_data[leaf_data == k]) / len(leaf_data) for k in classes])
        for i, _ in enumerate(p):
            if p[i] is None or p[i] < epsilon:
                p[i] = epsilon
        p /= torch.sum(p)
        log_p = torch.log(p)
        return Categorical(scope=[scope], log_p=log_p)     
    elif distribution == Bernoulli:
        mean = torch.mean(leaf_data)
        if mean is None or not torch.isfinite(mean):
            mean = torch.tensor(0.0)
        log_p = torch.log(mean)
        return Bernoulli(scope=[scope], log_p=log_p)
    elif distribution == Poisson:
        mean = torch.mean(leaf_data)
        if mean is None or not torch.isfinite(mean):
            mean = torch.tensor(epsilon)
        log_lambda = torch.log(mean)
        return Poisson(scope=[scope], log_lambda=log_lambda)        
    else:
        raise ValueError(f"unknown leaftype {distribution}")
