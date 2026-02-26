from queue import Queue
from collections import deque
from typing import Literal, Type
import torch
from scipy.stats import ecdf

from src.models.nodewise.pvalue_combination_functions import EmptyStrategy, PValueCombinationStrategy, TestStrategy


class Node(torch.nn.Module):
    def __init__(self, scope: list[int]) -> None:
        super().__init__()
        scope.sort()
        self.scope = scope
        self._results_cache = None

    def __description__(self) -> str:
        return f"Node @ {self.scope}"
    
    def __str__(self, level=0) -> str:
        return "\t"*level+repr(self.__description__())+"\n"
    
    def __repr__(self) -> str:
        return self.__description__()
    
    def is_valid(self) -> None:
        raise NotImplementedError
    
    def get_nodes_by_type(self, node_type: Type = None, nodes: Queue["Node"] = None):
        if nodes is None:
            nodes = Queue()
        if node_type is None or isinstance(self, node_type):
            nodes.put(self)
        if isinstance(self, (SumNode, ProductNode)):
            for child in self.children_circuit:
                child.get_nodes_by_type(node_type, nodes)
        return list(nodes.queue)

    def print_stats(self) -> None:
        print(self.get_stats())

    def get_stats(self):
        all_nodes = self.get_nodes_by_type()
        sum_nodes = self.get_nodes_by_type(node_type=SumNode)
        product_nodes = self.get_nodes_by_type(node_type=ProductNode)
        leaf_nodes = self.get_nodes_by_type(node_type=LeafNode)

        out_str = ""
        out_str += f"Number of nodes:\n"
        out_str += f"- Total:    {len(all_nodes)}\n"
        out_str += f"- Sums:     {len(sum_nodes)}\n"
        out_str += f"- Product:  {len(product_nodes)}\n"
        out_str += f"- Leaves:   {len(leaf_nodes)}\n"
        return out_str

    def mean(self) -> torch.Tensor:
        raise NotImplementedError
    
    def var(self) -> torch.Tensor:
        raise NotImplementedError
    
    def std(self) -> torch.Tensor:
        mixture_variance = self.var()
        mixture_stdev = torch.sqrt(mixture_variance)
        return mixture_stdev
    
    def log_pdf(self, data: torch.Tensor, write_cache: bool = False) -> torch.Tensor:
        raise NotImplementedError
    
    def pdf(self, data: torch.Tensor, write_cache: bool = False) -> torch.Tensor:
        raise NotImplementedError
    
    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError
    
    def test_membership_distribution(self, data: torch.Tensor, test_strategy: TestStrategy) -> torch.Tensor:
        """
        Tests if the data comes from the distribution encoded by the circuit.
        Evaluates univariate p-values in the univariate leafs, and combines them into a multivariate p-value.

        Arguments:
            - data:     Tensor containing the data to test on.
            - alpha:    Significance niveau, in (0, 1]. Should fix the false positive rate of the test.
            - side:     Side of the univariate distributions (leaves) to test on.

        Returns:
            - Combined p-values describing the probability that the observed data or more extreme events have occured.
        """
        raise NotImplementedError


class SumNode(Node):
    def __init__(self, scope: list[int], children_circuit: list[Node], log_weights: torch.Tensor = None) -> None:
        super().__init__(scope)
        
        if log_weights is None:
            weights = torch.rand(size=(len(children_circuit),))
            weights = weights / torch.sum(weights)
            log_weights = torch.log(weights)

        self.children_circuit = children_circuit
        self.log_weights = torch.nn.Parameter(log_weights, requires_grad=True)

    def __description__(self) -> str:
        with torch.no_grad():
            linear_weights = self._get_linear_weights()
            output = ["%.4f" % elem for elem in linear_weights]
            return f"Sum({output}) @ {self.scope}"
    
    def __str__(self, level=0) -> str:
        ret = "\t"*level+repr(self.__description__())+"\n"
        for child in self.children_circuit:
            ret += child.__str__(level+1)
        return ret
    
    def is_valid(self) -> None:
        assert torch.isclose(torch.sum(torch.exp(self.log_weights)), torch.tensor(1.0)), torch.exp(self.log_weights)
        for c in self.children_circuit:
            assert self.scope == c.scope, f"{self.scope}, {c.scope}"
            c.is_valid()
    
    def _get_linear_weights(self) -> torch.Tensor:
        log_weights = self.log_weights
        linear_weights = torch.nn.functional.softmax(log_weights, dim=0)
        return linear_weights

    def mean(self) -> torch.Tensor:
        means_children = torch.concat([child.mean() for child in self.children_circuit], dim=0)
        linear_weights = self._get_linear_weights().view(1, -1)
        mixture_mean = linear_weights @ means_children
        return mixture_mean

    def var(self):
        means_children = torch.concat([child.mean() for child in self.children_circuit], axis=0)
        var_children = torch.concat([child.var() for child in self.children_circuit], axis=0)
        linear_weights = self._get_linear_weights().view(1, -1)

        mixture_variance_term1 = linear_weights @ (torch.pow(means_children, 2) + var_children)
        mixture_variance_term2 = torch.pow(linear_weights @ means_children, 2)
        mixture_variance = mixture_variance_term1 - mixture_variance_term2
        
        return mixture_variance
    
    def log_pdf(self, data: torch.Tensor, write_cache: bool = False) -> torch.Tensor:
        children_log_densities = []
        for child in self.children_circuit:
            child_log_densities = child.log_pdf(data, write_cache=write_cache)
            children_log_densities.append(child_log_densities)
        concat_children = torch.cat(children_log_densities, dim=1)
        node_log_densities = torch.logsumexp(concat_children + self.log_weights, dim=1, keepdim=True)

        if write_cache:
            self._results_cache = node_log_densities

        return node_log_densities
    
    def pdf(self, data: torch.Tensor, write_cache: bool = False) -> torch.Tensor:
        linear_weights = self._get_linear_weights().unsqueeze(-1)
        children_densities = []
        for child in self.children_circuit:
            child_densities = child.pdf(data, write_cache=write_cache)
            children_densities.append(child_densities)
            
        concat_children = torch.cat(children_densities, dim=1)
        node_densities = concat_children @ linear_weights

        if write_cache:
            self._results_cache = node_densities

        return node_densities 
    
    def sample(self, n: int) -> torch.Tensor:
        # SumNode: sample child selector based on weights
        # Convert log_weights to probabilities
        weights = torch.exp(self.log_weights)  # Shape: (num_children,)
        weights = weights / torch.sum(weights)  # Normalize
        
        # Sample which child to use for each sample
        # selector shape: (n,) with values in [0, num_children)
        selector = torch.multinomial(weights, num_samples=n, replacement=True)
        
        # Initialize output tensor
        samples = torch.zeros(n, len(self.scope))
        
        # For each child, get samples from those indices that selected it
        for child_idx, child in enumerate(self.children_circuit):
            # Find which samples selected this child
            mask = (selector == child_idx)
            num_selected = mask.sum().item()
            
            if num_selected > 0:
                # Sample from this child
                child_samples = child.sample(num_selected)
                
                # Place samples in correct positions
                samples[mask] = child_samples
        
        return samples
    
    def test_membership_distribution(self, data: torch.Tensor, test_strategy: TestStrategy) -> torch.Tensor:
        children_pvalues = []
        for child in self.children_circuit:
            child_pvalues = child.test_membership_distribution(data, test_strategy)
            children_pvalues.append(child_pvalues)
        concat_children = torch.cat(children_pvalues, dim=1)

        # nan_mask = ~torch.isnan(concat_children).any(dim=0)
        # filtered_concat_children = concat_children[:, nan_mask]

        node_pvalues = test_strategy.sum_strategy.execute(concat_children, self)
        node_pvalues = node_pvalues.clip(min=0.0, max=1.0)
        return node_pvalues


class ProductNode(Node):
    def __init__(self, scope: list[int], children_circuit: list[Node]) -> None:
        super().__init__(scope)
        self.children_circuit = children_circuit

    def __description__(self) -> str:
        return f"Prod @ {self.scope}"
    
    def __str__(self, level=0) -> str:
        ret = "\t"*level+repr(self.__description__())+"\n"
        for child in self.children_circuit:
            ret += child.__str__(level+1)
        return ret
 
    def is_valid(self) -> None:
        children_scopes = []
        for c in self.children_circuit:
            for s in c.scope:
                assert s not in children_scopes, f"s:{s}, scopes: {children_scopes}"
                children_scopes.append(s)
            c.is_valid()
        assert set(children_scopes) == set(self.scope), f"{children_scopes}, {self.scope}"

    def mean(self):
        means_children = torch.concat([child.mean() for child in self.children_circuit], axis=1)
        return means_children
    
    def var(self):
        variance_children = torch.concat([child.var() for child in self.children_circuit], axis=1)
        return variance_children

    def log_pdf(self, data: torch.Tensor, write_cache: bool = False) -> torch.Tensor:
        children_log_densities = []
        for child in self.children_circuit:
            child_log_densities = child.log_pdf(data, write_cache=write_cache)
            children_log_densities.append(child_log_densities)
        concat_children = torch.cat(children_log_densities, dim=1)

        nan_mask = ~torch.isnan(concat_children).any(dim=0)
        filtered_concat_children = concat_children[:, nan_mask]

        node_log_densities = torch.sum(filtered_concat_children, dim=1, keepdim=True)

        if write_cache:
            self._results_cache = node_log_densities

        return node_log_densities
    
    def pdf(self, data: torch.Tensor, write_cache: bool = False) -> torch.Tensor:
        children_densities = []
        for child in self.children_circuit:
            child_densities = child.pdf(data, write_cache=write_cache)
            children_densities.append(child_densities)
        concat_children = torch.cat(children_densities, dim=1)
        node_densities = torch.prod(concat_children, dim=1, keepdim=True)

        if write_cache:
            self._results_cache = node_densities

        return node_densities  
    
    def sample(self, n: int) -> torch.Tensor:
        # ProductNode: sample from each child and concatenate
        # Each child has a disjoint scope
        
        # Initialize output tensor
        samples = torch.zeros(n, len(self.scope))
        
        # Sample from each child
        for child in self.children_circuit:
            child_samples = child.sample(n)  # Shape: (n, len(child.scope))
            
            # Place samples in correct scope positions
            for i, var in enumerate(child.scope):
                # Find position of var in node.scope
                pos = self.scope.index(var)
                samples[:, pos] = child_samples[:, i]
        
        return samples

    def test_membership_distribution(self, data: torch.Tensor, test_strategy: TestStrategy) -> torch.Tensor:
        children_pvalues = []
        for child in self.children_circuit:
            child_pvalues = child.test_membership_distribution(data, test_strategy)
            children_pvalues.append(child_pvalues)
        concat_children = torch.cat(children_pvalues, dim=1)

        nan_mask = ~torch.isnan(concat_children).any(dim=0)
        filtered_concat_children = concat_children[:, nan_mask]

        node_pvalues = test_strategy.prod_strategy.execute(filtered_concat_children, self)
        node_pvalues = node_pvalues.clip(min=0.0, max=1.0)
        return node_pvalues


class LeafNode(Node):
    def __init__(self, scope: list[int]) -> None:
        super().__init__(scope)
        
    def _get_linear_parameters(self) -> any:
        raise NotImplementedError

    def _get_distribution(self) -> torch.distributions.Distribution:
        raise NotImplementedError 
    
    def mean(self) -> torch.Tensor:
        distribution = self._get_distribution()
        mean = distribution.mean
        return mean.view(-1, 1)

    def var(self) -> torch.Tensor:
        distribution = self._get_distribution()
        variance = distribution.variance
        return variance.view(-1, 1)
    
    def log_pdf(self, data: torch.Tensor, write_cache: bool = False, self_extract_data: bool = True) -> torch.Tensor:
        if self_extract_data:
            leaf_data = data[:, self.scope]
        else:
            leaf_data = data

        nan_mask = torch.isnan(leaf_data)
        leaf_data = torch.where(torch.isnan(leaf_data), 1.0, leaf_data)

        distribution = self._get_distribution()
        log_densities = distribution.log_prob(leaf_data)

        log_densities[nan_mask] = torch.nan

        if write_cache:
            self._results_cache = log_densities

        return log_densities
    
    def pdf(self, data: torch.Tensor, write_cache: bool = False, self_extract_data: bool = True) -> torch.Tensor:
        log_densities = self.log_pdf(data, write_cache=write_cache, self_extract_data=self_extract_data)
        densities = torch.exp(log_densities)

        if write_cache:
            self._results_cache = densities

        return densities
    
    def cdf(self, data: torch.Tensor, self_extract_data: bool = True) -> torch.Tensor:
        if self_extract_data:
            leaf_data = data[:, self.scope]
        else:
            leaf_data = data

        nan_mask = torch.isnan(leaf_data)
        leaf_data = torch.where(torch.isnan(leaf_data), 1.0, leaf_data)

        distribution = self._get_distribution()
        probabilities = distribution.cdf(leaf_data)

        probabilities[nan_mask] = torch.nan

        return probabilities
    
    def sample(self, n: int) -> torch.Tensor:
        distribution = self._get_distribution()
        samples = distribution.sample((n, 1)).to(dtype=torch.get_default_dtype())
        return samples
    
    def test_membership_distribution(self, data: torch.Tensor, test_strategy: TestStrategy) -> torch.Tensor:
        prob = self.cdf(data)
        side = test_strategy.side
        if side == "left":
            pvalues = prob
        elif side == "right":
            pvalues = 1.0 - prob
        elif side == "both":
            pvalues = 2 * torch.minimum(prob, 1.0 - prob)
        else:
            raise ValueError(f"Could not resolve 'side'={side}")
        
        return pvalues


def get_nodes_by_type(node, ntype=Node):
    assert node is not None

    result = []

    def add_node(node):
        if isinstance(node, ntype):
            result.append(node)

    bfs(node, add_node)

    return result


def bfs(root, func):
    seen, queue = set([root]), deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        if not isinstance(node, LeafNode):
            for c in node.children_circuit:
                if c not in seen:
                    seen.add(c)
                    queue.append(c)

