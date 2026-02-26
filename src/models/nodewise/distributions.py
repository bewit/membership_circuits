import torch
from src.models.nodewise.nodes import LeafNode
from src.models.nodewise.pvalue_combination_functions import TestStrategy


class Gaussian(LeafNode):
    def __init__(self, scope: list[int], mean: torch.Tensor = None, log_stdev: torch.Tensor = None) -> None:
        super().__init__(scope)
        if mean is None:
            mean = torch.randn(1)[0]
        if log_stdev is None:
            log_stdev = torch.randn(1)[0]
        self._mean = torch.nn.Parameter(mean, requires_grad=True)
        self._log_stdev = torch.nn.Parameter(log_stdev, requires_grad=True)

    def __description__(self) -> str:
        with torch.no_grad():
            mean, stdev = self._get_linear_parameters()
            return f"N({mean.item():.4f}, {stdev.item():.4f}) @ {self.scope}"
        
    def is_valid(self) -> bool:
        return torch.isfinite(self._mean) and torch.isfinite(self._log_stdev)
    
    def _get_linear_parameters(self) -> any:
        mean = self._mean
        stdev = torch.exp(self._log_stdev)
        stdev = torch.nan_to_num(stdev, 1e-5)
        return mean, stdev
    
    def _get_distribution(self) -> torch.distributions.Distribution:
        mean, stdev = self._get_linear_parameters()
        distribution = torch.distributions.normal.Normal(loc=mean, scale=stdev)
        return distribution


class Bernoulli(LeafNode):
    def __init__(self, scope: list[int], log_p: torch.Tensor = None) -> None:
        super().__init__(scope)
        if log_p is None:
            log_p = torch.log(torch.rand(1)[0])
        self._log_p = torch.nn.Parameter(log_p, requires_grad=True)

    def __description__(self) -> str:
        with torch.no_grad():
            p = self._get_linear_parameters()
            return f"Br({p.item():.4f}) @ {self.scope}"
        
    def is_valid(self) -> bool:
        with torch.no_grad:
            p = self._get_linear_parameters()
            return 0.0 <= p <= 1.0
    
    def _get_linear_parameters(self) -> any:
        p = torch.exp(self._log_p)
        return p
    
    def _get_distribution(self) -> torch.distributions.Distribution:
        distribution = torch.distributions.bernoulli.Bernoulli(logits=self._log_p)
        return distribution
    
    def cdf(self, data: torch.Tensor, self_extract_data: bool = True) -> torch.Tensor:
        if self_extract_data:
            leaf_data = data[:, self.scope]
        else:
            leaf_data = data
        p = self._get_linear_parameters()
        results = torch.zeros_like(leaf_data)
        mask_data_0 = (leaf_data == 0.0)
        mask_data_1 = (leaf_data >= 1.0)
        results[mask_data_0] = 1.0 - p
        results[mask_data_1] = 1.0
        return results
      

class Binomial(LeafNode):
    def __init__(self, scope: list[int], log_p: torch.Tensor = None, n: int = None):
        super().__init__(scope)
        if log_p is None:
            log_p = torch.log(torch.rand(1)[0])
        if n is None:
            n = torch.randint(low=1, high=10, size=(1,)).item()
        self._log_p = torch.nn.Parameter(log_p, requires_grad=True)
        self._n = torch.nn.Parameter(n, requires_grad=False)

    def __description__(self) -> str:
        with torch.no_grad():
            p, n = self._get_linear_parameters()
            return f"Bm({n}, {p.item():.4f}) @ {self.scope}"
        
    def is_valid(self) -> bool:
        with torch.no_grad():
            n, p = self._get_linear_parameters()
            return n > 0 and 0.0 <= p <= 1.0

    def _get_linear_parameters(self) -> any:
        p = torch.exp(self._log_p)
        n = self._n
        return (p, n)
    
    def cdf(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("No differentiable CDF of Binomial, as Beta functions are still not part of torch (see https://github.com/pytorch/pytorch/issues/133702)")
    

class Categorical(LeafNode):
    def __init__(self, scope: list[int], log_p: torch.Tensor = None, k: int = None):
        super().__init__(scope)
        if log_p is None:
            if k is None:
                raise ValueError("Either 'log_p' or 'k' must not be None.")
            else:
                log_p = torch.randn(size=(k,))
        self._log_p = torch.nn.Parameter(log_p, requires_grad=True)

    def __description__(self) -> str:
        with torch.no_grad():
            p = self._get_linear_parameters()
            output = ["%.4f" % elem for elem in p]
            return f"C({output}) @ {self.scope}"
        
    def is_valid(self) -> bool:
        return torch.all(torch.isfinite(self._log_p))
    
    def _get_linear_parameters(self, temperature: float = 1.0, epsilon: float = 1e-6) -> any:
        p = torch.nn.functional.softmax(self._log_p / temperature, dim=0)
        return p
    
    def _get_distribution(self) -> torch.distributions.Distribution:
        p = self._get_linear_parameters()
        distribution = torch.distributions.categorical.Categorical(probs = p)
        return distribution 
    
    def cdf(self, data: torch.Tensor, self_extract_data: bool = True) -> torch.Tensor:
        """ As the Categorical distribution does not possess as CDF, return its PMF
        """
        if self_extract_data:
            leaf_data = data[:, self.scope]
        else:
            leaf_data = data
        return torch.exp(self.log_pdf(leaf_data))
    
    def test_membership_distribution(self, data: torch.Tensor, test_strategy: TestStrategy) -> torch.Tensor:
        """ As the p-value is defined as the probability obtaining a value at least as extreme as the observed data, and there are no "more extreme" events in a Categorical distribution, return its PMF as p-value.
        """
        return self.cdf(data)
    

class Poisson(LeafNode):
    def __init__(self, scope: list[int], log_lambda: torch.Tensor = None):
        super().__init__(scope)
        if log_lambda is None:
            log_lambda = torch.randn(1)[0]
        self.log_lambda = log_lambda

    def __description__(self) -> str:
        with torch.no_grad():
            l = self._get_linear_parameters()
            return f"P({l.item():.4f}) @ {self.scope}"
        
    def is_valid(self) -> bool:
        with torch.no_grad():
            return torch.isfinite(self.log_lambda)

    def _get_linear_parameters(self) -> any:
        l = torch.exp(self.log_lambda)
        return l
    
    def _get_distribution(self) -> torch.distributions.Distribution:
        l = self._get_linear_parameters()
        distribution = torch.distributions.poisson.Poisson(rate=l)
        return distribution
    
    def cdf(self, data: torch.Tensor, self_extract_data: bool = True) -> torch.Tensor:
        if self_extract_data:
            leaf_data = data[:, self.scope]
        else:
            leaf_data = data
        l = self._get_linear_parameters()
        k = torch.floor(leaf_data + 1.0)
        return torch.special.gammaincc(k, l)
    

