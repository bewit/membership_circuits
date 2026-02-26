from typing import Literal
import numpy as np
import torch
from abc import ABC

from src.models.nodewise import nodes
from src.utils.means import harmonic_mean_2d
from scipy.optimize import root_scalar
from scipy.stats import  t
from scipy.special import binom
from functools import cache

STANDARD_NORMAL = torch.distributions.Normal(loc=0.0, scale=1.0)


class TestStrategy:
    def __init__(self, alpha: float, side: Literal["both", "right", "left"], sum_strategy: "PValueCombinationStrategy", prod_strategy: "PValueCombinationStrategy", title: str = None):
        self.alpha = alpha
        self.side = side
        self.sum_strategy = sum_strategy
        self.prod_strategy = prod_strategy
        self.title = title

    def get_name(self):
        return f"{self.side}_{str(self.alpha).replace('.', '')}_S_{self.sum_strategy.abbreviation}_P_{self.prod_strategy.abbreviation}"
    
    def get_wandb_identifier(self):
        return f"{self.side}/S_{self.sum_strategy.abbreviation}_P_{self.prod_strategy.abbreviation}"
    

class PValueCombinationStrategy(ABC):
    abbreviation = "PVal"

    def __description__(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return self.__description__()
    
    def __str__(self):
        return self.__description__()

    def execute(self, pvalues: torch.Tensor, node: "Node"):
        raise NotImplementedError
    

class StrategyNotSetError(Exception):
    pass


class EmptyStrategy(PValueCombinationStrategy):
    abbreviation = "None"

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        raise StrategyNotSetError(f"You need to set a PValueCombinationStrategy for {type(node)} before combining p-values (e.g. by passing it to the test function)")


class ArithmeticMean(PValueCombinationStrategy):
    abbreviation = "AM_"

    def __init__(self, use_correction: bool = True, adaptive_correction: bool = True):
        self.use_correction = use_correction
        self.adaptive_correction = adaptive_correction

        if not use_correction and not adaptive_correction:
            self.abbreviation = "AMean"
        elif use_correction and not adaptive_correction:
            self.abbreviation = "AMean_fix"
        elif use_correction and adaptive_correction:
            self.abbreviation = "AMean_ada"

        super().__init__()

    def __description__(self):
        return f"{self.__class__.__name__}(use_correction={self.use_correction}, adaptive={self.adaptive_correction})"

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        # implementation in linear space for debugging
        # TODO: put computations in log space
        if type(node) == nodes.SumNode:
            linear_weights = node._get_linear_weights()
        if type(node) == nodes.ProductNode:
            n = len(node.children_circuit)
            linear_weights = torch.ones((n, 1)) / n

        if self.use_correction:
            if self.adaptive_correction:
                correction_factor = torch.minimum(torch.tensor(2.0), 1.0 / torch.max(linear_weights))
            else:
                correction_factor = 2.0
        else: 
            correction_factor = 1.0

        combined_pvalue = (correction_factor * (pvalues @ linear_weights)).view(-1, 1)
        return combined_pvalue

        # return torch.logsumexp(pvalues + weights + correction_factor, dim=1, keepdim=True)
    

class Multiplication(PValueCombinationStrategy):
    abbreviation = "Mult"

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        return torch.prod(pvalues, dim=1, keepdim=True)
    

class HarmonicMean(PValueCombinationStrategy):
    abbreviation = "HM_"

    def __init__(self, use_correction: bool = True, adaptive_correction: bool = True, admissible_form: bool = False):
        self.use_correction = use_correction
        self.adaptive_correction = adaptive_correction
        self.admissible_form = admissible_form

        if not use_correction and not adaptive_correction:
            self.abbreviation = "HMean"
        elif use_correction and not adaptive_correction:
            self.abbreviation = "HMean_fix"
        elif use_correction and adaptive_correction:
            self.abbreviation = "HMean_ada"
        if admissible_form:
            self.abbreviation += "_adm"

        super().__init__()

    def __description__(self):
        return f"{self.__class__.__name__}(use_correction={self.use_correction})"

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        K = pvalues.shape[1]

        if type(node) == nodes.SumNode:
            weights = node._get_linear_weights()
        else:
            weights = torch.ones((K, 1)) / K

        if self.admissible_form:
            dr_alpha = self._dr_alpha(K=K)
            admissible_pvalues = torch.minimum(pvalues, dr_alpha)
            pvalues = admissible_pvalues

        if self.use_correction:
            if self.adaptive_correction:
                if K > 2: 
                    correction_factor = self._compute_adaptive_correction_factor(K)
                else:
                    correction_factor = torch.e * torch.log(torch.tensor(K))
            else:
                correction_factor = torch.e * torch.log(torch.tensor(K)) 
        else:
            correction_factor = 1.0
        hmean = harmonic_mean_2d(pvalues, weights, dim=1)
        combined_pvalue = correction_factor * hmean
        return combined_pvalue
        
    @cache
    def _compute_adaptive_correction_factor(self, K):
        def compute_y(y, K):
            return y**2 - K * ((y + 1) * np.log(y + 1) - y)
        def compute_akh(K):
            y_k = root_scalar(compute_y, args=(K,), bracket=[1e-6, 1e6], method="brentq").root
            akh = (y_k+K)**2 / ((y_k+1)*K)
            return akh
        
        akh = compute_akh(K)
        correction_factor = akh * np.log(K)
        return correction_factor
    
    @cache
    def _dr_alpha(self, K, alpha=0.05, epsilon=1e-10):
        def compute_cr(c):
            return ((1-K*c) / (K*c*(1 - (K-1)*c))) - np.log((1/c) - (K-1))
        cr = root_scalar(compute_cr, bracket=[1e-10, 1/K], method="brentq").root
        dr = 1 - (K - 1) * cr
        assert 0 <= cr <= 1/K + epsilon, (K, cr)
        assert 1/K - epsilon <= dr <= 1, (K, dr)
        dr_alpha = dr * alpha * np.ones((1,K))

        return torch.tensor(dr_alpha)


class SelectMostProbableChild(PValueCombinationStrategy):
    abbreviation = "Slct"

    def __init__(self, weighted: bool = True):
        self.weighted = weighted
        super().__init__()

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        if type(node) == nodes.SumNode:
            linear_weights = node._get_linear_weights().view(1, -1)
        elif type(node) == nodes.ProductNode:
            n = len(node.children_circuit)
            linear_weights = torch.ones((1, n)) / n

        with torch.no_grad():
            children_densities = []
            for child in node.children_circuit:
                if child._results_cache is None:
                    raise ValueError("SelectMostProbableChild wants to access cached densities, but child._results_cache was None")
                child_densities = child._results_cache
                if torch.any(child_densities < 0.0):
                    child_densities = torch.exp(child_densities)
                children_densities.append(child_densities)
            concat_children = torch.cat(children_densities, dim=1)
            if self.weighted:
                children_densities = linear_weights * concat_children
            else:
                children_densities = concat_children
            indexes = torch.argmax(children_densities, dim=1)
            selector = torch.nn.functional.one_hot(indexes, num_classes=len(node.children_circuit))
        
        combined_pvalue = torch.sum(selector * pvalues, dim=1).view(-1, 1)
        return combined_pvalue
    

class Bonferroni(PValueCombinationStrategy):
    abbreviation = "Bonf"

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        K = pvalues.shape[1] # number of tests
        # p_min = torch.min(pvalues, dim=1, keepdim=True).values # most significant test
        with torch.no_grad():
            indexes  = torch.argmin(pvalues, dim=1)
            selector = torch.nn.functional.one_hot(indexes, num_classes=K)
        p_min = torch.sum(selector * pvalues, dim=1).view(-1, 1)
        combined_pvalue = K * p_min
        return combined_pvalue
    

class TippettsMethod(PValueCombinationStrategy):
    abbreviation = "Tipp"

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        K = pvalues.shape[1] # number of tests
        # p_min = torch.min(pvalues, dim=1, keepdim=True).values # most significant test
        with torch.no_grad():
            indexes  = torch.argmin(pvalues, dim=1)
            selector = torch.nn.functional.one_hot(indexes, num_classes=K)
        p_min = torch.sum(selector * pvalues, dim=1).view(-1, 1)
        combined_pvalue = 1.0 - ((1.0 - p_min) ** K)
        return combined_pvalue


class FishersMethod(PValueCombinationStrategy):
    abbreviation = "Fish"

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        K = pvalues.shape[1]
        T = -2 * pvalues.log().sum(dim=1)
        chi2 = torch.distributions.Gamma(concentration=0.5*(2*K), rate=0.5)
        combined_pvalue = 1 - chi2.cdf(T).view(-1, 1)
        return combined_pvalue


class StouffersMethod(PValueCombinationStrategy):
    abbreviation = "Stouf"

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        K = pvalues.shape[1]
        T_constant = 1 / torch.sqrt(torch.tensor(K)) 
        T_variable = (1 / STANDARD_NORMAL.cdf(1 - pvalues)).sum(dim=1)
        T = T_constant * T_variable
        combined_pvalue = 1 - STANDARD_NORMAL.cdf(T).view(-1, 1)
        return combined_pvalue

class WilkinsonsMethod(PValueCombinationStrategy):
    abbreviation = "Wilk"
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        super().__init__()

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        def _combined_pvalue(T, K):
            return sum([binom(K, k) * self.alpha**k * (1-self.alpha)**(K-k) for k in range(T, K+1)])
        K = pvalues.shape[1]
        significant_tests = pvalues <= self.alpha
        T = significant_tests.int().sum(dim=1)
        combined_pvalue = torch.tensor([_combined_pvalue(t, K) for t in T]).view(-1, 1)
        return combined_pvalue
    

class MudholkarGeorgesMethod(PValueCombinationStrategy):
    abbreviation = "MG"

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        K = pvalues.shape[1]
        T_constant = 1/torch.pi * torch.sqrt(torch.tensor(3*(5*K+4)) / (K*(5*K+2)))
        T_variable = ((1 - pvalues) / pvalues).log().sum(dim=1)
        T = T_constant * T_variable
        combined_pvalue = 1.0 - torch.tensor(t.cdf(T, df=5*K+4)).view(-1, 1)
        return combined_pvalue
    

class TaylorTibshiranisMethod(PValueCombinationStrategy):
    abbreviation = "TT"

    def execute(self, pvalues: torch.Tensor, node: "Node") -> torch.Tensor:
        K = pvalues.shape[1]
        K_range = torch.arange(1, K+1)
        T_constant = 1 / K
        T_variable = (1 - pvalues/K_range * (K+1)).sum(dim=1)
        T = T_constant * T_variable
        combined_pvalue = 1 - torch.distributions.Normal(loc=0.0, scale=1/torch.sqrt(torch.tensor(K))).cdf(T).view(-1, 1)
        return combined_pvalue


