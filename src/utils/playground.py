import torch
import numpy as np
import random
from src.models.nodewise.utils.get_default_circuits import get_default_circuit_sum_root, get_3d_circuit, generate_random_pc_binary
from src.models.nodewise.distributions import Gaussian, Poisson
from src.models.nodewise.pvalue_combination_functions import TestStrategy, SelectMostProbableChild, FishersMethod, ArithmeticMean, HarmonicMean
from src.models.nodewise.nodes import Node, ProductNode, SumNode, LeafNode
from src.models.nodewise.structural_manipulation import structural_marginalization, structural_conditioning

seed = 47
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)



pc = get_3d_circuit()
data = pc.sample(10)
print(data)

test_strategy = TestStrategy(alpha=0.05, side="both", sum_strategy=ArithmeticMean(), prod_strategy=HarmonicMean())
pvalues = pc.test_membership_distribution(data=data, test_strategy=test_strategy)
print(pvalues)


data[:, [1, 2]] = torch.nan
print(data)

test_strategy = TestStrategy(alpha=0.05, side="both", sum_strategy=ArithmeticMean(), prod_strategy=HarmonicMean())
pvalues = pc.test_membership_distribution(data=data, test_strategy=test_strategy)
print(pvalues)
