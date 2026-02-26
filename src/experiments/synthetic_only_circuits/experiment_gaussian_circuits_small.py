import numpy as np
import random
import torch
from itertools import product
import pandas as pd
from pathlib import Path

from src.models.nodewise.nodes import SumNode, ProductNode
from src.models.nodewise.distributions import Gaussian
from src.models.nodewise.utils.get_default_circuits import get_default_circuit_sum_root, get_default_circuit_prod_root
from src.models.nodewise.pvalue_combination_functions import TestStrategy, SelectMostProbableChild, ArithmeticMean, HarmonicMean, Bonferroni, FishersMethod, TippettsMethod

# logistics for experiment
experiment_title = "gaussian_circuits_small"
results_dir = "./src/experiments/synthetic_only_circuits/results/"
seeds = [1, 3, 7, 11, 47]

# configuration for experiment
n_samples = 1000
alphas = [0.05,]
test_sides = ["both", "right"]
sum_strategies = [SelectMostProbableChild(weighted=True), ArithmeticMean(use_correction=True, adaptive_correction=False), ArithmeticMean(use_correction=True, adaptive_correction=True)]
prod_strategies = [HarmonicMean(use_correction=True, adaptive_correction=False), HarmonicMean(use_correction=True, adaptive_correction=True), Bonferroni(), FishersMethod(), TippettsMethod()]
test_strategies = [TestStrategy(alpha, side, sum, prod) for (alpha, side, sum, prod) in product(alphas, test_sides, sum_strategies, prod_strategies)]


# setup
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# create default circuits and check for validity
pc0 = get_default_circuit_sum_root()
pc0.is_valid()
pc0_tag = "default_sum_root"

x0 = Gaussian(scope=[0], mean=torch.tensor(-1.0), log_stdev=torch.tensor(0.0))
x1 = Gaussian(scope=[0], mean=torch.tensor(-1.0), log_stdev=torch.tensor(0.0))
y0 = Gaussian(scope=[1], mean=torch.tensor( 0.0), log_stdev=torch.tensor(0.0))
y1 = Gaussian(scope=[1], mean=torch.tensor( 5.0), log_stdev=torch.tensor(0.0))
prod1 = ProductNode(scope=[0, 1], children_circuit=[x0, y0])
prod2 = ProductNode(scope=[0, 1], children_circuit=[x1, y1])
pc1 = SumNode(scope=[0, 1], children_circuit=[prod1, prod2], log_weights=torch.log(torch.tensor([0.3, 0.7])))
pc1.is_valid()
pc1_tag = "shifted_sum_root_cc"

results = []

for test_strategy in test_strategies:
    collector = {"nll_pc0_id": [], "nll_pc0_ood": [], "nll_pc1_id": [], "nll_pc1_ood": [], "fpr_pc0": [], "fpr_pc1": [], "tpr_pc0": [], "tpr_pc1": []}

    for seed in seeds:     
        # setup seeds   
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # sample data from both circuits
        samples_pc0 = pc0.sample(n_samples)
        samples_pc1 = pc1.sample(n_samples)

        # evalaute NLL and FPR/TPR from both circuits, in-distribution (id) and out-of-distribution (ood)
        # circuit_0: sum, circuit_1: prod
        nll_pc0_id = (-1) * torch.mean(pc0.log_pdf(samples_pc0, write_cache=True)).item()
        pvalues_pc0_id = pc0.test_membership_distribution(samples_pc0, test_strategy=test_strategy)
        decisions_pc0_id = (pvalues_pc0_id <= test_strategy.alpha).int()
        fpr_pc0 = decisions_pc0_id.sum().item() / n_samples

        nll_pc0_ood = (-1) * torch.mean(pc0.log_pdf(samples_pc1, write_cache=True)).item()
        pvalues_pc0_ood = pc0.test_membership_distribution(samples_pc1, test_strategy=test_strategy)
        decisions_pc0_ood = (pvalues_pc0_ood <= test_strategy.alpha).int()
        tpr_pc0 = decisions_pc0_ood.sum().item() / n_samples

        collector["nll_pc0_id"].append(nll_pc0_id)
        collector["nll_pc0_ood"].append(nll_pc0_ood)
        collector["fpr_pc0"].append(fpr_pc0)
        collector["tpr_pc0"].append(tpr_pc0)

        # circuit_0: prod, circuit_1: sum
        nll_pc1_id = (-1) * torch.mean(pc1.log_pdf(samples_pc1, write_cache=True)).item()
        pvalues_pc1_id = pc1.test_membership_distribution(samples_pc1, test_strategy=test_strategy)
        decisions_pc1_id = (pvalues_pc1_id <= test_strategy.alpha).int()
        fpr_pc1 = decisions_pc1_id.sum().item() / n_samples

        nll_pc1_ood = (-1) * torch.mean(pc1.log_pdf(samples_pc0, write_cache=True)).item()
        pvalues_pc1_ood = pc1.test_membership_distribution(samples_pc0, test_strategy=test_strategy)
        decisions_pc1_ood = (pvalues_pc1_ood <= test_strategy.alpha).int()
        tpr_pc1 = decisions_pc1_ood.sum().item() / n_samples

        collector["nll_pc1_id"].append(nll_pc1_id)
        collector["nll_pc1_ood"].append(nll_pc1_ood)
        collector["fpr_pc1"].append(fpr_pc1)
        collector["tpr_pc1"].append(tpr_pc1)


    if np.mean(collector["fpr_pc0"]) <= test_strategy.alpha:
        test_valid = "yes"
    elif np.mean(collector["fpr_pc0"]) <= (test_strategy.alpha * 1.1):
        test_valid = "tolerated (10%)"
    else:
        test_valid = "no"
    result_pc0 = (
        test_strategy.side, 
        test_strategy.sum_strategy.abbreviation, 
        test_strategy.prod_strategy.abbreviation, 
        test_strategy.alpha, 
        len(seeds),
        pc0_tag, 
        pc1_tag, 
        np.mean(collector["nll_pc0_id"]),
        np.std(collector["nll_pc0_id"]),
        np.mean(collector["nll_pc0_ood"]),
        np.std(collector["nll_pc0_ood"]),
        np.mean(collector["fpr_pc0"]),
        np.std(collector["fpr_pc0"]),
        np.mean(collector["tpr_pc0"]),
        np.std(collector["tpr_pc0"]),
        test_valid,
        )
    results.append(result_pc0)


    if np.mean(collector["fpr_pc1"]) <= test_strategy.alpha:
        test_valid = "yes"
    elif np.mean(collector["fpr_pc1"]) <= (test_strategy.alpha * 1.1):
        test_valid = "tolerated (10%)"
    else:
        test_valid = "no"
    result_pc1 = (
        test_strategy.side, 
        test_strategy.sum_strategy.abbreviation, 
        test_strategy.prod_strategy.abbreviation, 
        test_strategy.alpha, 
        len(seeds),
        pc1_tag, 
        pc0_tag, 
        np.mean(collector["nll_pc1_id"]),
        np.std(collector["nll_pc1_id"]),
        np.mean(collector["nll_pc1_ood"]),
        np.std(collector["nll_pc1_ood"]),
        np.mean(collector["fpr_pc1"]),
        np.std(collector["fpr_pc1"]),
        np.mean(collector["tpr_pc1"]),
        np.std(collector["tpr_pc1"]),
        test_valid,
        )
    results.append(result_pc1)


# create results table and store as csv
results_df = pd.DataFrame(data = results, columns=["test_side", "sum_strategy", "prod_strategy", "alpha", "no_of_seeds", "circuit_0_tag", "circuit_1_tag", "samples_0_nll_m", "samples_0_nll_s", "samples_1_nll_m", "samples_1_nll_s", "fpr_m", "fpr_s", "tpr_m", "tpr_s", "test_valid"])
formatted_results_df = results_df.round(decimals=4).sort_values(by=["test_valid", "tpr_m"], ascending=False)

Path(results_dir).mkdir(parents=True, exist_ok=True)
results_filename = results_dir + experiment_title + "_results.csv"
formatted_results_df.to_csv(results_filename)

# save configs of pcs
pc0_config_filename = results_dir + experiment_title + "_config_pc0.txt"
with open(pc0_config_filename, "w") as file:
    file.write(str(pc0))

pc1_config_filename = results_dir + experiment_title + "_config_pc1.txt"
with open(pc1_config_filename, "w") as file:
    file.write(str(pc1))