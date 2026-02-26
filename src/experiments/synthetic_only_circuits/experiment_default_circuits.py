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
experiment_title = "default_circuits"
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
pc_sum_root = get_default_circuit_sum_root()
pc_sum_root.is_valid()
sum_root_tag = "sum_root"

pc_prod_root = get_default_circuit_prod_root()
pc_prod_root.is_valid()
prod_root_tag = "prod_root"

results = []

for test_strategy in test_strategies:
    collector = {"nll_sum_id": [], "nll_sum_ood": [], "nll_prod_id": [], "nll_prod_ood": [], "fpr_sum": [], "fpr_prod": [], "tpr_sum": [], "tpr_prod": []}

    for seed in seeds:     
        # setup seeds   
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # sample data from both circuits
        samples_sum_root = pc_sum_root.sample(n_samples)
        samples_prod_root = pc_prod_root.sample(n_samples)

        # evalaute NLL and FPR/TPR from both circuits, in-distribution (id) and out-of-distribution (ood)
        # circuit_0: sum, circuit_1: prod
        nll_sum_root_id = (-1) * torch.mean(pc_sum_root.log_pdf(samples_sum_root, write_cache=True)).item()
        pvalues_sum_root_id = pc_sum_root.test_membership_distribution(samples_sum_root, test_strategy=test_strategy)
        decisions_sum_root_id = (pvalues_sum_root_id <= test_strategy.alpha).int()
        fpr_sum_root = decisions_sum_root_id.sum().item() / n_samples

        nll_sum_root_ood = (-1) * torch.mean(pc_sum_root.log_pdf(samples_prod_root, write_cache=True)).item()
        pvalues_sum_root_ood = pc_sum_root.test_membership_distribution(samples_prod_root, test_strategy=test_strategy)
        decisions_sum_root_ood = (pvalues_sum_root_ood <= test_strategy.alpha).int()
        tpr_sum_root = decisions_sum_root_ood.sum().item() / n_samples

        collector["nll_sum_id"].append(nll_sum_root_id)
        collector["nll_sum_ood"].append(nll_sum_root_ood)
        collector["fpr_sum"].append(fpr_sum_root)
        collector["tpr_sum"].append(tpr_sum_root)

        # circuit_0: prod, circuit_1: sum
        nll_prod_root_id = (-1) * torch.mean(pc_prod_root.log_pdf(samples_prod_root, write_cache=True)).item()
        pvalues_prod_root_id = pc_prod_root.test_membership_distribution(samples_prod_root, test_strategy=test_strategy)
        decisions_prod_root_id = (pvalues_prod_root_id <= test_strategy.alpha).int()
        fpr_prod_root = decisions_prod_root_id.sum().item() / n_samples

        nll_prod_root_ood = (-1) * torch.mean(pc_prod_root.log_pdf(samples_sum_root, write_cache=True)).item()
        pvalues_prod_root_ood = pc_prod_root.test_membership_distribution(samples_sum_root, test_strategy=test_strategy)
        decisions_prod_root_ood = (pvalues_prod_root_ood <= test_strategy.alpha).int()
        tpr_prod_root = decisions_prod_root_ood.sum().item() / n_samples

        collector["nll_prod_id"].append(nll_prod_root_id)
        collector["nll_prod_ood"].append(nll_prod_root_ood)
        collector["fpr_prod"].append(fpr_prod_root)
        collector["tpr_prod"].append(tpr_prod_root)

    if np.mean(collector["fpr_sum"]) <= test_strategy.alpha:
        test_valid = "yes"
    elif np.mean(collector["fpr_sum"]) <= (test_strategy.alpha * 1.1):
        test_valid = "tolerated (10%)"
    else:
        test_valid = "no"
    result_sum_root = (
        test_strategy.side, 
        test_strategy.sum_strategy.abbreviation, 
        test_strategy.prod_strategy.abbreviation, 
        test_strategy.alpha, 
        len(seeds), 
        sum_root_tag, 
        prod_root_tag, 
        np.mean(collector["nll_sum_id"]),
        np.std(collector["nll_sum_id"]),
        np.mean(collector["nll_sum_ood"]),
        np.std(collector["nll_sum_ood"]),
        np.mean(collector["fpr_sum"]),
        np.std(collector["fpr_sum"]),
        np.mean(collector["tpr_sum"]),
        np.std(collector["tpr_sum"]),
        test_valid,
        )
    results.append(result_sum_root)


    if np.mean(collector["fpr_prod"]) <= test_strategy.alpha:
        test_valid = "yes"
    elif np.mean(collector["fpr_prod"]) <= (test_strategy.alpha * 1.1):
        test_valid = "tolerated (10%)"
    else:
        test_valid = "no"
    result_prod_root = (
        test_strategy.side, 
        test_strategy.sum_strategy.abbreviation, 
        test_strategy.prod_strategy.abbreviation, 
        test_strategy.alpha,
        len(seeds), 
        prod_root_tag, 
        sum_root_tag, 
        np.mean(collector["nll_prod_id"]),
        np.std(collector["nll_prod_id"]),
        np.mean(collector["nll_prod_ood"]),
        np.std(collector["nll_prod_ood"]),
        np.mean(collector["fpr_prod"]),
        np.std(collector["fpr_prod"]),
        np.mean(collector["tpr_prod"]),
        np.std(collector["tpr_prod"]),
        test_valid,
        )
    results.append(result_prod_root)


# create results table and store as csv
results_df = pd.DataFrame(data = results, columns=["test_side", "sum_strategy", "prod_strategy", "alpha", "no_of_seeds", "circuit_0_tag", "circuit_1_tag", "samples_0_nll_m", "samples_0_nll_s", "samples_1_nll_m", "samples_1_nll_s", "fpr_m", "fpr_s", "tpr_m", "tpr_s", "test_valid"])
formatted_results_df = results_df.round(decimals=4).sort_values(by=["test_valid", "tpr_m"], ascending=False)

Path(results_dir).mkdir(parents=True, exist_ok=True)
results_filename = results_dir + experiment_title + "_results.csv"
formatted_results_df.to_csv(results_filename)