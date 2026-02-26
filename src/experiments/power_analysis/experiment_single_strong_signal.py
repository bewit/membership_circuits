MAX_NUM_PROCESSES = 16

import os
os.environ["OMP_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"
os.environ["MKL_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"

import numpy as np
import random
import torch
from itertools import product
import pandas as pd
from pathlib import Path
from copy import deepcopy

from src.models.nodewise.nodes import SumNode, ProductNode
from src.models.nodewise.distributions import Gaussian
from src.models.nodewise.utils.get_default_circuits import generate_random_pc_binary, generate_random_pc
from src.models.nodewise.pvalue_combination_functions import TestStrategy, SelectMostProbableChild, ArithmeticMean, HarmonicMean, Bonferroni, FishersMethod, TippettsMethod

# logistics for experiment
experiment_title = "single_strong_signal"
results_dir = "./src/experiments/power_analysis/results/"
seeds = [1, 3, 7, 11, 47]

# configuration for experiment
random_pc_number_of_rvs = [10, 100, 1000]
random_pc_leaftype = Gaussian
random_pc_leaves_parameters = {"log_stdev": torch.tensor(0.0)}
signal_min = 0.0
signal_max = 5.0
signal_incremental = 0.1
n_samples = 5000
ood_ratio = 0.2 # 20% of instances are injected with signal
alphas = [0.05, 0.01, 0.001]
test_sides = ["both", "right"]
sum_strategies = [SelectMostProbableChild(weighted=True), ArithmeticMean(use_correction=True, adaptive_correction=False), ArithmeticMean(use_correction=True, adaptive_correction=True)]
prod_strategies = [HarmonicMean(use_correction=True, adaptive_correction=False), HarmonicMean(use_correction=True, adaptive_correction=True), Bonferroni(), FishersMethod(), TippettsMethod()]
test_strategies = [TestStrategy(alpha, side, sum, prod) for (alpha, side, sum, prod) in product(alphas, test_sides, sum_strategies, prod_strategies)]


# setup
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)
torch.set_num_threads(MAX_NUM_PROCESSES)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


results = []

with torch.inference_mode():

    for no_rvs in random_pc_number_of_rvs:
        random_pc_depth = 2 * int(np.log2(no_rvs)) + 1
            
        for seed in seeds:
            # setup seeds   
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # create random circuit 
            pc = generate_random_pc_binary(num_variables=no_rvs, max_depth=random_pc_depth, leafnode=random_pc_leaftype, leaves_parameters=random_pc_leaves_parameters)
            pc.is_valid()
            pc_tag = f"random_binary_pc_rvs{no_rvs}_d{random_pc_depth}_leaf{random_pc_leaftype.__name__}"

            # generate indices for the instances to be injected with signals
            ood_indices = np.sort(np.random.choice(np.arange(n_samples), size=int(ood_ratio*n_samples), replace=False))
            ood_mask = torch.zeros(n_samples)
            ood_mask[ood_indices] = 1
            ood_mask = ood_mask.bool()

            # determine sizes of id and ood sets
            id_size = (~ood_mask).int().sum().item()
            ood_size = ood_mask.int().sum().item()

            # sample data from circuit
            samples = pc.sample(n_samples)

            # inject selected indices with signal 
            for signal_strength in np.arange(signal_min, signal_max+signal_incremental, signal_incremental):
                injected_samples = deepcopy(samples)
                injected_samples[ood_indices, 0] += signal_strength

                # compute model fit w.r.t. ID and OOD samples
                nlls = pc.log_pdf(injected_samples, write_cache=True)
                nlls_id = nlls[~ood_mask, :]
                nlls_ood = nlls[ood_mask, :]
                nll_id = (-1) * torch.mean(nlls_id).item()
                nll_ood = (-1) * torch.mean(nlls_ood).item()

                for test_strategy in test_strategies:
                    # carry out membership test and evaluate fpr and tpr
                    pvalues = pc.test_membership_distribution(data=injected_samples, test_strategy=test_strategy)
                    decisions = (pvalues <= test_strategy.alpha).int()
                    fpr = torch.sum(decisions[~ood_mask]).item() / id_size
                    tpr = torch.sum(decisions[ood_mask]).item() / ood_size

                    if fpr <= test_strategy.alpha:
                        test_valid = "yes"
                    elif fpr <= (test_strategy.alpha * 1.1):
                        test_valid = "tolerated (10%)"
                    else:
                        test_valid = "no"
                    result = (
                        test_strategy.side, 
                        test_strategy.sum_strategy.abbreviation, 
                        test_strategy.prod_strategy.abbreviation, 
                        test_strategy.alpha, 
                        pc_tag,
                        no_rvs,
                        seed,
                        id_size, 
                        ood_size,
                        float(signal_strength),
                        nll_id, 
                        nll_ood, 
                        fpr, 
                        tpr,
                        test_valid,
                        )
                    results.append(result)

                    print(result)

        # save configs of pcs
        pc0_config_filename = results_dir + experiment_title + f"_config_pc_rvs{no_rvs}.txt"
        with open(pc0_config_filename, "w") as file:
            file.write(str(pc))


# create results table and store as csv
results_df = pd.DataFrame(data = results, columns = ["test_side", "sum_strategy", "prod_strategy", "alpha", "pc_tag", "no_of_rvs", "seed", "id_size", "ood_size", "signal_strength", "nll_id", "nll_ood", "fpr", "tpr", "test_valid"])
formatted_results_df = results_df.round(decimals=4)

Path(results_dir).mkdir(parents=True, exist_ok=True)
results_filename = results_dir + experiment_title + "_results.csv"
formatted_results_df.to_csv(results_filename)




