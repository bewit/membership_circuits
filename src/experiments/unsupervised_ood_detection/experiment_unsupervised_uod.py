MAX_NUM_PROCESSES = 16

import os

os.environ["OMP_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"
os.environ["MKL_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"

import math
import numpy as np
import random
import torch
from itertools import product
import pandas as pd
from pathlib import Path
from copy import deepcopy
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import pickle as pkl
import traceback

from src.models.nodewise.nodes import SumNode, ProductNode
from src.models.nodewise.distributions import Gaussian
from src.models.nodewise.pvalue_combination_functions import TestStrategy, SelectMostProbableChild, ArithmeticMean, HarmonicMean, Bonferroni, FishersMethod, TippettsMethod
from src.models.nodewise.learn_pc import learn_pc_spflow


# logistics for experiment
experiment_title = "unsupervised_uod"
results_dir = "./src/experiments/unsupervised_ood_detection/results/"
data_dir = "./data/unsupervised_outlier_detection/"
models_dir = "./src/experiments/unsupervised_ood_detection/models/"
Path(models_dir).mkdir(parents=True, exist_ok=True)
file_names = [
    "breast-cancer-unsupervised-ad.csv", 
    "letter-unsupervised-ad.csv", 
    "penglobal-unsupervised-ad.csv",
    "penlocal-unsupervised-ad.csv", 
    "satellite-unsupervised-ad.csv", 
    "annthyroid-unsupervised-ad.csv", 
    "aloi-unsupervised-ad.csv", 
    "shuttle-unsupervised-ad.csv", 
    "kdd99-unsupervised-ad.csv",
    "speech-unsupervised-ad.csv",
]
seeds = [1, 3, 7, 11, 47]


# configuration for experiment
rdc_thresholds = [0.1, 0.3, 0.5]
min_instances_ratios = [0.05, ]
hyper_parameters = [(rdc, minstr) for (rdc, minstr) in product(rdc_thresholds, min_instances_ratios)]
alphas = [0.05,]
test_sides = ["both",]
sum_strategies = [SelectMostProbableChild(weighted=True), ArithmeticMean(use_correction=True, adaptive_correction=True)]
prod_strategies = [HarmonicMean(use_correction=True, adaptive_correction=True), Bonferroni(), FishersMethod(), TippettsMethod()]
test_strategies = [TestStrategy(alpha, side, sum, prod) for (alpha, side, sum, prod) in product(alphas, test_sides, sum_strategies, prod_strategies)]


# setup
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)
torch.set_num_threads(MAX_NUM_PROCESSES)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


for file_name in file_names:
    # load data
    data_path = data_dir + file_name
    dataset_name = file_name.split("-")[0]
    data = pd.read_csv(data_path, header=None, index_col=False)

    # extract train and test sets and remove labels
    labelsstr = data.iloc[:, -1].values
    labels = np.ones_like(labelsstr)
    labels[labelsstr == "o"] = 1
    labels[labelsstr == "n"] = 0
    labels = labels.astype(float)
    data = data.iloc[:, :-1]
    train = data.values
    train_torch = torch.tensor(train)
    test = train
    test_torch = train_torch
    id_size = np.sum(1.0 - labels).astype(int).item()
    ood_size = np.sum(labels).astype(int).item()


    results = []

    for seed in seeds:
        # setup seeds   
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        for rdc, minstr in hyper_parameters:
            model_path = models_dir + f"{experiment_title}_{dataset_name}_r{str(rdc).replace('.', '')}_m{str(minstr).replace('.', '')}_s{str(seed)}.pkl"
            try:
                with open(model_path, "rb") as file:
                    pc = pkl.load(file)

            except Exception as e:
                print(e)
                    
                # learn pc
                min_instances = int(math.ceil(minstr * train.shape[0]))
                distributions_per_scope = [Gaussian] * train.shape[1]
                pc = learn_pc_spflow(data=train, distributions_per_scope=distributions_per_scope, min_instances=min_instances, rdc_threshold=rdc, seed=seed)
                pc.is_valid()

                with open(model_path, "wb") as file:
                    pkl.dump(pc, file)
            

            lls  = pc.log_pdf(test_torch, write_cache=True)
            nll  = (-1) * torch.mean(lls).item()

            y_true = 1 - labels
            y_score = torch.exp(lls).detach().cpu().numpy()
            auroc_densities = roc_auc_score(y_true=y_true, y_score=y_score)
            fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            decisions = (y_score <= optimal_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, decisions).ravel()
            fpr_densities = (fp / id_size).item()
            tpr_densities = (tp / ood_size).item()

            for test_strategy in test_strategies:
                # carry out membership test and evaluate fpr and tpr
                pvalues = pc.test_membership_distribution(data=test_torch, test_strategy=test_strategy)
                decisions = (pvalues <= test_strategy.alpha).int()

                tn, fp, fn, tp = confusion_matrix(labels, decisions).ravel()
                fpr_testing = (fp / id_size).item()
                tpr_testing = (tp / ood_size).item()

                auroc_testing = roc_auc_score(y_true=1.0-labels, y_score=pvalues.detach().cpu().numpy())

                if fpr_testing <= test_strategy.alpha:
                    test_valid = "yes"
                elif fpr_testing <= (test_strategy.alpha * 1.2):
                    test_valid = "tolerated (20%)"
                else:
                    test_valid = "no"
                result = (
                    dataset_name, 
                    seed, 
                    test_strategy.side, 
                    test_strategy.sum_strategy.abbreviation, 
                    test_strategy.prod_strategy.abbreviation, 
                    test_strategy.alpha, 
                    id_size, 
                    ood_size,
                    rdc,
                    minstr,
                    nll, 
                    "auroc",
                    fpr_densities,
                    tpr_densities,
                    auroc_densities, 
                    fpr_testing, 
                    tpr_testing,
                    auroc_testing,
                    test_valid,
                )
                results.append(result)

                print(result[:-1])


    # create results table and store as csv
    results_df = pd.DataFrame(data = results, columns = ["dataset", "seed", "test_side", "sum_strategy", "prod_strategy", "alpha", "id_size", "ood_size", "rdc", "minstr", "nll", "threshold_method", "fpr_densities", "tpr_densities", "auroc_densities", "fpr_testing", "tpr_testing", "auroc_testing", "test_valid"])
    formatted_results_df = results_df.round(decimals=6)

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_filename = results_dir + experiment_title + f"_results_{dataset_name}.csv"
    formatted_results_df.to_csv(results_filename)




