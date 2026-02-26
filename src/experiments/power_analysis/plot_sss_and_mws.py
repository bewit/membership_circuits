import numpy as np
import random
import torch
from itertools import product
import pandas as pd
from pathlib import Path
from copy import deepcopy
from matplotlib import pyplot as plt
import scienceplots

from src.models.nodewise.nodes import SumNode, ProductNode
from src.models.nodewise.distributions import Gaussian
from src.models.nodewise.utils.get_default_circuits import generate_random_pc_binary, generate_random_pc
from src.models.nodewise.pvalue_combination_functions import TestStrategy, SelectMostProbableChild, ArithmeticMean, HarmonicMean, Bonferroni, FishersMethod, TippettsMethod

plt.rcParams.update({"font.size": 20})
plt.style.use(["no-latex", "grid"])
color_list_shade0 = ["#125156", "#792e00", "#352f4d", "#60001f"]
color_list_shade1 = ["#1b7b83", "#b74500", "#504673", "#8e002f"]
color_list_shade2 = ["#24a5af", "#f55a00", "#6b5d99", "#bf0040"]
color_list_shade3 = ["#69d8e0", "#ff9b60", "#a59cc2", "#ff407e"]
color_list = ["#24a5af", "#b74500", "#24a5af", "#b74500"]
line_styles = {10: "-", 100: "--"}

fig, (ax_sss, ax_mws) = plt.subplots(ncols=2)
fig.set_size_inches(12, 5)
# plt.subplots_adjust(bottom=0.25)


# logistics for experiment
experiment_title_sss = "single_strong_signal_learned"
experiment_title_mws = "multiple_weak_signals_learned"
results_dir_sss = "./src/experiments/b_single_strong_signal/results/"
results_dir_mws = "./src/experiments/c_multiple_weak_signals/results/"
plots_dir = "./src/experiments/b_single_strong_signal/plots/"
seeds = [1, 3, 7, 11, 47]

# configuration for experiment
random_pc_number_of_rvs = [10, 100]
random_pc_leaftype = Gaussian
random_pc_leaves_parameters = {"log_stdev": torch.tensor(0.0)}
signal_min = 0.0
signal_max = 5.0
signal_incremental = 0.25
signal_strengths = np.arange(signal_min, signal_max+signal_incremental, signal_incremental)
n_samples = 1000
ood_ratio = 0.2 # 20% of instances are injected with signal


target_column = "tpr"
alpha = 0.05
test_side = "both"
test_strategies = [
    TestStrategy(alpha, test_side, ArithmeticMean(), FishersMethod(), title="Aggr."),
    TestStrategy(alpha, test_side, SelectMostProbableChild(), TippettsMethod(), title="Sel."),
    # TestStrategy(alpha, test_side, ArithmeticMean(), Bonferroni(), title=r"$\bigoplus: \text{Aggregate}, \bigotimes: \text{Select}$"),
    # TestStrategy(alpha, test_side, ArithmeticMean(), FishersMethod(), title="Aggregation"),
]


data_path = results_dir_sss + experiment_title_sss + "_results.csv"
data = pd.read_csv(data_path, index_col=0)


for i, test_strategy in enumerate(test_strategies):
    for number_of_rvs in random_pc_number_of_rvs:
        alpha = test_strategy.alpha
        test_side = test_strategy.side
        sum_strategy = test_strategy.sum_strategy.abbreviation
        prod_strategy = test_strategy.prod_strategy.abbreviation

        strategy_data = data[(data.alpha==alpha) & (data.test_side==test_side) & (data.sum_strategy == sum_strategy) & (data.prod_strategy == prod_strategy)]
        strategy_data = strategy_data[strategy_data.no_of_rvs == number_of_rvs]

        signal_means = []
        signal_stdevs = []
        for signal_strength in signal_strengths:
            signal_data = strategy_data[strategy_data.signal_strength.round(2) == signal_strength.round(2)]
            
            mean = np.nanmean(signal_data[target_column])
            stdev = np.nanstd(signal_data[target_column])

            signal_means.append(mean)
            signal_stdevs.append(stdev)

        signal_means = np.array(signal_means)
        signal_stdevs = np.array(signal_stdevs)
        signal_stdevs = signal_stdevs / np.sqrt(len(seeds))

        if number_of_rvs == 10:
            color = color_list[i]
            linestyle = line_styles[10]
        if number_of_rvs == 100:
            color = color_list[i]
            linestyle = line_styles[100]
        label = test_strategy.title + f" $(d={number_of_rvs})$"
        ax_sss.plot(signal_strengths, signal_means, label=label, color=color, linestyle=linestyle)
        ax_sss.fill_between(signal_strengths, signal_means-signal_stdevs, signal_means+signal_stdevs, alpha=0.2, color=color, linestyle=linestyle)


ax_sss.set_xlabel(r"Signal Strength ($\sigma$)")
ax_sss.set_ylabel("True Positive Rate")
ax_sss.set_xticks([0, 1, 2, 3, 4, 5])
ax_sss.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax_sss.set_ylim(-0.05, 1.05)    
# plt.axhline(y=1.0, linestyle="--", c="black", alpha=0.5, xmin=0, xmax=1)
# ax_sss.legend(loc="upper center", framealpha=1.0, bbox_to_anchor=(0.5, 0.05), ncol=4)
# plt.title("Single Strong Signal")



# configuration for experiment
random_pc_number_of_rvs = [10, 100]
random_pc_leaftype = Gaussian
random_pc_leaves_parameters = {"log_stdev": torch.tensor(0.0)}
signal_strength = 1.0
dimensions_incremental = 0.1
dimension_ratios = np.arange(0.0, 1.0 + dimensions_incremental, dimensions_incremental)
dimension_ratios_int = (dimension_ratios * 100).astype(int)


data_path = results_dir_mws + experiment_title_mws + "_results.csv"
data = pd.read_csv(data_path, index_col=0)


for i, test_strategy in enumerate(test_strategies):
    for number_of_rvs in random_pc_number_of_rvs:
        alpha = test_strategy.alpha
        test_side = test_strategy.side
        sum_strategy = test_strategy.sum_strategy.abbreviation
        prod_strategy = test_strategy.prod_strategy.abbreviation

        strategy_data = data[(data.alpha==alpha) & (data.test_side==test_side) & (data.sum_strategy == sum_strategy) & (data.prod_strategy == prod_strategy)]
        strategy_data = strategy_data[strategy_data.no_of_rvs == number_of_rvs]

        signal_means = []
        signal_stdevs = []
        for dimension_ratio in dimension_ratios:
            signal_data = strategy_data[strategy_data.dimensions_ratio.round(2) == dimension_ratio.round(2)]
            
            mean = np.nanmean(signal_data[target_column])
            stdev = np.nanstd(signal_data[target_column])

            signal_means.append(mean)
            signal_stdevs.append(stdev)

        signal_means = np.array(signal_means)
        signal_stdevs = np.array(signal_stdevs)
        signal_stdevs = signal_stdevs / np.sqrt(len(seeds))


        if number_of_rvs == 10:
            color = color_list[i]
            linestyle = line_styles[10]
        if number_of_rvs == 100:
            color = color_list[i]
            linestyle = line_styles[100]
        label = test_strategy.title + f" $(d={number_of_rvs})$"
        ax_mws.plot(dimension_ratios_int, signal_means, label=label, color=color, linestyle=linestyle)
        ax_mws.fill_between(dimension_ratios_int, signal_means-signal_stdevs, signal_means+signal_stdevs, alpha=0.2, color=color, linestyle=linestyle)


ax_mws.set_xlabel("Dimensions with Signal (%)")
# ax_mws.set_ylabel("True Positive Rate")
ax_mws.set_xticks([0, 25, 50, 75, 100])
ax_mws.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
# plt.xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
ax_mws.set_ylim(-0.05, 1.05)    
# plt.axhline(y=1.0, linestyle="--", c="black", alpha=0.5, xmin=0, xmax=1)
# plt.legend(loc="upper left")
# plt.title("Multiple Weak Signals")


ax_mws.legend(loc="upper center", framealpha=1.0, bbox_to_anchor=(-0.19, -0.17), ncol=4, columnspacing=0.7, handletextpad=0.4)


# plt.tight_layout()

Path(plots_dir).mkdir(exist_ok=True, parents=True)
plot_path = plots_dir + "plot_sss_and_mws.pdf"
plt.savefig(plot_path, dpi=500, bbox_inches="tight")