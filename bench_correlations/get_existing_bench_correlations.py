""" Estimate the correlations in terms of robustness between existing natural and synthetic corruption benchmarks.
The computed correlation scores and their associated p-values are saved in pickles at ../Results/benchmark_correlations.

Require: performances on ImageNet-C and ImageNet-P of the models defined in ../models.py (obtained using test_inet_c.py, test_inet_p.py)
The accuracies of the models defined in ../models.py for all considered natural corruption benchmarks (obtained using test_inet_a.py, test_inet_r.py, test_inet_sk.py, test_inet_v2.py and test_objectnet.py) """


import pandas as pd
import numpy as np
import pickle
import scipy.stats
import os
import sys
p = os.path.abspath('..')
sys.path.insert(1, p)

natural_bench_name = ["inet_a","inet_r","inet_v2","onet","inet_sk"]
syn_bench_name = ["inet_c","inet_p"]


natural_bench = []
syn_bench = []
for name in natural_bench_name:
    with open(os.path.join("../Results/benchmark_correlations","{}_accuracies.pkl".format(name)), 'rb') as f:
        model_acc = pickle.load(f).squeeze()
    natural_bench.append(model_acc)

for name in syn_bench_name:
    if name != "inet_p":
        with open(os.path.join("../Results/benchmark_correlations","{}_accuracies.pkl".format(name)), 'rb') as f:
            model_acc = pickle.load(f).squeeze()
    else :
        with open(os.path.join("../Results/benchmark_correlations","inet_p_mfr.pkl"), 'rb') as f:
            model_acc = pickle.load(f).squeeze()
    syn_bench.append(model_acc)

with open(os.path.join("../Results/benchmark_correlations","inet_clean_accuracies.pkl"), 'rb') as f:
    inet_clean_acc = pickle.load(f).squeeze()



correlation_array = np.zeros([len(natural_bench),len(syn_bench)])
p_value_array = np.zeros([len(natural_bench),len(syn_bench)])

for i in range(len(natural_bench)):
    for j in range(len(syn_bench)):
        if syn_bench_name[j] != "inet_p":
            correlation_array[i,j], p_value_array[i,j] = scipy.stats.pearsonr(inet_clean_acc-natural_bench[i],inet_clean_acc-syn_bench[j])
        else:
            correlation_array[i,j], p_value_array[i,j] = scipy.stats.pearsonr(inet_clean_acc-natural_bench[i],syn_bench[j])

correlation_array = pd.DataFrame(correlation_array, index=natural_bench_name, columns=syn_bench_name)
p_value_array = pd.DataFrame(p_value_array, index=natural_bench_name, columns=syn_bench_name)
correlation_array.to_pickle(os.path.join("../Results/benchmark_correlations","existing_bench_correlations.pkl"))
correlation_array.to_html(os.path.join("../Results/benchmark_correlations","existing_bench_correlations.html"))
p_value_array.to_pickle(os.path.join("../Results/benchmark_correlations","existing_bench_p_values.pkl"))
p_value_array.to_html(os.path.join("../Results/benchmark_correlations","existing_bench_p_values.html"))
