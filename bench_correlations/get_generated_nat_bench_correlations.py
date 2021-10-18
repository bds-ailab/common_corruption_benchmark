""" Estimate the correlations in terms of robustness between natural corruption benchmarks and generated benchmarks.
The computed correlation scores and their associated p-values are saved in pickles at ../Results/benchmark_correlations

Require: the accuracies of the models defined in ../models.py for all candidate corruptions (obtained using test_inet_all_candidates.py).
The accuracies of the models defined in ../models.py for all considered natural corruption benchmarks (obtained using test_inet_a.py, test_inet_r.py, test_inet_sk.py, test_inet_v2.py and test_objectnet.py). """

import pandas as pd
import numpy as np
import pickle
import scipy.stats
import os
import sys
p = os.path.abspath('..')
sys.path.insert(1, p)


n = sys.argv[1]
k = sys.argv[2]
std = sys.argv[3]
correlations, p_values = [], []
nb_benchmarks_per_group = 200

with open(os.path.join("../Results/benchmark_correlations","inet_r_accuracies.pkl"), 'rb') as f:
    inet_r_acc = pickle.load(f).squeeze()
with open(os.path.join("../Results/benchmark_correlations","inet_a_accuracies.pkl"), 'rb') as f:
    inet_a_acc = pickle.load(f).squeeze()
with open(os.path.join("../Results/benchmark_correlations","inet_v2_accuracies.pkl"), 'rb') as f:
    inet_v2_acc = pickle.load(f).squeeze()
with open(os.path.join("../Results/benchmark_correlations","onet_accuracies.pkl"), 'rb') as f:
    onet_acc = pickle.load(f).squeeze()
with open(os.path.join("../Results/benchmark_correlations","inet_sk_accuracies.pkl"), 'rb') as f:
    inet_sk_acc = pickle.load(f).squeeze()
with open(os.path.join("../Results/benchmark_correlations","inet_clean_accuracies.pkl"), 'rb') as f:
    inet_clean_acc = pickle.load(f).squeeze()
with open(os.path.join("../Results/benchmark_correlations","inet_all_candidates_accuracies.pkl"), 'rb') as f:
    inet_candidates_acc = pickle.load(f).squeeze()


natural_bench = [inet_a_acc,inet_r_acc,inet_v2_acc,onet_acc,inet_sk_acc]
with open(os.path.join("../Results/generated_bench","n={}_k={}_std={}.pkl".format(n,k,std)), 'rb') as f:
    generated_bench = pickle.load(f)
generated_bench = generated_bench[0:nb_benchmarks_per_group]
correlation_array = np.zeros([len(natural_bench),1])
p_value_array = np.zeros([len(natural_bench),1])


for i in range(len(natural_bench)):
    for j in range(len(generated_bench)):
        cur_generated_bench_acc = inet_candidates_acc[generated_bench[j]].mean(1)
        cur_correlation, cur_p_value = scipy.stats.pearsonr(inet_clean_acc.squeeze()-cur_generated_bench_acc.squeeze(),inet_clean_acc.squeeze()-natural_bench[i].squeeze())
        correlations.append(cur_correlation)
        p_values.append(cur_p_value)

    correlation_array[i,0] = np.mean(cur_correlation)
    p_value_array[i,0] = np.mean(p_values)


correlation_array = pd.DataFrame(correlation_array, index=["inet_a","inet_r","inet_v2","onet","inet_sk"], columns=["n={}_k={}_std={}".format(n,k,std)])
p_value_array = pd.DataFrame(p_value_array, index=["inet_a","inet_r","inet_v2","onet","inet_sk"], columns=["n={}_k={}_std={}".format(n,k,std)])
correlation_array.to_pickle(os.path.join("../Results/benchmark_correlations","n={}_k={}_std={}_correlations.pkl".format(n,k,std)))
correlation_array.to_html(os.path.join("../Results/benchmark_correlations","n={}_k={}_std={}_correlations.html").format(n,k,std))
p_value_array.to_pickle(os.path.join("../Results/benchmark_correlations","n={}_k={}_std={}_p_values.pkl").format(n,k,std))
p_value_array.to_html(os.path.join("../Results/benchmark_correlations","n={}_k={}_std={}_p_values.html").format(n,k,std))
