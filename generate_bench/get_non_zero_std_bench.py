""" Substitute corruptions in generated benchmarks to make them having non-zero std value.
The list of corruption selections obtained are saved in a pickle at ../Results/generated_bench

Required: n,k benchmarks must have been generated using generate_n_k_bench.py
"""

import numpy as np
import pickle
import sys
import os
p = os.path.abspath('..')
sys.path.insert(1, p)
import tools

n = int(sys.argv[1])
k = int(sys.argv[2])
bench_size = n*k

accepted_bench_size_threshold = 200

with open(os.path.join("../Results/generated_bench","n={}_k={}_std=0.pkl".format(n,k)), 'rb') as f:
    balanced_benchmarks = pickle.load(f)

with open(os.path.join("../Results/corruption_cat","corruption_clusters.pkl"), 'rb') as f:
    corruption_clusters = pickle.load(f)
nb_clusters = len(list(corruption_clusters.keys()))

with open(os.path.join("../Results/corruption_cat","reverse_corruption_clusters.pkl"), 'rb') as f:
    reverse_clusters = pickle.load(f)

substitution_found = True
saved_benchmarks = []
nb_max_substitution = -1
dic_std_bench = {}
tile_factor = 10
balanced_benchmarks = balanced_benchmarks*tile_factor
unbalanced_benchmarks = list(balanced_benchmarks)

while(substitution_found):
    nb_max_substitution = nb_max_substitution+1
    substitution_found = False
    for j in range(len(unbalanced_benchmarks)):
        cur_unbalanced_bench = tools.substitute_cc(unbalanced_benchmarks[j], reverse_clusters, corruption_clusters)
        if sorted(cur_unbalanced_bench) != sorted(unbalanced_benchmarks[j]):
            substitution_found = True
            unbalanced_benchmarks[j] = cur_unbalanced_bench
        if sorted(cur_unbalanced_bench) not in saved_benchmarks:
            saved_benchmarks.append(sorted(cur_unbalanced_bench))

for i in range(len(saved_benchmarks)):
    cur_repres = list(tools.get_representation_bench(saved_benchmarks[i], reverse_clusters).values())
    cur_repres = [x for x in cur_repres if x !=0]
    cur_std = np.std(cur_repres).round(1)
    if cur_std not in list(dic_std_bench.keys()):
        dic_std_bench[cur_std] = []
    dic_std_bench[cur_std].append(saved_benchmarks[i])

for std in list(dic_std_bench.keys()):
    if len(dic_std_bench[std])> accepted_bench_size_threshold:
        print("std obtained that generates enough benchmarks {}".format(std))
        with open(os.path.join("../Results/generated_bench","n={}_k={}_std={}.pkl".format(n,k,std)), 'wb') as f:
            pickle.dump(dic_std_bench[std], f)
