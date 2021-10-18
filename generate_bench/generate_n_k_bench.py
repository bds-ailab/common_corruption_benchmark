""" Generate benchmarks with n corruption categories represented and k corruptions per represented category.
The list of corruption selections obtained are saved at ../Results/generated_bench

Required: Corruption Categories must have already been defined using get_corruption_clusters.py"""

import os
import pickle
import sys

p = os.path.abspath('..')
sys.path.insert(1, p)
import tools

n = int(sys.argv[1])
k = int(sys.argv[2])
nb_bench_to_get = int(sys.argv[3])
bench_size = n*k
list_bench = []

with open(os.path.join("../Results/corruption_cat","corruption_clusters.pkl"), 'rb') as f:
    corruption_clusters = pickle.load(f)
nb_clusters = len(list(corruption_clusters.keys()))

while(len(list_bench)<nb_bench_to_get):
    cur_bench = sorted(tools.get_balanced_benchmark(bench_size,n,corruption_clusters))
    if cur_bench not in list_bench:
        list_bench.append(cur_bench)

with open(os.path.join("../Results/generated_bench","n={}_k={}_std=0.pkl".format(n,k)), 'wb') as f:
    pickle.dump(list_bench, f)
