""" Apply the Kmeans algorithm using the vectors of overlapping scores associated with each candidate corruption.
The number of centroids is increased until the Same Cluster Corruptions have a mean overlapping score above 0.5
The obtained clusters are saved in a pickle at ../Results/corruption_cat

Required: the overlapping matrix obtained using get_overlapping_matrix.py"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
import seaborn as sb
import os
import sys

p = os.path.abspath('..')
sys.path.insert(1, p)
from candidate_corruptions import dict_corruptions

scc_overlapping_threshold = 0.5

scc = []
dcc = []

with open(os.path.join("../Results/corruption_cat",'overlappings_matrix.pkl'), 'rb') as f:
    overlappings = pickle.load(f)

for nb_clusters in range(2,len(list(dict_corruptions.keys()))+1):
    kmeans = KMeans(n_clusters=nb_clusters, n_init=100, max_iter=1000).fit(overlappings)
    clusters = [[] for i in range(nb_clusters)]
    for i in range(len(kmeans.labels_)):
        clusters[kmeans.labels_[i]].append(overlappings.index[i])
    scc_cor, dcc_cor = [], []
    for i in range(nb_clusters):
        for j in range(nb_clusters):
            for k in range(len(clusters[i])):
                for h in range(len(clusters[j])):
                    cur_corr = np.corrcoef(overlappings[:][clusters[i][k]], overlappings[:][clusters[j][h]])
                    if i == j:
                        if clusters[i][k] != clusters[j][h]:
                            scc_cor.append(cur_corr[0,1])
                    elif i != j:
                        dcc_cor.append(cur_corr[0,1])
    if np.mean(scc_cor) > scc_overlapping_threshold:
        print("SCC overlapping threshold reached")
        print(nb_clusters)
        break

corruption_clusters, reverse_clusters = {},{}
for i in range(nb_clusters):
    corruption_clusters[str(i)] = []
for i in range(len(list(kmeans.labels_))):
    corruption_clusters[str(kmeans.labels_[i])].append(list(overlappings.index)[i])
    reverse_clusters[list(overlappings.index)[i]] = str(kmeans.labels_[i])

with open(os.path.join("../Results/corruption_cat","corruption_clusters.pkl"), 'wb') as f:
     pickle.dump(corruption_clusters,f)
with open(os.path.join("../Results/corruption_cat","reverse_corruption_clusters.pkl"), 'wb') as f:
     pickle.dump(reverse_clusters,f)

dic_clusters =  dict(zip(list(overlappings.index),kmeans.labels_))
dic_clusters = dict(sorted(dic_clusters.items(), key=lambda item: item[1]))

new_corruption_order = list(dic_clusters.keys())
reordered_overlappings = overlappings.reindex(new_corruption_order)
reordered_overlappings = reordered_overlappings[:][new_corruption_order]
reordered_overlappings.to_pickle(os.path.join("../Results/corruption_cat", "reordered_overlappings_by_cluster.pkl"))

heatmap = sb.heatmap(reordered_overlappings.round(2), vmin=0, vmax=1, cmap="Greens", annot=True, annot_kws={"size": 9}, xticklabels=1, yticklabels=1)
figure = plt.gcf()
figure.set_size_inches(21, 12)
plt.savefig(os.path.join("../Results/corruption_cat", "reordered_overlappings_by_cluster.png"),bbox_inches='tight')
