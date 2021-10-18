""" For all possible pairs of corruptions c1,c2, in the set of candidate corruptions, compute the overlapping score between c1 and c2
The computed overlapping scores are saved at ../Results/corruption_cat

Required: the accuracies used to get the overlapping scores must have already been computed with get_candidate_acc.py """

import pandas as pd
import os
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import sys
p = os.path.abspath('..')
sys.path.insert(1, p)

rob_array = pd.read_pickle(os.path.join("../Results/corruption_cat","candidate_model_acc.pkl"))

nb_corruptions = (len(list(rob_array.columns))-1)
corruption_list = list(rob_array.columns)
corruption_list.remove("clean")
rob_array = rob_array.to_numpy()

overlappings = np.zeros([nb_corruptions,nb_corruptions])
for i in range(1,nb_corruptions+1):
    for j in range(i,nb_corruptions+1):
        cur_orverlap_score = 0.5*((rob_array[i,j] - rob_array[0,j])/(rob_array[j,j]-rob_array[0,j]) + (rob_array[j,i] - rob_array[0,i])/(rob_array[i,i]-rob_array[0,i]))
        if cur_orverlap_score > 0 :
            overlappings[i-1][j-1] = cur_orverlap_score
            overlappings[j-1][i-1] = cur_orverlap_score

overlap_array =  pd.DataFrame(overlappings, index=corruption_list, columns=corruption_list)
overlap_array.to_pickle(os.path.join("../Results/corruption_cat",'overlappings_matrix.pkl'))

heatmap = sb.heatmap(overlappings.round(2), vmin=0, vmax=1, cmap="Greens", annot=True, annot_kws={"size": 9}, xticklabels=1, yticklabels=1)
figure = plt.gcf()
figure.set_size_inches(21, 12)
plt.savefig(os.path.join("../Results/corruption_cat",'overlappings_matrix.png'),bbox_inches='tight')
