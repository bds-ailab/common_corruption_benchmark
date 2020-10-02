# Copyright 2020 BULL SAS All rights reserved #
# Uses the accuracy scores obtained with the 'get_accuracy.py' module, to compute the associated overlapping score
# The computed overlapping scores are saved in the results folder

import tools
import pandas as pd
import sys
import numpy as np
import seaborn
import matplotlib.pyplot as plt

# load the accuracy array
acc_array = pd.read_pickle('results/corruption_accuracy_array.pkl')
list_corruptions = list(acc_array.index)
acc_array = acc_array.to_numpy()

# Build the array that contains the computed overlapping score
overlapping_array = np.zeros([len(list_corruptions)-1,len(list_corruptions)-1])

for i in range(1,len(list_corruptions)):
    for j in range(i,len(list_corruptions)):

        #Compute the Robustness scores required to get the overlapping score
        Rij = acc_array[i,j]/acc_array[i,0]
        Rji = acc_array[j,i]/acc_array[j,0]
        Rii = acc_array[i,i]/acc_array[i,0]
        Rjj = acc_array[j,j]/acc_array[j,0]
        Rstandi = acc_array[0,i]/acc_array[0,0]
        Rstandj = acc_array[0,j]/acc_array[0,0]

        # Use the robsutness scores to get the overlapping score
        overlapping_score = 0.5*((Rij - Rstandj)/(Rjj-Rstandj) + (Rji - Rstandi)/(Rii-Rstandi))
        overlapping_score = np.maximum(overlapping_score,0)

        # Save the computed overlapping score
        overlapping_array[i-1][j-1] = overlapping_array[j-1][i-1] = overlapping_score

# Save the computed overlapping scores
overlapping_array = pd.DataFrame(overlapping_array, index=list_corruptions[1:], columns=list_corruptions[1:])
overlapping_array = overlapping_array.round(2)
overlapping_array.to_pickle('results/overlapping_array.pkl')
overlapping_array = seaborn.heatmap(overlapping_array, vmin=0, vmax=1, cmap="Greens", annot=True, annot_kws={"size": 5})
plt.savefig('results/overlapping_array.png',bbox_inches='tight')
