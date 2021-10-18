""" Get the accuracies on ImageNet-Sketch of the models defined in ../models.py
Obtained accuracies are saved in a pickle at ../Results/benchmark_correlations """

import torch
import numpy as np
import os
import sys
import pandas
from torch.utils.data import DataLoader

p = os.path.abspath('..')
sys.path.insert(1, p)
import models
import tools
import data


model_to_test = ["resnet18","resnet50","efficientnet_b0","densenet121","resnet152","resnext101_32x8d","SIN", "Augmix", "ANT", "DeepAugment","NoisyStudent", "MoPro","Cutmix","FastAutoAugment", "AT_Linf_4","RSC","AdvProp","SpatialAdv","Anti_Alias","WSL","SSL"]

batch_size = 200
num_workers = 4
album_mode = False
dataset_path = sys.argv[1]
model_path = sys.argv[2]

device = torch.device("cuda:0")
res_array = np.zeros([len(model_to_test),1])

test_set = data.get_Inet_Sketch(dataset_path, album_mode)
test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True,pin_memory=True)
test_set_size = len(test_set)
epoch_test_size = test_set_size//batch_size

for i in range(len(model_to_test)):
    test_acc = 0
    print("model to test :" + model_to_test[i])


    classifier = models.load(model_to_test[i], model_path)
    classifier.to(device)
    classifier.eval()


    with torch.no_grad() :
        for _, couple in enumerate(test_loader):
            x, l = couple
            if model_to_test[i] == "AdvProp":
                x = x/x.max()
            x, l = x.to(device), l.to(device)
            y = classifier(x)
            test_acc = tools.update_topk_acc(test_acc,y,l,epoch_test_size,1)

        res_array[i,0] = test_acc.item()*100
        test_acc = 0

res_array = pandas.DataFrame(res_array, index=model_to_test, columns=["acc"])
res_array.to_pickle(os.path.join("../Results/benchmark_correlations","inet_sk_accuracies.pkl"))
res_array.to_html(os.path.join("../Results/benchmark_correlations","inet_sk_accuracies.html"))
