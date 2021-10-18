""" Compute the accuracies required to compute the overlapping scores.
Namely, for each model m trained with data augmentation on one candidate corruption (trained with corruption_trainings.py),  get the accuracy of m for each candidate corruption (using the ImageNet validation set corrupted with the considered corruption)
Computed accuracies are saved in a pickle file at ../Results/corruption_cat

Required: one model must have been trained for each considered candidate corruption using corruption_trainings.py
"""
import torch
import numpy as np
import os
import sys
import torchvision
import pandas
import sys
from torch.utils.data import DataLoader

p = os.path.abspath('..')
sys.path.insert(1, p)
import data
import tools
import candidate_corruptions


dataset_path = sys.argv[1]
list_models = sys.argv[2:]
if list_models == ["all_candidates"]:
    list_models = ["clean"] + list(candidate_corruptions.dict_corruptions.keys())
list_corruptions = list(list_models)


batch_size = 100
num_workers = 4

device = torch.device("cuda:0")
load_path = "../Results/trained_models"

res_array = np.zeros([len(list_models),len(list_corruptions)])

test_acc = 0

for i in range(len(list_models)):
    print("Get test accuracies of {} model".format(list_models[i]))
    classifier = torchvision.models.resnet18(num_classes=100, pretrained=False)
    try:
        classifier.load_state_dict(torch.load(os.path.join(load_path,list_models[i],"checkpoint"), map_location=device))
        classifier = torch.nn.DataParallel(classifier, device_ids=[0])
    except:
        classifier = torch.nn.DataParallel(classifier, device_ids=[0])
        classifier.load_state_dict(torch.load(os.path.join(load_path,list_models[i],"checkpoint"), map_location=device))
    classifier.to(device)
    classifier.eval()

    for k in range(len(list_corruptions)):

        test_set = data.get_Inet100(dataset_path, "test", corruptions=[list_corruptions[k]], album_mode=True)
        test_loader =  DataLoader(test_set, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True,pin_memory=True)
        test_set_size = len(test_set)
        epoch_test_size = test_set_size//batch_size

        with torch.no_grad() :
            for _, couple in enumerate(test_loader):
                x, l = couple
                x, l = x.to(device), l.to(device)
                y = classifier(x)
                test_acc = tools.update_topk_acc(test_acc,y,l,epoch_test_size,1)

        res_array[i,k] = test_acc.item()
        test_acc = 0

res_array = pandas.DataFrame(res_array, index=list_models, columns=list_corruptions)
res_array.to_pickle(os.path.join("../Results/corruption_cat","candidate_model_acc" + '.pkl'))
res_array.to_html(os.path.join("../Results/corruption_cat","candidate_model_acc" + '.html'))
