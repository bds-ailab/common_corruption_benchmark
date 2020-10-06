# Copyright 2020 BULL SAS All rights reserved #
# Load the models saved in the "saved_models" folder, and compute their accuracy on a corrupted ImageNet validation set for each corruption in the CC_Transform file.

import torch
import torch.optim as optim
import numpy as np
import time
import os
import sys
import torchvision
import pandas
import scipy.misc
import sys
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

import tools
import CC_Transform
from settings import corruption_amount

validation_set_path = sys.argv[1]

corruption_list = ["none","quantization","gaussian","salt_pepper","blur","thumbnail_resize","pixelate","artifacts","vertical_artifacts","rhombus","rain","circle","obstruction","border","translation","shear","elastic","rotation","backlight","brightness","contrast","color_disortion","gray_scale","hue"]
list_models = list(corruption_list)

device = torch.device("cuda:0")
batch_size = 100
num_workers = 16

# build the array that contains the computed accuracies
res_array = np.zeros([len(list_models),len(corruption_list)])

test_acc, acc_model = 0, 0
for i in range(len(list_models)):
    print("Start evaluating the robustness of the model " + list_models[i])

    # Load the model trained with a data augmentation on the corruption called corruption_list[i]
    classifier = torchvision.models.resnet18(num_classes=100)
    classifier.load_state_dict(torch.load(os.path.join("saved_models",list_models[i]), map_location=device))
    classifier.to(device)
    classifier.eval()

    for k in range(len(corruption_list)):
        # Get the corrupted image loader
        pre_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),CC_Transform.CC_Transform(corruption_amount,corruption_list[k])])
        test_set = torchvision.datasets.ImageFolder(validation_set_path,transform=pre_transform)
        test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True,pin_memory=True)
        test_set_size = len(test_set)
        epoch_size = test_set_size//batch_size

        # Get the corruption accuracy of the currently considered model, for the considered corruption in corruption_list
        with torch.no_grad() :
            for _, couple in enumerate(test_loader):
                x, l = couple
                x, l = x.to(device), l.to(device)
                y = classifier(x)
                test_acc = tools.update_topk_acc(test_acc,y,l,epoch_size,1)

        # save the computed accuracy in the result array
        res_array[i,k] = test_acc.item()
        test_acc = 0

# Save the results array in a pandas DataFrame
res_array = pandas.DataFrame(res_array, index=list_models, columns=corruption_list)
res_array.to_pickle('results/corruption_accuracy_array.pkl')
res_array.to_html('results/corruption_accuracy_array.html')
print("Results have been saved in the \'results\' folder")
