# Copyright 2020 BULL SAS All rights reserved #
# Use the corruptions of the benchmark to get the mCE score of a model #

import torch
import numpy as np
import os
import torchvision
import pandas
import sys

import tools
import CC_Transform
from settings import corruption_amount
from settings import ref_error

# Define the corruptions used for the test. Other corruptions from the CC_Transform file can be used
list_corruption_bench = ["blur","rotation","brightness","rhombus","quantization","obstruction","hue"]
list_corruption_bench = ["none"] + list_corruption_bench

# Parameters that can be adapted to your experiment
image_height, image_width = 224,224
batch_size = 50
num_workers = 16
device = torch.device("cuda:0".format(0))

# Build and load the ResNet-50 model. Can be adapted to the model used
classifier = torchvision.models.resnet50(pretrained=True)
classifier.to(device)
classifier.eval()

print("Start Testing")
dataset_path = sys.argv[1]
test_error, mCE = 0, 0
res_array = np.zeros([1,len(list_corruption_bench)+1])

# Eval CE for every corruption of the bench
for k in range(len(list_corruption_bench)):

    # Prepare the dataloader for the considered corruption. The validation set can be replaced with your own test set
    # The used corruption functions require the input images to be float tensors with pixel values in the range [0-1] and color channels in the RGB format
    pre_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),torchvision.transforms.CenterCrop(224),torchvision.transforms.ToTensor(),CC_Transform.CC_Transform(corruption_amount,list_corruption_bench[k]), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    test_set =  torchvision.datasets.ImageFolder(dataset_path, transform=pre_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    epoch_test_size = len(test_set)//batch_size

    # Get the test error of the model for the considered corruption
    print("Testing the model towards: {}".format(list_corruption_bench[k]))
    with torch.no_grad() :
        for _, couple in enumerate(test_loader):
            x, l = couple
            x, l = x.to(device), l.to(device)
            y = classifier(x)
            test_error = tools.update_topk_acc(test_error,y,l,epoch_test_size,1)
        test_error = 1-test_error

    # Compute the current CE score
    if list_corruption_bench[k] != "none":
        res_array[0,k] = (test_error.item()-res_array[0,0])/(ref_error[list_corruption_bench[k]]-ref_error["none"])*100
        mCE = mCE+res_array[0,k]
    else :
        res_array[0,0] = test_error.item()
    test_error = 0

# Compute and save the mCE score
res_array[0,k+1] = (mCE/(len(list_corruption_bench)-1))

# Save the resutls into an html file
res_array = pandas.DataFrame(res_array, index=["tested_model"], columns=list_corruption_bench + ["mCE"])
res_array.to_html('CE_array.html')
print("Results have been saved in an html file")
