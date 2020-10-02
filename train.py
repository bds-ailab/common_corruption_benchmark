# Copyright 2020 BULL SAS All rights reserved #
# train several models, one for each corruption modeled in the CC_Transform file.

import torch
import time
import os
import sys
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

import tools
import CC_Transform
from settings import corruption_amount

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)
    if type(m) == torch.nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

training_set_path = sys.argv[1]

corruption_list = ["none","quantization","gaussian","salt_pepper","blur","thumbnail_resize","pixelate","artifacts","vertical_artifacts","rhombus","rain","circle","obstruction","border","translation","shear","elastic","rotation","backlight","brightness","contrast","color_disortion","gray_scale","hue"]
device = torch.device("cuda:0")
model_save_path = "saved_models"

# The training hyperparameters
learning_rate = 1e-1
batch_size = 256
num_workers = 16
decay_rate = 0.1
nb_decay = 3

cross_entrop = torch.nn.CrossEntropyLoss()

# Get the clean image dataloader
clean_pre_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
clean_set = torchvision.datasets.ImageFolder(training_set_path, transform=clean_pre_transform)

for corruption in corruption_list:
    #Get the corrupted image dataloader
    augmented_pre_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.RandomHorizontalFlip(), transforms.ToTensor(), CC_Transform.CC_Transform(corruption_amount,corruption)])
    augmented_set = torchvision.datasets.ImageFolder(training_set_path,transform=augmented_pre_transform)

    # Combine the two dataloaders
    train_set = torch.utils.data.ConcatDataset([augmented_set,clean_set])
    epoch_size = len(clean_set)*2//batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True,pin_memory=True)

    # Build the model to train
    classifier = torchvision.models.resnet18(num_classes=100)
    classifier.apply(init_weights)
    classifier.to(device)

    cur_learning_rate = learning_rate
    optim = torch.optim.SGD(classifier.parameters(), lr=cur_learning_rate, weight_decay=1e-4, momentum=0.9)

    print("Start training with a data augmentation on " + corruption)
    num_epoch = 0
    while(1):
        start_time = time.time()
        num_epoch = num_epoch + 1
        train_acc = 0

        classifier.train()
        for i, couple in enumerate(train_loader):
            optim.zero_grad()
            x, l = couple
            x, l = x.to(device), l.to(device)
            y = classifier(x)
            loss = cross_entrop(y, l)
            loss.backward()
            optim.step()
            with torch.no_grad() :
                train_acc = tools.update_topk_acc(train_acc,y,l,epoch_size,1)

        epoch_time = time.time() - start_time
        print("epoch : {} | epoch_duration : {} | train_acc = {}".format(num_epoch,epoch_time,train_acc.item()))
        if num_epoch==20 or num_epoch==30 :
            print("learning rate decayed")
            cur_learning_rate = cur_learning_rate*decay_rate
            optim = torch.optim.SGD(classifier.parameters(), lr=cur_learning_rate, weight_decay=1e-4, momentum=0.9)
        if num_epoch==40 :
            print("End of training, the model is saved")
            torch.save(classifier.cpu().state_dict(),os.path.join(model_save_path,corruption))
            break
