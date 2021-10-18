""" Train the models required to compute the overlapping scores.
Namely, train one model with data augmentation for each candidate corruption.
Trained models are saved at ../Results/trained_models
"""
import torch
import time
import os
import sys
import torchvision
from torch.utils.data import DataLoader

p = os.path.abspath('..')
sys.path.insert(1, p)
import data
import tools
import candidate_corruptions

dataset_path = sys.argv[1]
list_corruptions = sys.argv[2:]
if list_corruptions == ["all_candidates"]:
    list_corruptions = ["clean"] + list(candidate_corruptions.dict_corruptions.keys())

learning_rate = 1e-1
batch_size = 256
num_workers = 4
decay_rate = 0.1
device = torch.device("cuda:0")

cross_entrop = torch.nn.CrossEntropyLoss()
clean_set = data.get_Inet100(dataset_path, "train", album_mode=True)
epoch_size = len(clean_set)*2//batch_size

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

for k in range(len(list_corruptions)):
    print("Start training a model with data augmentation on " + list_corruptions[k])

    if not os.path.exists(os.path.join("../Results/trained_models",list_corruptions[k])):
        os.makedirs(os.path.join("../Results/trained_models",list_corruptions[k]))

    augmented_set = data.get_Inet100(dataset_path, "train", corruptions=[list_corruptions[k]], album_mode=True)
    train_set = torch.utils.data.ConcatDataset([augmented_set,clean_set])
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True,pin_memory=True)

    classifier = torchvision.models.resnet18(num_classes=100, pretrained=False)
    classifier.apply(init_weights)
    classifier.to(device)

    cur_learning_rate = learning_rate
    optim = torch.optim.SGD(classifier.parameters(), lr=cur_learning_rate, weight_decay=1e-4, momentum=0.9)

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
            torch.save(classifier.cpu().state_dict(),os.path.join("../Results/trained_models",list_corruptions[k],"checkpoint"))
            num_epoch = 0
            checkpoint_acc = 0
            break
