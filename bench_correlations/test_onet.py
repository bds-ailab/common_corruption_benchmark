""" Get the accuracies on ObjectNet of the models defined in ../models.py
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

device = torch.device("cuda:0")
batch_size = 200
num_workers = 4
res_array = np.zeros([len(model_to_test),1])
album_mode = False
dataset_path = sys.argv[1]
model_path = sys.argv[2]

inet_to_onet = {"409": 1, "530": 1, "414": 2, "954": 4, "419": 5, "790": 8, "434": 9, "440": 13, "703": 16, "671": 17, "444": 17, "446": 20, "455": 29, "930": 35, "462": 38, "463": 39, "499": 40, "473": 45, "470": 46, "487": 48, "423": 52, "559": 52, "765": 52, "588": 57, "550": 64, "507": 67, "673": 68, "846": 75, "533": 78, "539": 81, "630": 86, "740": 88, "968": 89, "729": 92, "549": 98, "545": 102, "567": 109, "578": 83, "589": 112, "587": 115, "560": 120, "518": 120, "606": 124, "608": 128, "508": 131, "618": 132, "619": 133, "620": 134, "951": 138, "623": 139, "626": 142, "629": 143, "644": 149, "647": 150, "651": 151, "659": 153, "664": 154, "504": 157, "677": 159, "679": 164, "950": 171, "695": 173, "696": 175, "700": 179, "418": 182, "749": 182, "563": 182, "720": 188, "721": 190, "725": 191, "728": 193, "923": 196, "731": 199, "737": 200, "811": 201, "742": 205, "761": 210, "769": 216, "770": 217, "772": 218, "773": 219, "774": 220, "783": 223, "792": 229, "601": 231, "655": 231, "689": 231, "797": 232, "804": 235, "806": 236, "809": 237, "813": 238, "632": 239, "732": 248, "759": 248, "828": 250, "850": 251, "834": 253, "837": 255, "841": 256, "842": 257, "610": 258, "851": 259, "849": 268, "752": 269, "457": 273, "906": 273, "859": 275, "999": 276, "412": 284, "868": 286, "879": 289, "882": 292, "883": 293, "893": 297, "531": 298, "898": 299, "543": 302, "778": 303, "479": 304, "694": 304, "902": 306, "907": 307, "658": 309, "909": 310}
inet_to_onet = {int(k): v for k, v in inet_to_onet.items()}

for id in range(1000):
    if id not in list(inet_to_onet.keys()) :
        inet_to_onet[id] = 1001


test_acc = 0
for i in range(len(model_to_test)):
    print("Start evaluating the robustness of the model " + model_to_test[i])

    classifier = models.load(model_to_test[i], model_path)
    classifier.to(device)
    classifier.eval()


    test_set = data.get_ObjectNet(dataset_path, album_mode)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True,pin_memory=True)
    test_set_size = len(test_set)
    epoch_size = test_set_size//batch_size

    with torch.no_grad() :
        for _, couple in enumerate(test_loader):
            x, l = couple
            if model_to_test[i] == "AdvProp":
                x = x/x.max()
            x, l = x.to(device), l.to(device)
            y = classifier(x)
            for h in range(y.shape[0]):
                y[h] = inet_to_onet[torch.argmax(y,1)[h].item()]
            y = y[:,0]
            test_acc = test_acc + torch.eq(y,l).float().mean()/epoch_size

        res_array[i,0] = test_acc.item()*100
    # save the computed accuracy in the result array
    test_acc = 0

res_array = pandas.DataFrame(res_array, index=model_to_test, columns=["acc"])
res_array.to_pickle(os.path.join("../Results/benchmark_correlations","onet_accuracies.pkl"))
res_array.to_html(os.path.join("../Results/benchmark_correlations","onet_accuracies.html"))
