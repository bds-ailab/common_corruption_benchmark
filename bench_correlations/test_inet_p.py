""" Using the ImageNet-P benchmark, get the mean Flip Rate of the models defined in ../models.py
Obtained scores are saved in a pickle at ../Results/benchmark_correlations
This code has been adapted from https://github.com/hendrycks/robustness """

import numpy as np
import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.transforms.functional as trn_F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from scipy.stats import rankdata
import pandas

p = os.path.abspath('..')
sys.path.insert(1, p)
import models
dataset_path = sys.argv[1]
model_path = sys.argv[2]

if __package__ is None:
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from video_loader import VideoFolder

batch_size = 4

difficulty = 1
device = torch.device("cuda:0")
torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
cudnn.benchmark = True  # fire on all cylinders

# print('Model Loaded\n')

# /////////////// Data Loader ///////////////
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]



list_pert = ['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur', 'brightness',
'translate', 'rotate', 'tilt', 'scale', 'snow']

model_to_test = ["resnet18","resnet50","efficientnet_b0","densenet121","resnet152","resnext101_32x8d",
"SIN", "Augmix", "ANT", "DeepAugment","NoisyStudent", "MoPro","Cutmix","FastAutoAugment", "AT_Linf_4",
"RSC","AdvProp","SpatialAdv","Anti_Alias","WSL","SSL"]

res_array = np.zeros([len(model_to_test),len(list_pert)])

for x in range(len(model_to_test)):
    print(model_to_test[x])
    net = models.load(model_to_test[x], model_path)
    net = torch.nn.DataParallel(net, device_ids=[0])
    net.to(device)
    net.eval()

    for y in range(len(list_pert)):

        if difficulty > 1 and 'noise' in list_pert[y]:
            loader = torch.utils.data.DataLoader(
                VideoFolder(root= dataset_path + '/' + sys.argv[1] +
                                 list_pert[y] + '_' + str(difficulty),
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])),
                batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)
        else:
            loader = torch.utils.data.DataLoader(
                VideoFolder(root=dataset_path + '/' + list_pert[y],
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])),
                batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)

        # print('Data Loaded\n')


        # /////////////// Stability Measurements ///////////////

        identity = np.asarray(range(1, 1001))
        cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (999 - 5)))
        recip = 1./identity

        # def top5_dist(sigma):
        #     result = 0
        #     for i in range(1,6):
        #         for j in range(min(sigma[i-1], i) + 1, max(sigma[i-1], i) + 1):
        #             if 1 <= j - 1 <= 5:
        #                 result += 1
        #     return result

        def dist(sigma, mode='top5'):
            if mode == 'top5':
                return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma-1][:5]))
            elif mode == 'zipf':
                return np.sum(np.abs(recip - recip[sigma-1])*recip)


        def ranking_dist(ranks, noise_perturbation=True if 'noise' in list_pert[y] else False, mode='top5'):
            result = 0
            step_size = 1 if noise_perturbation else difficulty

            for vid_ranks in ranks:
                result_for_vid = []

                for i in range(step_size):
                    perm1 = vid_ranks[i]
                    perm1_inv = np.argsort(perm1)

                    for rank in vid_ranks[i::step_size][1:]:
                        perm2 = rank
                        result_for_vid.append(dist(perm2[perm1_inv], mode))
                        if not noise_perturbation:
                            perm1 = perm2
                            perm1_inv = np.argsort(perm1)

                result += np.mean(result_for_vid) / len(ranks)

            return result


        def flip_prob(predictions, noise_perturbation=True if 'noise' in list_pert[y] else False):
            result = 0
            step_size = 1 if noise_perturbation else difficulty

            for vid_preds in predictions:
                result_for_vid = []

                for i in range(step_size):
                    prev_pred = vid_preds[i]

                    for pred in vid_preds[i::step_size][1:]:
                        result_for_vid.append(int(prev_pred != pred))
                        if not noise_perturbation: prev_pred = pred

                result += np.mean(result_for_vid) / len(predictions)

            return result


        # /////////////// Get Results ///////////////

        predictions, ranks = [], []
        with torch.no_grad():

            for data, target in loader:
                num_vids = data.size(0)
                data = data.view(-1,3,224,224).cuda()

                output = net(data/data.max())

                for vid in output.view(num_vids, -1, 1000):
                    predictions.append(vid.argmax(1).to('cpu').numpy())
                    ranks.append([np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])

        ranks = np.asarray(ranks)

        # print('Computing Metrics\n')

        # print('Flipping Prob\t{:.5f}'.format(flip_prob(predictions)))
        # print('Top5 Distance\t{:.5f}'.format(ranking_dist(ranks, mode='top5')))
        # print('Zipf Distance\t{:.5f}'.format(ranking_dist(ranks, mode='zipf')))
        cur_flip_prob = flip_prob(predictions)
        print(cur_flip_prob)
        res_array[x,y] = float(cur_flip_prob)

res_array = res_array.mean(1)
res_array = pandas.DataFrame(res_array, index=model_to_test, columns=["mFR"])
res_array.to_pickle(os.path.join("../Results/benchmark_correlations","inet_p_mfr.pkl"))
res_array.to_html(os.path.join("../Results/benchmark_correlations","inet_p_mfr.html"))
