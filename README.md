# A Benchmark of Non-Overlapping Corruptions
This repository contains the code associated with the paper entitled [A Benchmark of Non-Overlapping Corruptions](https://github.com/bds-ailab/common_corruption_benchmark).
In this paper, a new benchmark of Non-Overlapping Corruptions called ImageNet-NOC is proposed. This benchmark evaluates the robustness of image classifiers towards common corruptions.
ImageNet-NOC is based on eight image transformations that have been chosen to cover a very large range of common corruptions. Here is an illustration of the ImageNet-NOC corruptions:<br/>

<img align="center" src="illustrations/benchmark_illustration.png" width="900">

## Requirements
Pytorch 1.5

Scipy 1.4

Pandas 1.0

Seaborn 0.11

## Estimate the Common Corruption Robustness of a Model with ImageNet-NOC
To get the ImageNet-NOC CE scores of the torchvision pretrained ResNet-50, launch the following command:<br/>
```
python3 get_mCE.py PATH_TO_THE_VAL_SET_FOLDER
```
With PATH_TO_THE_VAL_SET_FOLDER the path to the ImageNet validation set.<br/>
A few code lines can be changed to load your own model instead of the torchvision ResNet-50.<br/>

## Robustness Landmarks
We provide the ImageNet-NOC mCE score of the pretrained torchvision ResNet-50.<br/>
Submit a pull request if you want to display the robustness of your own model to ImageNet-NOC. Your model must have a ResNet-50 architecture to be compared to the other models displayed in the array <br/>

| Model     | Paper    | mCE   |
| :------------- | :------------- | :------------- |
| Standard ResNet-50       |        | 81     |

## Compute the Overlapping Scores between a Group of Corruptions
We provide the code used to compute the overlapping scores between several corruptions.<br/>
In the CC_Transform file, we provide the modelings of twenty-three common corruptions.<br/>
To train one model with data augmentation, for each of the modeled corruptions, launch:<br/>
```
python3 train.py PATH_TO_THE_VAL_SET_FOLDER
```

The default neural network architecture used is a ResNet-18. The results found in our [paper](https://github.com/bds-ailab/common_corruption_benchmark) are computed using this architecture and the ImageNet subset: ImageNet-100.<br/>
The weights of the trained models are saved in the "saved_models" folder.<br/>
To get the accuracies of each trained model, computed with the twenty-three ImageNet validation sets that are each corrupted with one corruption of the CC_Transform file, use the following command:<br/>
```
python3 get_accuracy.py PATH_TO_THE_VAL_SET_FOLDER
```

The values of the computed accuracies are saved in the "results" folder.<br/>
Use the computed accuracies to obtain the overlapping scores between all the modeled corruptions with:<br/>
```
python3 get_overlappings.py
```

The computed overlapping scores are saved in the 'results' folder.<br/>

## Citation
Paper under review.

If you have any question about the benchmark, do not hesitate to contact us at alfred.laugros@atos.net.<br/>
