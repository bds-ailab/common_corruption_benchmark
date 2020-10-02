# Increasing the Coverage and Balance of Robustness Benchmarks by Using Non-Overlapping Corruptions
This repository contains the code associated with the paper entitled [Increasing the Coverage and Balance of Robustness Benchmarks by Using Non-Overlapping Corruptions](https://linktothepaper)
In this paper, a new benchmark of Non-Overlapping Corruptions called ImageNet-NOC is proposed. This benchmark evaluates the robustness of image classifiers towards common corruptions.
ImageNet-NOC is based on eight image transformations that have been chosen to cover a very large range of common corruptions. Here is an illustration of the ImageNet-NOC corruptions:
<img align="center" src="illustrations/benchmark_illustration.png" width="900">

## Requirements
Pytorch 1.5
Scipy 1.4
Pandas 1.0
Seaborn 0.11

## Test the Robustness of a Model to ImageNet-NOC
To get the ImageNet-NOC CE scores of the torchvision pretrained ResNet-50, launch the following command:
`python3 get_mCE.py PATH_TO_THE_VAL_SET_FOLDER`
With PATH_TO_THE_VAL_SET_FOLDER the path to the ImageNet validation set.
The code can be adapted to load your own model instead of the torchvision ResNet-50.

## Robustness Landmarks
We provide the ImageNet-NOC CE scores of the pretrained torchvision ResNet-50.<br/>
Submit a pull request if you want to display the robustness of your own model to ImageNet-NOC. Your model should have a ResNet-50 architecture to be compared to the other models displayed in this array <br/>

| Model     | Paper    | mCE   |
| :------------- | :------------- | :------------- |
| Standard ResNet-50       |        | 81     |

## Compute the Overlapping Scores between some Corruptions
We provide the code used to compute the overlapping scores between several corruptions.<br/>
In the CC_Transform file, we provide the modelings of twenty-three common corruptions.<br/>
To train twenty-three models, each trained using a data augmentation with one of these modeled common corruption, launch:<br/>
`python3 train.py PATH_TO_THE_VAL_SET_FOLDER`<br/>
The default neural network architecture used is a ResNet-18. The results found in our [paper](https://linktothepaper) are computed using this architecture and the ImageNet subset: ImageNet-100.<br/>
The weights of the trained models are saved in the "saved_models" folder.<br/>
To obtain the accuracy of each trained model, on the twenty-three ImageNet validation sets that are each corrupted with one corruption of CC_Transform, use the following command:<br/>
`python3 get_accuracy.py PATH_TO_THE_VAL_SET_FOLDER`<br/>
The values of the computed accuracies are saved in the "results" folder.<br/>
Use the computed accuracies to obtain the overlapping scores between all the modeled corruptions with:<br/>
`python3 get_overlappings.py`<br/>
The computed overlapping scores are saved in the 'results' folder<br/>

## Citation
Paper under review.
