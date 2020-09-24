# Increasing the Coverage and Balance of Benchmarks by Using Non-Overlapping Corruptions
This repository contains the code associated with the paper entitled [Increasing the Coverage and Balance of Benchmarks by Using Non-Overlapping Corruptions](https://linktothepaper)
In this paper a new benchmark called NOCS is proposed. This benchmark evaluates the robustness of image classifiers towards common corruptions.
NOCS is based on eight image transformations that have been chosen to cover a very large range of diverse common corruptions. Here is the illustration of the NOCS corruptions:


<img align="center" src="illustrations/benchmark_illustration.png" width="700">


## Requirements
Pytorch 1.5<br/>
scipy 1.4<br/>
pandas 1.0

## Test the Robustness of a Model to NOCS
To get the NOCS CE scores of a standard pretrained ResNet-50, launch the following command: <br/>
`python3 get_mCE.py PATH_TO_THE_VAL_SET_FOLDER`<br/>
With PATH_TO_THE_VAL_SET_FOLDER the path to the ImageNet validation set.<br/>
The code can be adapted to load your own model instead of a standard ResNet-50.<br/>


## Performances of Various Models to NOCS
We provide the NOCS CE scores of the pretrained torchvision ResNet-50.
Submit a pull request if you want to display the robustness of your model to NOCS. Your model should have a ResNet-50 architecture to be compared to the other models displayed in this array <br/>

| Model     | Paper    | mCE   |
| :------------- | :------------- | :------------- |
| Standard ResNet-50       |        | 81     |

## Citation

    @article{bibkey:laugros2020inoc,
      title={},
      author={},
      journal={},
      year={}
    }
