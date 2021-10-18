This repository contains the code associated with the paper entitled "Using Synthetic Corruptions to Measure Robustness to Natural Distribution Shifts", which is available [here](https://arxiv.org/abs/2107.12052).

# Modules Used
pytorch: 1.7.1

albumentations: 0.5.2

sklearn: 0.24.1

seaborn: 0.11.1

pandas: 1.1.5

OpenCV: 4.5.1

scipy: 1.5.4

torchvision: 0.8.2

matplotlib: 3.3.4

## Datasets Used
ImageNet-A: https://github.com/hendrycks/natural-adv-examples

ImageNet-R: https://github.com/hendrycks/imagenet-r/

ImageNet-V2: https://github.com/modestyachts/ImageNetV2

ImageNet-P: https://github.com/hendrycks/robustness

ImageNet-C: https://github.com/hendrycks/robustness

ObjectNet: https://objectnet.dev/

ImageNet-Sketch: https://www.kaggle.com/wanghaohan/imagenetsketch

## Code Structure
The code is split into 4 folders:
1) get_corruption_cat : code used to get the overlapping matrix displayed in Figure 2.
2) generate_bench : code used to generate benchmarks (implementation of Algorithm 1 and the substitution operation).
3) benchmark_correlations : code used to estimate the correlations in terms of robustness between benchmarks. It is used to get the results displayed in Table 2 and 3.
4) Results : directory that stores the output of the scripts contained in the three folders mentioned above.
Files in the root directory are shared by the scripts of these four folders.

## Replicating Results of the Paper

**I] Get the overlapping matrix**

Navigate to the get_corruption_cat directory.

To train the models required to get the overlapping scores launch:
```
python3 corruption_trainings.py '/path/to/ImageNet-100/' all_candidates
```

To compute the accuracies required to get the overlapping scores using the models trained above launch:
```
python3 get_candiate_acc.py '/path/to/ImageNet-100/' all_candidates
```

To get the overlapping matrix using the accuracies obtained with the previous command launch:
```
python3 get_overlapping_matrix.py
```

The corruption categories are then obtained with :
```
python3 get_corruption_clusters.py
```


**II] Get the synthetic corruption benchmarks**

Navigate to the generate_bench directory

Generate 1000 different benchmarks with n=6 corruption categories represented and k=2 corruptions per represented category using the following command:
```
python3 generate_n_k_bench.py 6 2 1000
```
(any n,k values can be used)

To substitute corruptions in the benchmarks generated using n=6,k=2 to get benchmarks with non-zero std, enter:
```
python3 get_non_zero_std_bench.py 6 2
```
(any n,k values can be used)

**III] Obtain the correlations in terms of robustness between benchmarks**

Navigate to the benchmark_correlations directory

In the script entitled 'get_models_accuracies.sh' replace the path of the natural and synthetic corruption benchmarks with their location in your environment.

Do the same for the path containing the checkpoints of the used trained models (models of Table 1)

Then get the accuracies of the robust models and their plain counterparts on the natural and synthetic corruption benchmarks with:
```
./get_models_accuracies.sh
```

Use the computed accuracies to get the correlations in terms of robustness between the synthetic corruption benchmarks (ImageNet-P, ImageNet-C) and natural corruption benchmarks, using the following command:
```
python3 get_existing_bench_correlations.py
```

The correlations in terms of robustness between natural corruption benchmarks and any generated benchmarks (here n=6, k=2 and std=1.5) can be obtained using:
```
python3 get_generated_nat_bench_correlations.py 6 2 1.5
```

## Citation

>@article{DBLP:journals/corr/abs-2107-12052,
>
>  author    = {Alfred Laugros and Alice Caplier and Matthieu Ospici},
>
>  title     = {Using Synthetic Corruptions to Measure Robustness to Natural Distribution Shifts},
>
>  journal   = {CoRR},
>
>  volume    = {abs/2107.12052},
>
>  year      = {2021},
>}

If you have any question, do not hesitate to contact us at alfred.laugros@atos.net.<br/>
