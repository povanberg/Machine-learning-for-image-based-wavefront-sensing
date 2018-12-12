# TFE: Phase Retrieval

Light detectors, such as photographic plates or CCDs, measure only the intensity of the light that hits them. This measurement is incomplete because a light wave has not only an amplitude (related to the intensity), but also a phase, which is systematically lost in a measurement. Phase retrieval is the process of algorithmically retrieving the lost phase trough iterative algorithms (Gerchberg-Saxton, ...) or as in this project through machine learning (Convolutional neural network).

<p align="center"><img src="https://github.com/pvanberg/phase-retrieval/blob/dev/assets/architecture.png" /></p>

## Architectures

For incoming wavelength λ of 2200 micrometers and input size of 128x128 px. 

| Model | RMSE Zernike | RMSE Phase | Showcase |
| --- | --- | --- | --- |
| Unet | - | 5.32nm |
| [Resnet-50](experiments/resnet) | 1.35 nm | 6.05 nm | [check](experiments/resnet/evaluation.ipynb)  |
| [InceptionV3](experiments/inception)  | 1.97 nm | 8.81 nm | [check](experiments/inception/evaluation.ipynb) |
| [Custom v0.3](experiments/custom_v0.3) | 2.40 nm | 10.72 nm | [check](experiments/custom_v0.3/evaluation.ipynb) |
| [Densenet-161](experiments/densenet)  | 2.83 nm | 12.67 nm | [check](experiments/densenet/evaluation.ipynb) |


## Algorithms

Gerchberg-Saxton algorithm is available [here](algorithms/Gerchberg–Saxton.ipynb).

## Monitoring

GPU monitoring informations are available [here](monitoring.ipynb).
