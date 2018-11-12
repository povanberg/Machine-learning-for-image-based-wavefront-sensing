# TFE: Phase Retrieval

Light detectors, such as photographic plates or CCDs, measure only the intensity of the light that hits them. This measurement is incomplete because a light wave has not only an amplitude (related to the intensity), but also a phase, which is systematically lost in a measurement. Phase retrieval is the process of algorithmically retrieving the lost phase trough iterative algorithms (Gerchberg-Saxton, ...) or as in this project through machine learning (Convolutional neural network).

<p align="center"><img src="https://github.com/pvanberg/phase-retrieval/blob/dev/assets/architecture.png" /></p>

## Architectures

For incoming wavelength λ of 2200 micrometers and input size of 128x128 px. 

| Model | RMSE Zernike | RMSE Phase |
| --- | --- | --- |
| [Custom v0.3](experiments/custom_v0.3) | 7.91 nm | 31.33 nm |
| [InceptionV3](experiments/inception)  | 21.92 nm | 90.55 nm |
| [Resnet-50](experiments/resnet) | 0 | 0 |
| Custom v0.2 | 0 | 0 |
| Custom v0.1 | 0 | 0 |

## Algorithms

Gerchberg-Saxton algorithm is available [here](algorithms/Gerchberg–Saxton.ipynb).

## Monitoring

GPU monitoring informations are available here.
