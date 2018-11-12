# TFE: Phase Retrieval

Light detectors, such as photographic plates or CCDs, measure only the intensity of the light that hits them. This measurement is incomplete because a light wave has not only an amplitude (related to the intensity), but also a phase, which is systematically lost in a measurement. Phase retrieval is the process of algorithmically retrieving the lost phase trough iterative algorithms (Gerchberg-Saxton, ...) or as in this project through machine learning (Convolutional neural network).

<p align="center"><img src="https://github.com/pvanberg/phase-retrieval/blob/dev/assets/architecture.png" /></p>

## Architectures

For incoming wavelength λ of 2200 micrometers and input size of 128x128 px. 

| Model | RMSE Zernike | RMSE Phase |
| --- | --- | --- |
| Custom v0.3 | 0 | 0 |
| Resnet-50 | 0 | 0 |
| InceptionV3 | 0 | 0 |
| Custom v0.2 | 0 | 0 |
| Custom v0.1 | 0 | 0 |

The root mean squared error are normalized with respect to λ.

## Algorithms

Gerchberg-Saxton algorithm is available here.

## Monitoring

GPU monitoring informations are available here.
