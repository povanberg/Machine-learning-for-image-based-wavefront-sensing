# Machine learning for image-based wavefront sensing

Astronomical images are often degraded by the disturbance of the Earth’s atmosphere. This thesis proposes to improve image-based wavefront sensing techniques using machine learning algorithms. Deep convolutional neural networks (CNN) have thus been trained to estimate the wavefront using one or multiple intensity measurements.


<p align="center">
<img src="https://github.com/pvanberg/Machine-learning-for-image-based-wavefront-sensing/blob/master/assets/unet.png" width="600" height="300">
</p>

## Getting Started
 	
### Prerequisites

First, make sure the following python libraries are installed.

```
Aotools
Astropy
Soapy
Scipy
Pytorch
Visdom
```
### Examples

The dataset generation can be run using. The dataset size and other parameters can be set in the same file.

```
python src/generation/generator.py
```

Some notebooks to highlights the networks and the dataset.

- [Overview of the dataset](examples/data.ipynb)
- [Network Training](examples/training.ipynb)
- [Network evaluation](examples/evaluation.ipynb)

Finally some classical algorithms (Gerchberg–Saxton) can be directly tested on the dataset.

```
python src/algorithms/Gerchberg–Saxton.py
```
