---
title: "Application of Machine Learning Methods on Astronomical Databases"
subtitle: "Predicting Galaxy Redshifts Using Convolutional Neural Networks: A Comparative Study with the Gaia Unresolved Galaxy Classifier"
author: 
    - "Apostolos Kiraleos"
abstract: "Galaxy redshift is a crucial parameter in astronomy that provides information on the distance, age, and evolution of galaxies. In this thesis, we explore the use of convolutional neural networks (CNNs)  for predicting galaxy redshifts from spectra, using data from the Gaia mission. We compare the performance of our CNN model with that of the Gaia Unresolved Galaxy Classifier (UGC), which uses support vector machines (SVMs) for classification."
csl: "./acm.csl"
linestretch: 1.25
papersize: "a4"
indent: true
geometry: "left=3cm,right=3cm,top=3cm,bottom=3cm"
fontsize: 12pt
---

## 0 Table of Contents

- [0 Table of Contents](#0-table-of-contents)
- [1 Introduction](#1-introduction)
  - [1.1 Gaia obvservatory](#11-gaia-obvservatory)
  - [1.2 Galaxy redshift](#12-galaxy-redshift)
  - [1.3 Convolutional Neural Networks](#13-convolutional-neural-networks)
- [2 Our problem](#2-our-problem)

## 1 Introduction

### 1.1 Gaia obvservatory

The Gaia mission is a European Space Agency (ESA) space observatory that has been in operation since 2013. The primary goal of the mission is to create a three-dimensional map of our galaxy, the Milky Way, by measuring the positions, distances, and motions of over a billion stars.

The Gaia spacecraft is equipped with two telescopes that capture images of stars and other celestial objects, as well as a complex set of instruments that measure the brightness, color, and spectrum of these objects. The mission has already produced a wealth of data that has enabled significant advances in our understanding of the structure and evolution of the Milky Way.

In addition to its primary mission objectives, Gaia data is also being used for a wide range of scientific studies in fields such as exoplanet detection, galaxy evolution, and cosmology. One of the key applications of Gaia data is in the field of galaxy redshift estimation, which is the focus of this thesis.

### 1.2 Galaxy redshift

Galaxy redshift is a fundamental astronomical property that describes the relative motion of a galaxy with respect to Earth. The redshift of a galaxy is measured by analyzing the spectrum of light emitted by the galaxy, which appears to be shifted towards longer wavelengths due to the Doppler effect. Redshift is a crucial parameter in astronomy, as it provides information about the distance, velocity, and evolution of galaxies.

The redshift of a galaxy is measured in units of "z", which is defined as the fractional shift in the wavelength of light emitted by the galaxy. Specifically, the redshift "z" is defined as:

```python
z = (lambda_observed - lambda_emitted) / lambda_emitted
```

where `lambda_observed` is the observed wavelength of light from the galaxy, and `lambda_emitted` is the wavelength of that same light as emitted by the galaxy. A redshift of `z=0` corresponds to no shift in the wavelength (i.e., the observed and emitted wavelengths are the same), while a redshift of `z=1` corresponds to a shift of 100% in the wavelength (i.e., the observed wavelength is twice as long as the emitted wavelength).

Accurate and efficient estimation of galaxy redshift is therefore essential for a wide range of astronomical studies, including galaxy formation and evolution, large-scale structure of the universe, and dark matter distribution. However, measuring galaxy redshifts can be a challenging task due to various factors such as observational noise, instrumental effects, and variations in galaxy spectra.

In recent years, machine learning techniques have emerged as powerful tools for galaxy redshift estimation, leveraging large datasets and complex algorithms to improve the accuracy and efficiency of the estimation process. In this thesis, we focus on the application of convolutional neural networks (CNNs) to galaxy redshift estimation, and compare their performance with the Gaia Unresolved Galaxy Classifier (UGC).

### 1.3 Convolutional Neural Networks

Convolutional neural networks (CNNs) are a type of artificial neural network (ANN) that has proven to be highly effective for tasks involving image and video analysis, such as object detection, segmentation, and classification. CNNs are inspired by the structure and function of the visual cortex in animals, which contains specialized cells called neurons that are tuned to detect specific visual features. In a similar way, CNNs are designed to learn and extract meaningful visual features from raw image data.

At the core of a CNN are individual processing units called neurons, which are organized into layers. Each neuron receives input from other neurons in the previous layer, applies a mathematical operation to that input, and passes the output to the next layer. The output of the final layer of neurons is the predicted output of the network for a given input.

The key innovation of CNNs is their use of convolutional layers, which enable the network to automatically learn and extract local spatial features from raw input data. In a convolutional layer, each neuron is connected only to a small, localized region of the input data, known as the receptive field. By sharing weights across all neurons within a receptive field, the network can efficiently learn to detect local patterns and features, regardless of their location within the input image.

CNNs typically also include pooling layers, which downsample the output of the previous layer by taking the maximum or average value within small local regions. This helps to reduce the dimensionality of the input and extract higher-level features from the local features learned in the previous convolutional layer.

The final layers of a CNN are typically fully connected layers, which take the outputs of the previous convolutional and pooling layers and use them to make a prediction. In the case of image classification, for example, the output of the final fully connected layer might be a vector of probabilities indicating the likelihood of each possible class.

Overall, CNNs are a powerful and flexible tool for image analysis tasks, and have achieved state-of-the-art performance on many benchmark datasets. In this thesis, we explore the application of CNNs to galaxy redshift estimation, and compare their performance to traditional methods like the UGC classifier. We also investigate the use of transfer learning, data augmentation, and other techniques to improve the performance of CNNs on this task.

## 2 Our problem
