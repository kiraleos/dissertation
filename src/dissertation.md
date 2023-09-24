---
title: "Application of Machine Learning Methods on Astronomical Databases"
author: 
    - "Apostolos Kiraleos"
abstract: "Galaxy redshift is a crucial parameter in astronomy that provides information on the distance, age, and evolution of galaxies. This dissertation investigates the application of machine learning for predicting galaxy redshifts. It involves the development and training of a neural network to analyze galaxy spectra sourced from the European Space Agency's Gaia mission and showcases the practical implementation of machine learning in astronomy."
linestretch: 1.25
papersize: "a4"
indent: true
numbersections: true
geometry: "left=3cm,right=3cm,top=2cm,bottom=2cm"
fontsize: 12pt
---

\newpage

\renewcommand{\contentsname}{Table of Contents}
\tableofcontents

\newpage

\listoffigures

\newpage

# Introduction

## Object

This dissertation primarily addresses the critical issue of accurate galaxy redshift estimation. Galaxy redshifts are pivotal in modern astronomy, providing crucial insights into cosmic distances, universe evolution, and more. The topic is of significant interest and relevance within the field.

## Purpose and Objectives

The main goal of this dissertation is to advance galaxy redshift estimation by applying advanced machine learning techniques, specifically Convolutional Neural Networks (CNNs). The objective is evaluating the performance and accuracy of a trained CNN model in predicting galaxy redshifts with the aim to understand the practical applications of this model in astronomical research.

## Methodology

The methodology of this dissertation involves designing, training, and evaluating the performance of a trained Convolutional Neural Network (CNN). The CNN is trained on a dataset of galaxy spectra sourced from the European Space Agency's Gaia mission which came already preprocessed and cleaned, ready for model training. The CNN is then trained on the dataset and evaluated on a test set of galaxy spectra.

For the actual implementation of the CNN, the Python programming language was used, along with the Keras deep learning library.

## Innovation

The contribution of this work lies in advancing knowledge within the field of galaxy redshift estimation. By applying state-of-the-art machine learning techniques, we aim to provide an innovative approach to predicting galaxy redshifts, potentially improving accuracy and efficiency in astronomy.

## Structure

The subsequent chapters of this dissertation are organized as follows:

1. **Introduction** (the current chapter): Provides an overview of the dissertation's focus, objectives, methodology, and innovation.
2. **Background**: Introduce foundational concepts related to the Gaia mission, galaxy redshift and machine learning methods.
3. **Data Preparation**: Details the process of selecting and cleaning galaxy spectra data for model training.
4. **Methodology**: Elaborates on the methodologies used for CNN model training and evaluation.
5. **Results and Discussion**: Presents the findings of our experiments, assesses model performance, and discusses implications.
6. **Conclusion**: Summarizes key takeaways and outlines potential areas for future research.

# Background

## Gaia space obvservatory

The Gaia mission is a European Space Agency (ESA) space observatory that has been in operation since 2013. The primary goal of the mission is to create a three-dimensional map of our galaxy by measuring the positions, distances, and motions of over two billion stars.

At its core, Gaia is equipped with two optical telescopes accompanied by three scientific instruments, which collaborate to accurately ascertain the positions and velocities of stars. Additionally, these instruments disperse the starlight into spectra, facilitating detailed analysis.

Throughout its mission, the spacecraft executes a deliberate rotation, systematically scanning the entire celestial sphere with its two telescopes. As the detectors continuously record the positions of celestial objects, they also capture the objects' movements within the galaxy, along with any alterations therein.

During its mission, Gaia conducts approximately 14 observations each year for its designated stars. Its primary goals include accurately mapping the positions, distances, motions, and brightness variations of stars. Gaia's mission is expected to uncover various celestial objects, including exoplanets and brown dwarfs, and thoroughly study hundreds of thousands of asteroids within our Solar System. Additionally, the mission involves studying over 1 million distant quasars and conducting new assessments of Albert Einstein's General Theory of Relativity.

## Galaxy redshift

Galaxy redshift is a fundamental astronomical property that describes the relative motion of a galaxy with respect to Earth. The redshift of a galaxy is measured by analyzing the spectrum of light emitted by the galaxy, which appears to be shifted towards longer wavelengths due to the Doppler effect. Redshift is a crucial parameter in astronomy, as it provides information about the distance, velocity, and evolution of galaxies.

The redshift of a galaxy is measured in units of "$z$", which is defined as the fractional shift in the wavelength of light emitted by the galaxy. Specifically, the redshift "$z$" is defined as:

$$z = \frac{\lambda_{obsv} - \lambda_{emit}}{\lambda_{emit}}$$

where $\lambda_{obsv}$ is the observed wavelength of light from the galaxy, and $\lambda_{emit}$ is the wavelength of that same light as emitted by the galaxy. A redshift of $z=0$ corresponds to no shift in the wavelength (i.e., the observed and emitted wavelengths are the same), while a redshift of $z=1$ corresponds to a shift of 100% in the wavelength (i.e., the observed wavelength is twice as long as the emitted wavelength).

Accurate and efficient estimation of galaxy redshift is essential for a wide range of astronomical studies, including galaxy formation and evolution, large-scale structure of the universe, and dark matter distribution. However, measuring galaxy redshifts can be a challenging task due to various factors such as observational noise, instrumental effects, and variations in galaxy spectra.

## Convolutional Neural Networks

### Overview

Convolutional neural networks (CNNs) are a type of artificial neural network (ANN) that has proven to be highly effective for tasks involving image and video analysis, such as object detection, segmentation, and classification. CNNs are inspired by the structure and function of the visual cortex in animals, which contains specialized cells called neurons that are tuned to detect specific visual features. In a similar way, CNNs are designed to learn and extract meaningful visual features from raw image data.

At the core of a CNN are individual processing units called neurons, which are organized into layers. Each neuron receives input from other neurons in the previous layer, applies a mathematical operation to that input, and passes the output to the next layer. The output of the final layer of neurons is the predicted output of the network for a given input.

The key innovation of CNNs is their use of convolutional layers, which enable the network to automatically learn and extract local spatial features from raw input data. In a convolutional layer, each neuron is connected only to a small, localized region of the input data, known as the receptive field. By sharing weights across all neurons within a receptive field, the network can efficiently learn to detect local patterns and features, regardless of their location within the input image.

CNNs typically also include pooling layers, which downsample the output of the previous layer by taking the maximum or average value within small local regions. This helps to reduce the dimensionality of the input and extract higher-level features from the local features learned in the previous convolutional layer.

The final layers of a CNN are fully connected layers, which take the outputs of the previous convolutional and pooling layers and use them to make a prediction. In the case of image classification, for example, the output of the final fully connected layer might be a vector of probabilities indicating the likelihood of each possible class. In our case, instead of class probabilities, the output of the final fully connected layer will yield a single numeric value representing the predicted redshift of the observed galaxy.

### Mechanism of operation

Neural networks are trained using a process called backpropagation, which involves iteratively adjusting the weights of the network to minimize a loss function. The loss function measures the difference between the predicted output of the network and the actual output. By minimizing the loss function, the network learns to make more accurate predictions.

The weights of a neural network are adjusted using an algorithm called gradient descent, which involves computing the gradient of the loss function with respect to each weight. The gradient indicates the direction in which the loss function is decreasing the fastest, which is the direction in which the weight should be adjusted to minimize the loss function.

### Loss functions

Loss functions are mathematical functions that are used to measure the difference between the predicted output of a neural network and the actual output. They are used to guide the training process by indicating how well the network is performing. The most common loss functions used in neural networks are the mean squared error (MSE) and mean absolute error (MAE) functions.

The MSE function is defined as:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2$$

The MAE function is defined as:

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|$$

where $y_i$ is the actual output and $\hat{y_i}$ is the predicted output for the $i$th sample, and $n$ is the total number of samples.

Another loss function that is commonly used in neural networks (and the one we end up using) is the huber loss function. It is less sensitive to outliers in data than the squared error loss.

$$L_{\delta} = \frac{1}{n} \sum_{i=1}^{n} \begin{cases} \frac{1}{2}(y_i - \hat{y_i})^2 & \text{if } |y_i - \hat{y_i}| \leq \delta \\ \delta |y_i - \hat{y_i}| - \frac{1}{2} \delta^2 & \text{otherwise} \end{cases}$$

where $\delta$ is a small constant.

### Activation functions

Activation functions are mathematical functions that are applied to the output of each neuron in a neural network. They are used to introduce non-linearity into the network, which is essential for learning complex patterns and relationships in the data. The most common activation functions used in neural networks are the sigmoid, tanh, and ReLU functions. For *convolutional* neural networks, the ReLU function is typically used for all layers except the output layer, which uses a linear activation function.

The ReLU function is simply defined as:

$$ReLU(x) = max(0, x)$$

where $x$ is the input to the function. The ReLU function is a piecewise linear function that returns the input if it is positive, and zero otherwise.

### Optimizers

Optimizers are algorithms that are used to adjust the weights of a neural network during training. They are used to minimize the loss function and improve the performance of the network. The most common optimizers used in CNNs are the stochastic gradient descent (SGD), Adam, and RMSprop optimizers.

The SGD optimizer is defined as:

$$w_{t+1} = w_t - \alpha \nabla L(w_t)$$

The Adam optimizer is defined as:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(w_t)$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla L(w_t)^2$$

$$\hat{m_t} = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v_t} = \frac{v_t}{1 - \beta_2^t}$$

$$w_{t+1} = w_t - \alpha \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}$$

The RMSprop optimizer is defined as:

$$v_t = \beta v_{t-1} + (1 - \beta) \nabla L(w_t)^2$$

$$w_{t+1} = w_t - \alpha \frac{\nabla L(w_t)}{\sqrt{v_t} + \epsilon}$$

where $w_t$ is the weight at time $t$, $\alpha$ is the learning rate, $\beta_1$ and $\beta_2$ are the exponential decay rates for the first and second moments, respectively, and $\epsilon$ is a small constant to prevent division by zero.

# Data Preparation

## Source

The data used for this dissertation is a combination of the Sloan Digital Sky Survey Data Release 16 (SDSS DR16) and the Gaia Data Release 3 (DR3) datasets. Combined, they form a dataset of galaxies with known redshifts and spectra.

## Composition

The dataset consists of 520.000 galaxy spectra, each with 186 data points. Each data point is the flux at a specific wavelength, ranging from 366 to 996 nanometers. The dataset also contains the redshift of each galaxy, ranging from 0 to 0.6 $z$. which is the target variable that we aim to predict.

## Preprocessing

The only preprocessing step that was performed was to to apply a min-max normalization to the flux values, which rescales them to the range $[0, 1]$. Min-max normalization is a common practice in machine learning, particularly for neural networks. It standardizes the data, ensuring that all features are on the same scale. This not only aids in model convergence but also enhances training stability and performance. It is mathematically defined as:
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
where $x$ is the original data and $x_{min}$ and $x_{max}$ are the minimum and maximum values of $x$, respectively.

## Splitting

The dataset was split into a training set, a validation set, and a test set. The training set contained 90% of the data, while the test set contained the remaining 10%. The training set was used to train the neural network, while the test set was used to evaluate the performance of the trained model. Of the training set, 30% was used as a validation set, which was used to evaluate the model during training and tune the model's hyperparameters. Hyperparameters will be explained in more detail in the next chapter.

# Methodology

## Network architecture

The Convolutional Neural Network (CNN) architecture chosen for this task is specifically tailored to handle the spectral data obtained from the Gaia mission. Unlike traditional image-based CNNs, which analyze two-dimensional images, this architecture is designed to work with one-dimensional spectra.

**Input Layer**: The input layer accepts the normalized galaxy spectra. Each spectrum, represented as a 1D array of flux values at different wavelengths, is fed into the network. The input layer's size is determined by the number of data points in each spectrum.

**Convolutional Layers**: These layers are responsible for feature extraction. Convolutional filters slide over the input spectra, capturing local patterns and features at different scales. Given the varying nature of spectral data, multiple convolutional layers with different filter sizes are used to capture both fine-grained and broader features. This adaptability is essential since galaxy spectra can exhibit a wide range of shapes and characteristics.

**Pooling Layers**: After each convolutional layer, a pooling layer follows. Max-pooling is applied to reduce the spatial dimensionality of the feature maps, retaining the most important features. Pooling helps in managing the computational complexity of the network while preserving crucial information.

**Fully Connected (Dense) Layers**: Following the convolutional and pooling layers, fully connected layers are used for the final prediction. These layers gradually reduce the dimensionality of the features learned by previous layers, ultimately yielding a single numeric value: the predicted galaxy redshift.

**Output Layer**: The output layer consists of a single neuron with a linear activation function. This neuron's output represents the predicted redshift of the observed galaxy. During training, this prediction is compared to the actual redshift to compute the loss, which guides the network's weight adjustments.
\

Below is the Keras code for the architecture. We used 4 convolutional layers each followed by a max-pooling layer, followed by 4 fully connected layers, and 1 output layer. The number of filters in each convolutional layer was 256, 256, 128, and 64, respectively. The kernel size of each convolutional layer was 5, 5, 3, and 2, respectively. The number of neurons in each fully connected layer was 256, 256, 128, and 64, respectively. We used the ReLU activation function for all layers except the output layer, which used a linear activation function. The decision to use these specific hyperparameters was based on known common practices and extensive experimentation.

```python
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(186, 1)),
    tf.keras.layers.Conv1D(filters=256, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(),
    tf.keras.layers.Conv1D(filters=256, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(),
    tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling1D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])
```
