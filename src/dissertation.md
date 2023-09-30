---
title: "Application of Machine Learning Methods on Astronomical Databases"
author: 
    - "Apostolos Kiraleos"
abstract: "Galaxy redshift is a crucial parameter in astronomy that provides information on the distance, age, and evolution of galaxies. This dissertation investigates the application of machine learning for predicting galaxy redshifts. It involves the development and training of a neural network to analyze galaxy spectra sourced from the European Space Agency's Gaia mission and showcases the practical implementation of machine learning in astronomy."
linestretch: 1.25
papersize: "a4"
indent: true
numbersections: true
geometry: "left=3cm,right=3cm,top=3cm,bottom=3cm"
fontsize: 12pt
# mainfont: "Georgia"
---

\newpage

\renewcommand{\contentsname}{Table of Contents}
\tableofcontents

\newpage

\listoffigures

\newpage

# Introduction

## Object, Purpose, and Objectives

This dissertation primarily addresses the critical issue of accurate galaxy redshift estimation. Galaxy redshifts are pivotal in modern astronomy, providing crucial insights into cosmic distances, universe evolution, and more. The topic is of significant interest and relevance within the field.

The main goal of this dissertation is to advance galaxy redshift estimation by applying advanced machine learning techniques, specifically Convolutional Neural Networks (CNNs). The objective is evaluating the performance and accuracy of a trained CNN model in predicting galaxy redshifts with the aim to understand the practical applications of this model in astronomical research.

The contribution of this work lies in advancing knowledge within the field of galaxy redshift estimation. By applying state-of-the-art machine learning techniques, we aim to provide an innovative approach to predicting galaxy redshifts, potentially improving accuracy and efficiency in astronomy.

## Methodology

The methodology of this dissertation involves designing, training, and evaluating the performance of a trained Convolutional Neural Network (CNN). The CNN is trained on a dataset of galaxy spectra sourced from the European Space Agency's Gaia mission which came already preprocessed and cleaned, ready for model training. The CNN is then trained on the dataset and evaluated on a test set of galaxy spectra.

For the actual implementation of the CNN, the Python programming language was used, along with the Keras deep learning library.

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

Gaia is equipped with two optical telescopes accompanied by three scientific instruments, which collaborate to accurately ascertain the positions and velocities of stars. Additionally, these instruments disperse the starlight into spectra, facilitating detailed analysis.

Throughout its mission, the spacecraft executes a deliberate rotation, systematically scanning the entire celestial sphere with its two telescopes. As the detectors continuously record the positions of celestial objects, they also capture the objects' movements within the galaxy, along with any alterations therein.

During its mission, Gaia conducts approximately 14 observations each year for its designated stars. Its primary goals include accurately mapping the positions, distances, motions, and brightness variations of stars. Gaia's mission is expected to uncover various celestial objects, including exoplanets and brown dwarfs, and thoroughly study hundreds of thousands of asteroids within our Solar System. Additionally, the mission involves studying over 1 million distant quasars and conducting new assessments of Albert Einstein's General Theory of Relativity.[^gaia_overview]

## Galaxy redshift

Galaxy redshift is a fundamental astronomical property that describes the relative motion of a galaxy with respect to Earth. The redshift of a galaxy is measured by analyzing the spectrum of light emitted by the galaxy, which appears to be shifted towards longer wavelengths due to the Doppler effect. Redshift is a crucial parameter in astronomy, as it provides information about the distance, velocity, and evolution of galaxies.

The redshift of a galaxy is measured in units of "$z$", which is defined as the fractional shift in the wavelength of light emitted by the galaxy. Specifically, the redshift "$z$" is defined as:

$$z = \frac{\lambda_{obsv} - \lambda_{emit}}{\lambda_{emit}}$$

where $\lambda_{obsv}$ is the observed wavelength of light from the galaxy, and $\lambda_{emit}$ is the wavelength of that same light as emitted by the galaxy. A redshift of $z=0$ corresponds to no shift in the wavelength (i.e., the observed and emitted wavelengths are the same), while a redshift of $z=1$ corresponds to a shift of 100% in the wavelength (i.e., the observed wavelength is twice as long as the emitted wavelength).

Accurate and efficient estimation of galaxy redshift is essential for a wide range of astronomical studies, including galaxy formation and evolution, large-scale structure of the universe, and dark matter distribution. However, measuring galaxy redshifts can be a challenging task due to various factors such as observational noise, instrumental effects, and variations in galaxy spectra.[^redshift]

## Artificial Intelligence

Artificial Intelligence, commonly abbreviated as AI, signifies the result of extensive research and development spanning several decades, aimed at equipping machines with intelligence and decision-making abilities resembling those of humans.

One of the defining features of AI is its adaptability—machines equipped with AI algorithms can analyze vast datasets, identify intricate patterns, and make decisions guided by these insights. This adaptability is particularly evident in the field of Machine Learning, a subset of AI that focuses on the development of algorithms capable of learning from data.

![Venn diagram of Artificial Intelligence[^ai_venn]](./figures/ai_venn.png){width=50%}

## Machine Learning

Machine Learning (ML) is the driving force behind the remarkable progress witnessed in AI. At its core, ML provides the tools and methodologies for machines to learn from experience and iteratively improve their performance on a specific task. Unlike traditional programming, where explicit instructions dictate behavior, ML algorithms discern patterns and relationships within data, allowing them to generalize and make informed decisions when exposed to new information.

ML empowers machines to evolve autonomously, making it indispensable in an era characterized by ever-expanding datasets and complex problems. It encompasses various paradigms, including supervised learning, unsupervised learning, and reinforcement learning, each tailored to specific applications and data types.

### Artificial Neural Networks

Artificial neural networks (ANNs) are a type of machine learning algorithm that is inspired by biological neural networks in animals. ANNs are composed of individual processing units called neurons, which are organized into layers. Each neuron receives input from other neurons in the previous layer, applies a mathematical operation to that input, and passes the output to the next layer. The output of the final layer of neurons is the predicted output of the network for a given input.

The input layer is the entry point of the neural network, accepting the initial data or features. Each neuron in this layer corresponds to a specific feature, and the values assigned to these neurons represent the input data.

Hidden layers, as the name suggests, are intermediary layers that lie between the input and output layers. These layers contain neurons that process and transform the data as it flows through the network. The presence of multiple hidden layers enables ANNs to capture complex patterns and relationships within the data.

The output layer is responsible for producing the final result, whether it's a classification, prediction, or decision. The number of neurons in this layer corresponds to the desired number of output classes or the nature of the prediction task.

![Visualization of an artificial neural network[^neural_network]](./figures/neural_network.png){width=70%}

Inside every connection between neurons are numerical parameters known as weights and biases. These parameters are the essence of a neural network, as they determine the strength and significance of the connections between neurons.

Weights represent the strength of the connections between neurons. Each connection has an associated weight, which multiplies the output of one neuron before it's passed as input to the next neuron. During training, ANNs adjust these weights to minimize the difference between their predictions and the actual outcomes, effectively learning from the data.

Biases are additional parameters that are essential for fine-tuning the behavior of individual neurons. They ensure that neurons can activate even when the weighted sum of their inputs is zero. Biases enable ANNs to model complex functions and make predictions beyond linear relationships in the data.

A single neuron can be described as the sum of its inputs multiplied by their corresponding weights, plus the bias. This sum is then passed through an activation function, which determines the neuron's output. The activation function is a mathematical function that introduces non-linearity into the network, allowing it to learn complex patterns and relationships in the data. We will discuss activation functions in more detail later on.

Mathematically, the output of a neuron can be described as:

$$ y = f(\sum_{i=1}^{n} w_i x_i + b) $$

where $y$ is the output of the neuron, $f$ is the activation function, $w_i$ is the weight of the $i$th input, $x_i$ is the $i$th input, $b$ is the bias, and $n$ is the total number of inputs.

### Mechanism of operation

The operation of an ANN is divided into two fundamental phases: the forward pass and the backward pass. During the forward pass, input data is propagated through the network from the input layer to the output layer. Each neuron processes its inputs, applies the activation function, and passes the result to the next layer.

The backward pass, also known as backpropagation, is where the magic of learning happens. In this phase, the network compares its predictions with the actual outcomes, calculating the discrepancy between the two. It then propagates this error backward through the network, adjusting the weights and biases to minimize a specified loss function. The loss function serves as a measure of the error between the network's predictions and the actual target values. By diminishing this loss, the neural network enhances its capacity to make increasingly accurate predictions.[^backpropagation]

The backpropagation algorithm process depends on an optimization technique known as gradient descent. Gradient descent determines how the network's weights should be updated to minimize the loss function. To achieve this, it computes the gradient of the loss function with respect to each weight in the network. In other words, it calculates the rate of change of the loss concerning individual weights. This gradient offers crucial guidance, pointing the way toward the most rapid reduction in the loss function.

As the gradient highlights the direction of steepest descent, it becomes the "compass" for the adjustment of weights. Weight updates are made proportionally to the gradient, ensuring that changes are made more significantly in areas where the loss function is decreasing most rapidly. This iterative process of calculating gradients and updating weights continues until the network converges to a state where the loss is minimized to the greatest extent possible.[^gradient_descent]

### Convolutional Neural Networks

Convolutional neural networks (CNNs) are a specific type of ANN that has proven to be highly effective for tasks involving image and video analysis, such as object detection, segmentation, and classification. CNNs are inspired by the structure and function of the visual cortex in animals that is tuned to detect specific visual features. In a similar way, CNNs are designed to learn and extract meaningful features from raw data.

The key differentiator of CNNs is their use of convolutional layers, which enable the network to automatically learn and extract local spatial features from raw input data. In a convolutional layer, each neuron is connected only to a small, localized region of the input data, known as the receptive field. By sharing weights across all neurons within a receptive field, the network can efficiently learn to detect local patterns and features, regardless of their location within the input image.

![Neurons of a convolutional layer(right) connected to their receptive field (left)[^receptive_field]](./figures/receptive_field.png){width=50%}

CNNs typically also include pooling layers, which downsample the output of the previous layer by taking the maximum or average value within small local regions. This helps to reduce the dimensionality of the input and extract higher-level features from the local features learned in the previous convolutional layer.

![Pooling layer with a 2x2 filter[^max_pooling]](./figures/max_pooling.png){width=60%}

The final layers of a CNN are fully connected layers, which take the outputs of the previous convolutional and pooling layers and use them to make a prediction. In the case of image classification, for example, the output of the final fully connected layer might be a vector of probabilities indicating the likelihood of each possible class. In our case, instead of class probabilities, the output of the final fully connected layer will yield a single numeric value representing the predicted redshift of the observed galaxy.[^cnns]

![Typical CNN architecture[^cnn_architecture]](./figures/cnn_architecture.png){width=90%}

### Activation functions

Activation functions are mathematical functions that are applied to the output of each neuron in a neural network. They are used to introduce non-linearity into the network, which is essential for learning complex patterns and relationships in the data. The most common activation functions used in neural networks are the sigmoid, tanh, and ReLU functions. For *convolutional* neural networks, the ReLU function is typically used for all layers except the output layer, which uses a linear activation function.[^relu_in_deep_learning]

The sigmoid function is defined as:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

where $x$ is the input to the function. The sigmoid function is a smooth, S-shaped function that returns a value between 0 and 1. It is commonly used in binary classification problems, where it is used to convert the output of the final layer to a probability between 0 and 1.[^activation_functions]

![The sigmoid function[^sigmoid]](./figures/sigmoid.png){width=50%}

The tanh function is defined as:

$$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

where $x$ is the input to the function. The tanh function is also a smooth, S-shaped function that returns a value between -1 and 1. It is similar to the sigmoid function, but it is zero-centered.[^activation_functions]

![The tanh function[^tanh]](./figures/tanh.png){width=50%}

Finally, the ReLU function is defined as:

$$ReLU(x) = \begin{cases} 0 & \text{if } x < 0 \\ x & \text{otherwise} \end{cases}$$

where $x$ is the input to the function. The ReLU function is a simple function that returns 0 if the input is negative, and the input itself if the input is positive.

![The ReLU function[^relu]](./figures/relu.png){width=70%}

### Loss functions

Loss functions are mathematical functions that are used to measure the difference between the predicted output of a neural network and the actual output. They are used to guide the training process by indicating how well the network is performing. The most common loss functions used in neural networks are the mean squared error (MSE) and mean absolute error (MAE) functions.

The MSE function is defined as:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2$$

The MAE function is defined as:

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|$$

where $y_i$ is the actual output and $\hat{y_i}$ is the predicted output for the $i$th sample, and $n$ is the total number of samples.

Another loss function that is commonly used in neural networks (and the one we end up using) is the Huber loss function. It is less sensitive to outliers in data than the squared error loss. It's defined as:

$$L_{\delta} = \frac{1}{n} \sum_{i=1}^{n} \begin{cases} \frac{1}{2}(y_i - \hat{y_i})^2 & \text{if } |y_i - \hat{y_i}| \leq \delta \\ \delta |y_i - \hat{y_i}| - \frac{1}{2} \delta^2 & \text{otherwise} \end{cases}$$

where $\delta$ is a small constant.

### Optimizers

Optimizers serve as pivotal components in the training of neural networks, enabling the iterative adjustment of network weights to minimize the loss function and enhance overall performance. They play a crucial role in guiding the network's convergence toward optimal solutions. In the realm of CNNs, several optimizers are frequently employed to fine-tune model parameters. Notable among them are the stochastic gradient descent (SGD), Adam, and Adamax optimizers.

SGD is a foundational optimizer that forms the basis for many modern variants. It operates by updating weights in the direction that reduces the loss function. Although simple, SGD can be effective in optimizing neural networks, especially when coupled with proper learning rate schedules.[^sgd]

The Adam optimizer, which stands for Adaptive Moment Estimation, is a powerful and widely adopted optimization algorithm. It combines the benefits of both momentum-based updates and adaptive learning rates. Adam maintains two moving averages for each weight, resulting in efficient and adaptive weight updates. This optimizer excels in handling non-stationary or noisy objective functions, making it a popular choice for training CNNs.[^adam]

Adamax is an extension of the Adam optimizer that offers certain advantages in terms of computational efficiency. In our experiments, we found that Adamax yielded better results than Adam, which is why we ended up using it.

# Data Preparation

## Source & Composition

*N.B: The data collection and cleaning was already done by Ioannis Bellas-Velidis and Despina Hatzidimitriou who worked on Gaia's Unresolved Galaxy Classifier (UGC) and who generously provided us with the dataset. What follows is **their** process of obtaining it.[^ugc]*

The instances of the data set are selected galaxies with known redshifts. The target value is the redshift of the source galaxy or a specific value derived from it. The input data are the flux values of the sampled BP/RP (blue/red photometers, the instruments on board Gaia) spectrum of the galaxy[^bprp]. The edges of the BP spectrum are truncated by removing the first 34 and the last 6 samples, to avoid low signal-to-noise data. Similarly, the first 4 and the last 10 samples are removed from the RP spectrum. The "truncated" spectra are then concatenated to form the vector of 186 (80 BP + 106 RP) fluxes in the 366nm to 996nm wavelength range.

The galaxies used for the dataset were selected from the Sloan Digital Sky Survey Data Release 16 (SDSS DR16) archive. Galaxies with bad or missing photometry, size, or redshift were rejected. The SDSS galaxies were cross-matched with the observed Gaia galaxies. The result was a dataset of SDSS galaxies that were also observed by Gaia. Due mainly to the photometric limit of the Gaia observations, most of the high-redshift galaxies ($z>1.0$) are absent. The high redshift regime is very sparsely populated and would lead to a very unbalanced training set. Thus, an upper limit of $z=0.6$ was imposed to the SDSS redshifts, rendering a total of 520.393 sources with $0 \le z \le 0.6$ forming the final dataset.

## Analysis

Below is a sample of the dataset which shows 3 randomly selected galaxies. The first column is the redshift $z$ of the galaxy, and the remaining columns are its first 5 flux values.

| $z$       | $flux_{0}$ | $flux_{1}$ | $flux_{2}$ | $flux_{3}$ | $flux_{4}$ |
| --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 0.1735581 | 0.539      | 0.514      | 0.439      | 0.034      | 0.224      |
| 0.2647779 | -0.213     | -0.14      | 0.052      | 0.254      | 0.349      |
| 0.1288678 | 0.394      | 0.329      | 0.287      | 0.265      | 0.262      |

The figure below shows the distribution of the redshifts in the dataset. The distribution is not uniform, as the number of galaxies decreases dramatically as the redshift increases. The 99th percentile of the redshifts is approximately $0.37$, which means that *99%* of the galaxies have a redshift of $z \le 0.37$. As we will see later on, this will have an impact on the performance of the model in the redshifts of that range.

![Redshift distribution per bin of 0.01](./figures/redshift_distribution.png){width=85%}

The figure below shows five randomly selected galaxy spectra. The x-axis represents the wavelength in nanometers, and the y-axis represents the flux values. The dip in the middle of the spectra is the point where the BP and RP spectra are concatenated.

![Five randomly selected galaxy spectra](./figures/galaxy_spectra.png)

This figure shows the average flux values for galaxy spectra within various redshift ranges.

We split the redshifts into 6 ranges: $[0, 0.1)$, $[0.1, 0.2)$, $[0.2, 0.3)$, $[0.3, 0.4)$, $[0.4, 0.5)$, and $[0.5, 0.6]$. For each range, we calculated the average flux values of 1000 randomly chosen galaxies (or all available, if fewer than 1000 were in a range).

The figure shows that the average flux values decrease as the redshift increases. This might also explain why our model performs worse on higher redshifts, as the flux values are "flatter".

![Average flux values for galaxy spectra within various redshift ranges](./figures/average_flux.png)

## Preprocessing

Data preprocessing is a crucial step in machine learning, as it can significantly impact the performance of a model. It involves transforming raw data into a format that is more suitable for machine learning algorithms. This process can include various steps, such as data cleaning, feature scaling and normalization, and data splitting.[^preprocessing]

Feature scaling is a preprocessing step to standardize the range of independent variables or features in the dataset. It ensures that no single feature disproportionately influences the learning process. Common scaling techniques include min-max scaling, Z-score normalization, and robust scaling. Properly scaled features promote faster convergence and more stable training.

On our dataset, as the data was already cleaned, the only preprocessing step we had to apply was min-max scaling to the flux values, which rescales them to the range $[0, 1]$. Other scaling techniques were also tested, but min-max scaling yielded the best results.

It is mathematically defined as:

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

where $x$ is the original data and $x_{min}$ and $x_{max}$ are the minimum and maximum values of $x$, respectively.

## Splitting

Data is typically split into three sets: a training set, a validation set, and a test set. The training set is used to train the neural network, the validation set is used to fine-tune hyperparameters and monitor model performance during training, and the test set evaluates the model's performance on unseen data. This separation ensures that the model's performance metrics are reliable indicators of its generalization ability.

Our dataset was first split into a training and a test set. The training set contained 90% of the data, while the test set contained the remaining 10%. Then of the training set, 30% was used as a validation set. This resulted in a training set of 63% of the data, a validation set of 27% of the data, and a test set of 10% of the data.

This is not a common split, as the validation set is usually a smaller percentage of the original dataset. However, due to the large size of our dataset we decided to use a larger validation set because we could afford it.

![Data split](./figures/data_split.png){width=100%}

# Methodology

## Hyperparameter Optimization

Hyperparameters are the knobs and levers of a machine learning model. While regular parameters are learned from the training data (e.g., the weights in a neural network), hyperparameters are predefined settings that dictate how a model learns. These settings influence various aspects of the learning process, including the learning rate, the number of hidden layers and their units, the batch size, etc.

## Network architecture

The Convolutional Neural Network (CNN) architecture chosen for this task is specifically tailored to handle the spectral data obtained from the Gaia mission. Unlike traditional image-based CNNs, which analyze two-dimensional images, this architecture is designed to work with one-dimensional spectra.

**Input Layer**: The input layer accepts the normalized galaxy spectra. Each spectrum, represented as a 1D array of flux values at different wavelengths, is fed into the network. The input layer's size is determined by the number of data points in each spectrum.

**Convolutional Layers**: These layers are responsible for feature extraction. Convolutional filters slide over the input spectra, capturing local patterns and features at different scales. Given the varying nature of spectral data, multiple convolutional layers with different filter sizes are used to capture both fine-grained and broader features. This adaptability is essential since galaxy spectra can exhibit a wide range of shapes and characteristics.

**Pooling Layers**: After each convolutional layer, a pooling layer follows. Max-pooling is applied to reduce the spatial dimensionality of the feature maps, retaining the most important features. Pooling helps in managing the computational complexity of the network while preserving crucial information.

**Fully Connected (Dense) Layers**: Following the convolutional and pooling layers, fully connected layers are used for the final prediction. These layers gradually reduce the dimensionality of the features learned by previous layers, ultimately yielding a single numeric value: the predicted galaxy redshift.

**Output Layer**: The output layer consists of a single neuron with a linear activation function. This neuron's output represents the predicted redshift of the observed galaxy. During training, this prediction is compared to the actual redshift to compute the loss, which guides the network's weight adjustments.

## Training

### Batch size

The batch size constitutes a pivotal hyperparameter in the training process, influencing how the network learns from the data. It signifies the number of data samples that are processed in a single forward and backward pass during each training iteration. The choice of batch size carries significant implications for the network's training dynamics.

Utilizing a larger batch size can expedite the training process, as more samples are processed in parallel. This can lead to faster convergence, especially on hardware optimized for parallel computations. However, there is a trade-off, as larger batch sizes can increase the risk of overfitting. The network might memorize the training data rather than learning to generalize from it.

Conversely, a smaller batch size entails processing fewer samples at once. This can result in slower training progress, particularly on hardware with limited parallelism. However, smaller batch sizes often lead to better generalization, as the network receives a more diverse set of samples during training. It is less likely to memorize the training data and is more likely to extract meaningful patterns.[^batch_size]

In our experiments, we fine-tuned the batch size and discovered that a setting of 16 yielded the optimal balance between training efficiency and generalization performance.

### Epochs

Epochs refer to the number of times the entire training dataset is processed by the neural network. Each epoch represents a complete cycle through the dataset, during which the network updates its weights based on the observed errors. The choice of the number of epochs is another vital training hyperparameter.

Training for too few epochs may result in an underfit model. In this scenario, the network has not had sufficient exposure to the data to learn complex patterns, and it may not perform well on unseen data.

Conversely, training for an excessive number of epochs can lead to overfitting. The network may become too specialized in capturing the idiosyncrasies of the training data, diminishing its ability to generalize.

Determining the ideal number of epochs involves a balance between achieving convergence and avoiding overfitting. Typically, researchers employ techniques like early stopping, which monitors validation performance and halts training when it starts to degrade, to guide epoch selection.

In our experiments, we found through early stopping that training for 20 epochs yielded satisfactory results.

------------------------------------------------------------------------
[^gaia_overview]: Gaia Overview. ESA. September 26 2023. <https://www.esa.int/Science_Exploration/Space_Science/Gaia/Gaia_overview>
[^redshift]: Redshift. Wikipedia. September 26 2023. <https://en.wikipedia.org/wiki/Redshift>
[^ai_venn]: Artificial Intelligence relation to Generative Models subset, Venn diagram. Wikipedia. September 28 2023. <https://en.wikipedia.org/wiki/File:Artificial_Intelligence_relation_to_Generative_Models_subset,_Venn_diagram.png>
[^cnns]: Convolutional Neural Networks. Wikipedia. September 26 2023. <https://en.wikipedia.org/wiki/Convolutional_neural_network>
[^backpropagation]: Backpropagation. Wikipedia. September 27 2023. <https://www.wikipedia.com/en/Backpropagation>
[^gradient_descent]: What is Gradient Descent?. IBM. September 27 2023. <https://www.ibm.com/topics/gradient-descent>
[^preprocessing]: Why Data should be Normalized before Training a Neural Network. Towards Data Science. September 26 2023. <https://towardsdatascience.com/why-data-should-be-normalized-before-training-a-neural-network-c626b7f66c7d>
[^sgd]: Stochastic Gradient Descent (SGD). GeeksForGeeks. September 27 2023. <https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/>
[^adam]: Adam: A Method for Stochastic Optimization. arXiv. September 27 2023. <https://arxiv.org/abs/1412.6980>
[^relu_in_deep_learning]: Rectified Linear Units (ReLU) in Deep Learning. Kaggle. September 27 2023. <https://www.kaggle.com/code/dansbecker/rectified-linear-units-relu-in-deep-learning>
[^activation_functions]: Activation Functions in Neural Networks. Towards Data Science. September 27 2023. <https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6>
[^batch_size]: How to Control the Stability of Training Neural Networks With the Batch Size. Machine Learning Mastery. September 27 2023. <https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/>
[^cnn_architecture]: Phung, & Rhee,. (2019). A High-Accuracy Model Average Ensemble of Convolutional Neural Networks for Classification of Cloud Image Patches on Small Datasets. Applied Sciences. 9. 4500. 10.3390/app9214500.
[^neural_network]: Michael A. Nielsen. Neural networks and Deep Learning. Determination Press. 2015
[^receptive_field]: Input volume connected to a convolutional layer. Wikipedia Commons. September 28 2023. <https://en.wikipedia.org/wiki/File:Conv_layer.png>
[^max_pooling]: Pooling layer with a 2x2 filter and stride = 2. Wikipedia Commons. September 28 2023. <https://en.wikipedia.org/wiki/File:Max_pooling.png>
[^sigmoid]: Sigmoid function. Wikipedia Commons. September 28 2023. <https://en.wikipedia.org/wiki/File:Logistic-curve.svg>
[^tanh]: Tanh function. Wikipedia Commons. September 28 2023. <https://en.wikipedia.org/wiki/File:Hyperbolic_Tangent.svg>
[^relu]: Jason Brownlee. A Gentle Introduction to the Rectified Linear Unit (ReLU). Machine Learning Mastery. September 28 2023. <https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks>
[^bprp]: René Andrae. Sampled Mean Spectrum generator (SMSgen). Gaia Archive. September 30 2023. <https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_smsgen.html>
[^ugc]: Bellas-Velidis & Hatzidimitriou. Unresolved Galaxy Classifier (UGC). September 30 2023. <https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_ugc.html>
