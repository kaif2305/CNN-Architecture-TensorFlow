# CIFAR-10 Image Classification with TensorFlow CNN

This project demonstrates the implementation, training, and evaluation of a Convolutional Neural Network (CNN) using TensorFlow/Keras for image classification on the CIFAR-10 dataset.

## Project Overview

The script performs the following key steps:

1.  **Dataset Loading and Preprocessing**: Loads the CIFAR-10 dataset, normalizes pixel values, and one-hot encodes the labels.
2.  **CNN Model Building**: Defines a sequential CNN model with convolutional, pooling, flatten, dense, and dropout layers.
3.  **Model Compilation**: Configures the model with an optimizer, loss function, and metrics.
4.  **Model Training**: Trains the CNN model on the training data with validation.
5.  **Model Evaluation**: Assesses the trained model's performance on the unseen test dataset.
6.  **Visualization**: Plots the training/validation accuracy and loss over epochs to visualize model learning.

## Dataset

The **CIFAR-10 dataset** is a widely used benchmark in computer vision. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The 10 different classes represent airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## CNN Architecture

The CNN model is built using `tf.keras.Sequential` and comprises the following layers:

1.  **`Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))`**:
    * First convolutional layer with 32 filters, each of size 3x3.
    * Uses ReLU (Rectified Linear Unit) as the activation function, which introduces non-linearity.
    * `input_shape=(32, 32, 3)` specifies the dimensions of the input images (height, width, color channels).
2.  **`MaxPooling2D((2, 2))`**:
    * Max pooling layer with a 2x2 pool size. This downsamples the feature maps, reducing their spatial dimensions and making the model more robust to small variations in object position.
3.  **`Conv2D(64, (3, 3), activation='relu')`**:
    * Second convolutional layer with 64 filters, also of size 3x3. It learns more complex features from the downsampled output of the previous layer.
4.  **`MaxPooling2D((2, 2))`**:
    * Another max pooling layer for further dimensionality reduction.
5.  **`Flatten()`**:
    * Flattens the 2D feature maps into a 1D vector. This prepares the data for input into the fully connected (dense) layers.
6.  **`Dense(128, activation='relu')`**:
    * A fully connected layer with 128 neurons and ReLU activation. This layer learns high-level patterns from the flattened features.
7.  **`Dropout(0.5)`**:
    * A dropout layer that randomly sets 50% of the input units to 0 at each update during training. This helps prevent overfitting by reducing complex co-adaptations between neurons.
8.  **`Dense(10, activation='softmax')`**:
    * The output layer with 10 neurons, corresponding to the 10 classes in CIFAR-10.
    * `softmax` activation function is used to output a probability distribution over the classes, where the sum of probabilities for all classes equals 1.

The `model.summary()` output provides a detailed breakdown of each layer, including its output shape and the number of trainable parameters.

## Training and Evaluation

### Compilation

The model is compiled with:
* **Optimizer**: `'adam'` (Adaptive Moment Estimation), a popular optimization algorithm that efficiently handles sparse gradients and non-stationary objectives.
* **Loss Function**: `'categorical_crossentropy'`, suitable for multi-class classification problems where labels are one-hot encoded.
* **Metrics**: `['accuracy']` to monitor the classification accuracy during training and evaluation.

### Training Process

The model is trained for 10 epochs using a `batch_size` of 64. A `validation_split` of 0.2 means 20% of the training data is held out for validation, allowing monitoring of the model's performance on unseen data during training and helping to detect overfitting.

### Evaluation

After training, the model is evaluated on the separate `X_test` and `y_test` datasets to get a final, unbiased assessment of its generalization capability.

## Visualization

The script generates two plots using `matplotlib.pyplot`:

1.  **Model Accuracy**: Shows the training accuracy and validation accuracy over each epoch. This plot helps to understand if the model is learning effectively and if there's a significant gap between training and validation performance (indicating overfitting).
2.  **Model Loss**: Displays the training loss and validation loss over each epoch. This plot helps to observe the convergence of the model and identify potential issues like underfitting or overfitting.

## Setup and Usage

### Prerequisites

* Python 3.x
* `tensorflow` (including Keras)
* `matplotlib`
* `numpy` (usually installed with other libraries)

You can install the required libraries using pip:

```bash
pip install tensorflow matplotlib numpy