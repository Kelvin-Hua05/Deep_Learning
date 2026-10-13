You are allowed to use **high-level** API, such as PyTorch, TensorFlow / Keras, etc.

# 1. Using Convolutional Neural Network for Image Recognition

In this exercise, you will construct a convolutional neural network (CNN) for image recognition using the MNIST dataset. This dataset is a subset of a larger set available from NIST. The MNIST dataset contains a total of 70,000 images of handwritten digits from 0 to 9 with corresponding labels, of which 55,000 examples are in the training set, 5,000 in the validation set, and 10,000 in the test set. The digits have been size-normalized and centered in a fixed- size image, so you don’t need to do any preprocessing to the images. The picture below is an example of the MNIST data:  

<p align="left">
  <img src="Q1_ref.png" alt="Q1_reference" width="50%">
</p>

You can download the dataset from: [http://yann.lecun.com/exdb/mnist/](<http://yann.lecun.com/exdb/mnist/>), or use the code below:

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = tf.keras.dataset.mnist
(x_train, y_train), (x_test, ytest) = mnist.load_data()
```

(*Hint: `train` can be replaced with `test` or `validation`.*)

## 1-1

Please implement a CNN for image recognition using the MNIST dataset. You have to design your network architecture and **analyze** the effect of different stride size and filter size. Also, plot the learning curve, accuracy of training and test sets, and distributions of weights and biases. (The figures shown below are only for example, you should get the results by yourself.)

<p align="left">
  <img src="Q1_ref.png" alt="Q1_reference" width="50%">
</p>

## 1-2

Show some examples of correctly classified and miss-classified images and **discuss your results**.

<p align="left">
  <img src="Q1_ref.png" alt="Q1_reference" width="50%">
</p>

## 1-3

Following 1-2, observe the feature maps from different convolutional layers and describe how a feature map changes with increasing depth.

<p align="left">
  <img src="Q1_ref.png" alt="Q1_reference" width="50%">
</p>

## 1-4

Following 1-1, please add L2 regularization to the CNN implemented in 1-1 and **discuss** its effect. 

\[E = -\frac{1}{N}\sum_{n=1}^{N}\sum_{k=1}^{K} y_{nk}\ln t_{nk}+ \alpha \lVert \omega \rVert_2^2\]

<p align="left">
  <img src="Q1_ref.png" alt="Q1_reference" width="50%">
</p>

# 2. Robustness and Interpretability on CIFAR-10

In this exercise, you will train CNNs on the **CIFAR-10** dataset (32 × 32 × 3, 10 classes) to analyze model robustness against corruptions and inspect feature attribution via Grad-CAM. 
The link where you can get the CIFAR-10 dataset is below and please refer to the website for more information about the dataset.

[https://www.cs.toronto.edu/~kriz/cifar.html](<https://www.cs.toronto.edu/~kriz/cifar.html>)

## 2-1

- Implement and train a baseline CNN on the **original CIFAR-10 dataset** (with standard normalization only, without data augmentation or added noise).
- Show your (1) network architecture, (2) learning curve (loss and accuracy), and (3) accuracy on the test set.

## 2-2

- Create two corrupted test sets from the original test data:

  (1) Gaussian Noise: Add zero-mean Gaussian noise (\(\sigma \approx 0.08 \sim 0.1\)).

  (2) Gaussian Blur: Apply a Gaussian filter (\(\sigma \approx 0.8 \sim 1.0\)).

- Evaluate your Baseline CNN on both corrupted test sets. Report the test accuracies and discuss why performance drops.

## 2-3

- Train two additional models:
  
  (1) L2 Regularization: Baseline CNN with weight decay / L2 penalty.

  (2) Data Augmentation: Baseline CNN trained with augmentations (e.g., random crop, horizontal flip, or noise). 

- Provide a summary table comparing the test accuracies (Clean, Noise, and Blur) of:

  (1) Baseline CNN

  (2) L2 Regularization

  (3) Data Augmentation 

- Discuss the results (e.g., why certain strategies improve or degrade robustness under different corruptions). 

## 2-4

- Implement Grad-CAM on the last convolutional layer. 
- Pick at least 2 test samples and visualize a side-by-side comparison for both Baseline and Data Augmented models (including Original Input, Clean+CAM, Noise+CAM, and Blur+CAM):

<p align="left">
  <img src="Q1_ref.png" alt="Q1_reference" width="50%">
</p>

- Discuss whether data augmentation helps the model maintain focus on the target object under noise and blur.
