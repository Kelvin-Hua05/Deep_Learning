- You are allowed to use **high-level** API, such as PyTorch, TensorFlow / Keras, etc.

# 1. Using Convolutional Neural Network for Image Recognition

In this exercise, you will construct a convolutional neural network (CNN) for image recognition using the MNIST dataset. This dataset is a subset of a larger set available from NIST. The MNIST dataset contains a total of 70,000 images of handwritten digits from 0 to 9 with corresponding labels, of which 55,000 examples are in the training set, 5,000 in the validation set, and 10,000 in the test set. The digits have been size-normalized and centered in a fixed- size image, so you don’t need to do any preprocessing to the images. The picture below is an example of the MNIST data:  

## 

Please implement a CNN for image recognition using the MNIST dataset. You have to design your network architecture and **analyze** the effect of different stride size and filter size. Also, plot the learning curve, accuracy of training and test sets, and distributions of weights and biases. (The figures shown below are only for example, you should get the results by yourself.)

##

Show some examples of correctly classified and miss-classified images and **discuss your results**.

##

Following 1-2, observe the feature maps from different convolutional layers and describe how a feature map changes with increasing depth.

##

Following 1-1, please add L2 regularization to the CNN implemented in 1-1 and **discuss** its effect. 


# 2. Robustness and Interpretability on CIFAR-10

In this exercise, you will train CNNs on the **CIFAR-10** dataset (32 × 32 × 3, 10 classes) to analyze model robustness against corruptions and inspect feature attribution via Grad-CAM. 
The link where you can get the CIFAR-10 dataset is below and please refer to the website for more information about the dataset.

[https://www.cs.toronto.edu/~kriz/cifar.html](<https://www.cs.toronto.edu/~kriz/cifar.html>)
















