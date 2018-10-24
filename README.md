## Convolutional Variational Autoencoder

This repository contains a convolutional implementation of the  described in
[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
The implemented model uses the MNIST dataset for classification in addition
to the ADAM optimizer, batch normalization, weight decay, and ReLU non-linearities.

`example.ipynb` was written for a [blog post](https://dancsalo.github.io/2018/10/24/semi/)
and shows a supervised and semi-supervised approach (using the VAE framework)
to classifying patients with benign or malignant tumors
[Breast Cancer Wisconsin Diagnostic Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

### Dependencies
 * Python 3.5 or greater
 * Tensorflow 0.12.0 or greater