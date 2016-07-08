# Semisupervised Learning on MNIST
The goal of this brief exercise was to make semisupervised classifiers for MNIST data.

This was done by implementing a memory-efficient version of the label propagation algorithm described in
Zhu, Xiaojin, and Zoubin Ghahramani. Learning from labeled and unlabeled data with label propagation. 
        Technical Report CMU-CALD-02-107, Carnegie Mellon University, 2002.

## Report
You can read about how the went in a Jupyter Notebook located at notebooks/Interactive\_report.ipynb

This notebook calls on label\_propagation.py, which contains a LabelPropagation class as well as a couple of functions of general utility.

The other two notebooks show some scraps of code from other approaches that I tried.

## Installation
For this to run, you will need:
1. Python 2 with the scientific packages:
Numpy
Scikit-learn
Matplotlib
Tensorflow
(developed on Python 2.7 Anaconda installation with tensorflow)
2. Jupyter Notebook
3. MNIST dataset (locate this in the dat directory)
first download and unzip the MNIST dataset, available from http://yann.lecun.com/exdb/mnist/ at time of writing, and put its four files in the dat directory. Your dat directory should look as follows:
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte


