# mnist_semisupervised_exercise
Carrying out an exercise of making semisupervised classifiers for mnist digits in two days

# installation
You need:
1. Python 2 with the scientific packages:
Numpy
Scikit-learn
Matplotlib
(tested with 2.7 Anaconda installation)
2. Jupyter Notebook
3. MNIST dataset (locate this in the dat directory)
first download and unzip the MNIST dataset, available from http://yann.lecun.com/exdb/mnist/ at time of writing, and put its four files in the dat directory. Your dat directory should look as follows:
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte

# Report
* The main demonstration of semisupervised learning algorithms is jup/Interactive\_report.ipynb
* This calls on the script models/label\_propagation.py, which contains the LabelPropagation class as well as a couple of functions of general utility.
* The other two notebooks show some scraps of code from other approaches that I have tried.

