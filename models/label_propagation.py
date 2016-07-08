from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt


class LabelPropagation:
        """
        LabelPropagation supervised learning algorithm. 

        This class assigns labels from a labelled dataset to unlabelled data. It is designed to 
        improve the performance of a classifier trainde on all of the labelled and unlabelled data.
        
        It implements a memory-efficient version of the label propagation algorithm described in:
        Zhu, Xiaojin, and Zoubin Ghahramani. Learning from labeled and unlabeled data with label propagation. 
        Technical Report CMU-CALD-02-107, Carnegie Mellon University, 2002.

        In order to be memory-efficient, the change made is that instead of propagating labels using a 
        gaussian kernel, labels are propagated to their nearest neighbours.

        Parameters
        ----------
            n_nearest_neighbours: int, default: '5'
                How many nearest-neighbour indices labels should be propagated from. 
                This should be set higher if clustering behaviour is noisy or nonlocal.


            iters: int, default: '10'
                How many times the labels will be propagated to their neighbours. 
                This controls how far labels can propagate through the graph, and can 
                lead to overfitting or prediction of class imbalances if set to a high value.

            normalize: bool, default: 'True'
                Specifies whether the predicted probabilities of each class are normalized 
                to make predicted classes more equal. 

            verbose: bool, default: 'True'
                Specifies whether progress in populating the matrix of neighbours is printed

        Attributes
        ----------
            self._yunl_proba: ndarray, shape (num_unlabelled_examples, num_classes)
                a one-hot encoded version of the encoded labels

            self._neighbour_indices: ndarray, 
                shape (num_labelled_examples + num_unlabelled examples, n_nearest_neighbours)
                the indices of the k nearest neighbours of each example
        """

    def __init__(self, n_nearest_neighbours=5, iters = 10, normalize=True, verbose = False):
        self._n_nearest_neighbours = n_nearest_neighbours
        self._neighbour_indices = None
        self.iters = iters
        self.normalize = normalize
        self.verbose = verbose

        
    def _precompute_nearest_neighbour_indices(self, X):
        """ 
        Get a lookup table of nearest neighbours for each datapoint

        arguments:
        X: a dataset in which to find the nearest neighbours numpy array (n x m)

        """
        
        self._neighbour_indices = np.zeros((X.shape[0],self._n_nearest_neighbours), dtype=int)
        # for each datapoint, find indices of nearest datapoints
        if self.verbose:
            print('filling neighbour row 1 / {}'.format(X.shape[0]))

        for i in range(len(X)):
            if self.verbose and (i%500==0 and i>1):
                print(i, end=' ')
            # A KDTree gives an asymptotic speedup here in theory but it's pretty slow in practise
            dists = ((X-X[i])**2).sum(axis=1) 
            nearest_indices = np.argpartition(dists, self._n_nearest_neighbours)[:(self._n_nearest_neighbours+1)]
            nearest_indices = np.array([j for j in nearest_indices if j!=i]) #cannot be own nearest neighbour
            self._neighbour_indices[i] = nearest_indices

            
    def _avg_nearby_labels(self, ytr, nearest_indices):
        """ 
        Finds the average of soft labels of nearby examples. 

        arguments: 
            ytr: the full set of soft labels (ndarray; num_examples x num_labels)
            nearest_indices: indexes for nearby elements (ndarray)

        returns:
            nearest_labels: an average of nearby labels (ndarray; num_labels,)
        """

        nearest_labels = ytr[nearest_indices]
        nearest_labels = nearest_labels.sum(axis=0)
        nearest_labels /= nearest_labels.sum()
        return nearest_labels

    
    def _propagate_labels(self, ytr):
        """
        Assigns the average of nearby labels to each unlabelled example.
        Repeats this self.iters times.

        arguments:
            ytr: the full set of soft labels (ndarray; num_examples x num_labels)
        """
        if len(ytr.shape)!=2:
            raise ValueError('ytr has shape {}, must be one-hot (2D)'.format(ytr.shape))
        
        n = self._neighbour_indices.shape[0] - ytr.shape[0]   #number of unlabelled examples
        self._yunl_proba = np.ones((n, ytr.shape[1]))/ytr.shape[1] # starting labels
        new_labels = np.zeros_like(self._yunl_proba)
        for j in range(self.iters):
            ys = np.vstack((ytr,self._yunl_proba))
            for i in range(n):
                new_labels[i] = self._avg_nearby_labels(ys, self._neighbour_indices[ytr.shape[0]+i])
            self._yunl_proba = new_labels.copy()
            
        if self.normalize:
            self._yunl_proba / self._yunl_proba
        
        self.yunl = self._yunl_proba.argmax(axis=1)
        return self.yunl
        
    def propagate(self, Xtr, ytr, Xunl):
        """ 
        Uses a set of labelled examples to assign soft labels to an unlabelled set.

        arguments:
            Xtr: the set of examples that have labels 
                (ndarray; num_labelled_examples, num_features)
            ytr: the set of labels 
                (ndarray; num_labelled_examples, num_labels)
            Xunl: the set of examples that lack labels
                (ndarray; num_unlabelled_examples, num_features)

        returns:
            self.yunl: the set of soft labels for the unlabelled examples
                (ndarray; num_unlabelled_examples, num_labels)

        """

        self._precompute_nearest_neighbour_indices(np.vstack((Xtr, Xunl)))
        self._propagate_labels(ytr)
        return self.yunl


def plot_supervised_learner_performance(accs1, accs2, title, xlab, ylab, ticklabs, legend, ylim): 
    # plots a barchart using two 2D arrays, each of which is (n x k) where 
    # k is the number of cross-validation folds performed
    # based on http://matplotlib.org/examples/api/barchart_demo.html
    plt.style.use('ggplot')

    kfolds = 5

    ind = np.arange(accs1.shape[0])
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, accs1.mean(axis=1), width, color='r', yerr=accs1.std(axis=1)/np.sqrt(accs1.shape[1]))
    rects2 = ax.bar(ind + width, accs2.mean(axis=1), width, color='b', yerr=accs2.std(axis=1)/np.sqrt(accs2.shape[1]))

    # add some text for labels, title and axes ticks
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xticks(ind + width)
    ax.set_ylim(ylim)
    ax.set_xticklabels(ticklabs)

    ax.legend((rects1[0], rects2[0]), (legend), loc='lower right')

    plt.show()


def random_sample(dataset_size, n_train, n_test, random_seed=None):
    """gets indices for non-overlapping training and test set samples"""
    if (n_train + n_test) > dataset_size:
        raise ValueError('dataset_size ({}) must be bigger than sum of n_train ({}) and n_test({})'.format(
                dataset_size, n_train, n_test))
    np.random.seed(random_seed)
    np.random.permutation(5)
    random_perm = np.random.permutation(range(dataset_size))
    i_train = random_perm[:n_train]
    i_test = random_perm[(-n_test):]
    return (i_train, i_test)
