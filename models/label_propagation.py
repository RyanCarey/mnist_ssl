from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt


class LabelPropagation:
    def __init__(self, n_nearest_neighbours, iters = 10, normalize=True, verbose = False):
        """
        arguments:
            n_nearest_neighbours: how many nearest-neighbour indices to find
            iters: how many iterations the labels should be propagated for
            normalize: correct for bandwagon effects among classes (bool). Is useful
            if class frequences are roughly equally balanced.
        """
        self._n_nearest_neighbours = n_nearest_neighbours
        self._neighbour_indices = None
        self.iters = iters
        self.normalize = normalize
        self.verbose = verbose
        
    def _precompute_nearest_neighbour_indices(self, X):
        """ get a lookup table of nearest neighbours for each datapoint

        arguments:
        X: a dataset in which to find the nearest neighbours numpy array (n x m)

        returns:
        nearest_indices: the nearest k neighbours for each of the n datapoints, not 
        necessarily given in order of proximity(n x k) 
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
        nearest_labels = ytr[nearest_indices]
        nearest_labels = nearest_labels.sum(axis=0)
        nearest_labels /= nearest_labels.sum()
        return nearest_labels
    
    def _propagate_labels(self, ytr):
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
