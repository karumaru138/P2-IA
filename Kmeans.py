__authors__ = 'Pau Barredo | Victor Niccolai | Sergi Lopez'
__group__ = '20'

import numpy as np
import utils


class KMeans:
    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        X = np.array(X, dtype=float)
        if X.ndim > 2:
         
            X = X.reshape(-1, X.shape[-1])
        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def distance(X, centroids):
        """
        Calculates Euclidean distance between each point in X and each centroid
        """
        return np.sqrt(np.sum((X[:, np.newaxis, :] - centroids)**2, axis = 2))

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        if self.options['km_init'].lower() == 'first':
         
            _, idx = np.unique(self.X, axis=0, return_index=True)
            unique_X = self.X[np.sort(idx)]
            self.centroids = unique_X[:self.K].copy()

        elif self.options['km_init'].lower() == 'random':
     
            idx = np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids = self.X[idx].copy()

        else:
       
            idx = [np.random.randint(self.X.shape[0])]
            for _ in range(1, self.K):
           
                dists = self.distance(self.X, self.X[idx])        
                min_dists = np.min(dists, axis=1) ** 2        
                probs = min_dists / min_dists.sum()
                idx.append(np.random.choice(self.X.shape[0], p=probs))
            self.centroids = self.X[idx].copy()


        self.old_centroids = self.centroids + np.inf

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        dist = self.distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)  
    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids.copy()
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.K):
            points = self.X[self.labels == k]
            if len(points) > 0:
                new_centroids[k] = np.mean(points, axis=0)
            else:
     
                new_centroids[k] = self.centroids[k]
        self.centroids = new_centroids

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.all(np.abs(self.centroids - self.old_centroids) <= self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self._init_centroids()
        self.num_iter = 0
        while True:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            if self.converges() or self.num_iter >= self.options['max_iter']:
                break

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        wcd = 0.0
        for k in range(self.K):
            points = self.X[self.labels == k]
            if len(points) > 0:
                wcd += np.sum((points - self.centroids[k]) ** 2)
        return wcd / self.X.shape[0]

    def find_bestK(self, max_K, threshold=0.2):
        """
        finds the best k analysing the results up to 'max_K' clusters
        :param max_K: maximum number of clusters to try
        :param threshold: threshold for the decrease percentage (default 0.2 = 20%)
        """
        prev_wcd = None
        for k in range(2, max_K + 1):
            km = KMeans(self.X, k, self.options)
            km.fit()
            curr_wcd = km.withinClassDistance()
   
            if prev_wcd is not None:
                decrease = (prev_wcd - curr_wcd) / prev_wcd
                if decrease < threshold:      
                    self.K = k - 1
                    return
   
            prev_wcd = curr_wcd
   
        self.K = max_K
   
    def test_different_thresholds(self, max_K, thresholds=[0.1, 0.15, 0.2, 0.25, 0.3]):
        """
        Tests different thresholds to find which one gives the best K
        :param max_K: maximum number of clusters to try
        :param thresholds: list of thresholds to test
        :return: dictionary with results for each threshold
        """
        results = {}
        for thr in thresholds:
            # Guardem el K actual
            original_K = self.K
           
            # Trobem el millor K amb aquest llindar
            self.find_bestK(max_K, threshold=thr)
            results[thr] = self.K
           
            # Restaurem el K original
            self.K = original_K
       
        return results
   
   
    def distance(X, C):
        """
        Calculates the distance between each pixel and each centroid
        Args:
            X (numpy array): PxD 1st set of data points (usually data points)
            C (numpy array): KxD 2nd set of data points (usually cluster centroids points)
   
        Returns:
            dist: PxK numpy array position ij is the distance between the
            i-th point of the first set an the j-th point of the second set
        """
       
        diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
   
   
    def get_colors(centroids):
        """
        for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
        Args:
            centroids (numpy array): KxD 1st set of data points (usually centroid points)
   
        Returns:
            labels: list of K labels corresponding to one of the 11 basic colors
        """
        color_labels = []
        for centroid in centroids:
       
            pixel = centroid.reshape(1, 1, -1)        
            probs = utils.get_color_prob(pixel)          
            best_idx = np.argmax(probs.flatten())
            color_labels.append(utils.colors[best_idx])
        return color_labels
