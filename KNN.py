__authors__ = 'Pau Barredo | Victor Niccolai | Sergi Lopez'
__group__ = '20'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from utils import rgb2gray


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if train_data.ndim == 4:
            train_data = rgb2gray(train_data)
       
       
       
        self.train_data = np.reshape(train_data, (train_data.shape[0], -1)).astype(float)
       
       

    def get_k_neighbours(self, test_data, k, distance_metric='euclidean'):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if test_data.ndim == 4:
           
            test_data = rgb2gray(test_data)
       
        test_data = np.reshape(test_data, (test_data.shape[0], -1)).astype(float)
       
       
        matrizDist = cdist(test_data,self.train_data,metric=distance_metric)
       
        posOrdenadas = np.argsort(matrizDist,axis=1)
       
        vecinosCercanos = posOrdenadas[:, :k]
       
       
        self.neighbors = self.labels[vecinosCercanos]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
         a       (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
       
        predict = []
       
        for fila in self.neighbors:
            votos = {}
            for ropa in fila:
                if ropa in votos:
                    votos[ropa] += 1
                else:
                    votos[ropa] = 1
            wRopa = max(votos.items(),key= operator.itemgetter(1))[0]
            predict.append(wRopa)
       
        return np.array(predict)
   
    def predict(self, test_data, k, distance_metric='euclidean'):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """
       
        self.get_k_neighbours(test_data, k, distance_metric)
        return self.get_class()