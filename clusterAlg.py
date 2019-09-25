#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the CLUSTERING class
# Author: Aya Saad
# Date created: 24 September 2019
#
#################################################################################################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

class ClusterAlgorithm:
    def __init__(self, name, train_x, train_y):
        self.name = name
        self.model = self.build_model()
        self.train_x = train_x
        self.train_y = train_y



    def train(self):
        self.model.fit(self.train_x, self.train_y)

    def evaluate(self, test_x):
        print('[INFO] ... Evaluate .... ')
        predict = []
        for x in test_x:
            pred = self.model.predict(x)
            predict.append(pred)
        return predict

    def performance(self, test_y, pred_y):
        # Evaluate the K-Means clustering accuracy.
        accuracy = metrics.accuracy_score(test_y, pred_y)
        precision = metrics.precision_score(test_y, pred_y)
        recall = metrics.recall_score(test_y,pred_y)
        f1score = metrics.f1_score(test_y,pred_y)

        print("Accuracy: {}%".format(100 * accuracy))
        print("Precision: {}%".format(100 * precision))
        print("Recall: {}%".format(100 * recall))
        print("F1 Score: {}%".format(100 * f1score))
        return accuracy, precision, recall, f1score


    def build_model(self):
        if self.name == 'k-nearest':
            return self._KNearest()

    ''' 
    The k-nearest neighbor classifier is a “lazy learning” algorithm where nothing is actually “learned”. 
    Instead, the k-Nearest Neighbor (k-NN) training phase simply accepts a set of feature vectors and labels and stores them 
    Classifying a new feature vector is done as follows:
    - it accepts the feature vector, 
    - computes the distance to all stored feature vectors (normally using the Euclidean distance
    - sorts them by distance
    - returns the top k “neighbors” to the input feature vector. 
    Each of the k neighbors vote as to what they think the label of the classification is.
    '''
    def _KNearest(self, n_neighbors=1):
        return KNeighborsClassifier(n_neighbors=n_neighbors)



