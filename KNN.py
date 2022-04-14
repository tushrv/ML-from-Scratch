import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

class KNN:
    def __init__(self,k):
        self.k = k
        print(self.k)
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y 
    def euclidean_dist(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2)) # calculates euclidean distance
    def pred(self,x):
        distance = [self.euclidean_dist(x,x_train) for x_train in self.X_train] # list of distance from x to all points in X_train
        top_k = np.argsort(distance)[:self.k] # returns indexs after sorting, slicing returns top k elements
        k_neighbours = [self.y_train[i] for i in top_k] # using index from top_k get class values from y_train
        most_common = Counter(k_neighbours).most_common(1)[0][0] # choose the most common class among the list
        return most_common
    def predict(self,X):
        y_pred = [self.pred(x) for x in X] 
        return np.array(y_pred)


iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
K = [1,3,5,7,25,50,100] # for value of k that is too small or too big accuracy will generally decrease (NOTE: Elbow for KNN)
def accuracy(y_true,y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
for k in K:
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))
