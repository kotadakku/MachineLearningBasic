import numpy as np
from collections import Counter
import pandas as pd


def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


def main():
    data = pd.read_csv("../Data/weather.csv")

    outlook = {'sunny': 0, 'overcast': 1, 'rainy': 2}
    tem = {'hot': 0, 'cool': 1, 'mild': 2}
    hum = {'high': 0, 'normal': 1}
    wind = {'weak': 0, 'strong': 1}
    data['outlook'], _ = pd.factorize(data['outlook'])
    data['temperature'] = data['temperature'].map(tem)
    data['humidity'] = data['humidity'].map(hum)
    data['wind'] = data['wind'].map(wind)

    X = data.iloc[:, 1:5].values
    y = data.iloc[:, -1].values

    knn = KNN(3)
    knn.fit(X, y)
    y_prediction = knn.predict(X)
    print("Du lieu du doan: ", y_prediction)
    print("Du lieu that   : ", y)

if __name__ == '__main__':
    main()