from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/data/iris.data"
data = pd.read_csv(dataset_path)
X_train = data.values[:,:-1].astype(np.float64)
# assert(X_train.shape == (150, 4))
y_train = data.values[:, -1]
# assert(y_train.shape == (150,))

# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)


def d(a, b):
    return np.sum(np.square(b - a)) ** 0.5


def find_center(points):
    return np.mean(points, axis=0)


def k_means(C):
    if type(C) == list:
        C = np.array(C)
    while True:
        prev = deepcopy(C)
        assignment = np.array([
            np.argmin([d(point, c_i) for c_i in C])
            for point in X_train])
        for cid in range(C.shape[0]):
            points = X_train[assignment == cid]
            if not len(points):
                continue
            C[cid,:] = find_center(points)
        # if np.array_equal(prev, C):
        # if np.all(prev - C < 1e-3):
        if np.abs(np.linalg.norm(prev) - np.linalg.norm(C)) < 10e-3:
            break
    return C


if __name__ == "__main__":
    C = k_means(C)
    print("Final Centers")
    print(C)


# uniq_y = np.unique(y_train)
# centers = np.array([find_center(X_train[y_train == c_i]) for c_i in uniq_y])
# print("Real Centers")
# print(centers)



