import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        """
        Train OVR binary classifiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        """
        result = {}
        for label in self.labels:
            y_relabeled = (y == label).astype(np.float32)
            clf = svm.LinearSVC(random_state=12345)
            clf.fit(X, y_relabeled)
            result[label] = clf
        return result

    def bsvm_ovo_student(self, X, y):
        """
        Train OVO binary classifiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        """
        result = {}
        for label_a in self.labels:
            for label_b in self.labels:
                if label_a >= label_b:
                    continue

                sel = (y == label_a) | (y == label_b)
                X_ab = X[sel]
                y_ab = (y[sel] == label_a).astype(np.float32)
                clf = svm.LinearSVC()
                clf.fit(X_ab, y_ab)
                result[(label_a, label_b)] = clf

        return result

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = []
        for label in self.labels:
            scores.append(self.binary_svm[label].decision_function(X))
        return np.array(scores).T


    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        raw = []
        for labels, clf in self.binary_svm.items():
            raw.append([labels[int(1-i)] for i in clf.predict(X)])
        raw = np.array(raw).T
        return np.array(
            [np.bincount(preds, minlength=max(self.labels) + 1) for preds in
             raw])

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arguments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        norm = 0.5 * np.linalg.norm(W) ** 2.
        summation = 0
        WX = W.dot(X.T)
        for i in range(X.shape[0]):
            delta = np.zeros(W.shape[0])
            delta[y[i]] = 1
            col = 1 - delta + WX[:,i]
            summation += (np.max(col) - W[y[i]].dot(X[i]))
        return norm + C * summation

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arguments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The gradient of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        WX = W.dot(X.T)
        result = np.zeros(W.shape)
        for i in range(X.shape[0]):
            psi = np.zeros(W.shape)
            delta = np.zeros(W.shape[0])
            delta[y[i]] = 1
            col = 1 - delta + WX[:, i]
            y_hat = np.argmax(col)
            psi[y_hat] = np.ones(W.shape[1]) * X[i]
            psi[y[i]] = -X[i]
            result += psi
        return W + C * result
