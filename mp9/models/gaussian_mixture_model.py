"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal, mode
from scipy.cluster.vq import kmeans2

import matplotlib.pyplot as plt


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1e-6):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        # np.random.seed(42)
        # print(np.random.rand())
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        # np.array of size (n_components, n_dims)
        self._mu = np.random.rand(n_components, n_dims)

        # Initialized with uniform distribution.
        # np.array of size (n_components, 1)
        # self._pi = np.array([1.0/n_components] * n_components)[:, np.newaxis]
        self._pi = np.random.dirichlet(np.ones(n_components), size=1)
        self._pi = self._pi.reshape(self._pi.shape[1], 1)
        # self._pi = np.random.uniform(0, 1, (n_components, 1))

        # Initialized with identity.
        # np.array of size (n_components, n_dims, n_dims)
        self._sigma = np.array([
            np.identity(n_dims)
            for _ in range(n_components)
        ]) * 100.0

        self.cluster_label_map = None

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        self._mu, _ = kmeans2(x, self._n_components)
        # self._mu = x[np.random.choice(x.shape[0], self._n_components, replace=False)]

        self._pi = np.random.dirichlet(np.ones(self._n_components), size=1)
        self._pi = self._pi.reshape(self._pi.shape[1], 1)

        self._sigma = np.array([
            np.identity(self._n_dims)
            for _ in range(self._n_components)
        ]) * 100.0


        for i in range(self._max_iter):
            z_k = self._e_step(x)
            self._m_step(x, z_k)
            # print('pi', self._pi)
            #
            z_ik = self.get_posterior(x)
            y_hat = np.argmax(z_ik, axis=1)
            x_0 = x[y_hat == 0, :]
            x_1 = x[y_hat == 1, :]
            if x_0.shape[0]:
                plt.scatter(x_0[:,0], x_0[:,1], label='0', marker='o')
            if x_1.shape[0]:
                plt.scatter(x_1[:, 0], x_1[:, 1], label='1', marker='+')
            plt.plot(self._mu[0,0], self._mu[0,1], marker='s', c='r')
            plt.plot(self._mu[1,0], self._mu[1,1], marker='x', c='g')
            plt.show()


    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        z_ik = self.get_posterior(x)
        return z_ik

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        N, d = x.shape
        K = self._n_components
        self._pi = np.sum(z_ik, axis=0)[:, np.newaxis] / N
        assert(self._pi.shape == (K, 1))

        self._mu = np.array([
            z_ik[:, k].dot(x) for k in range(K)]) / (N * self._pi)
        assert(self._mu.shape == (K, d))

        sigma = np.zeros((K, d, d))
        for k in range(K):
            diff = x - self._mu[k,:]
            z_i = z_ik[:,k][:, np.newaxis]
            sigma[k,:,:] = np.dot(diff.T, z_i * diff) / (N * self._pi[k, 0])
        self._sigma = sigma + np.diag([self._reg_covar] * d)


    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N,, n_components).
        """
        ret = np.array([
            self._multivariate_gaussian(x, self._mu[k], self._sigma[k])
            for k in range(self._n_components)]).T
        assert(ret.shape == (x.shape[0], self._n_components))

        return ret

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Marginalizes:
            \sum_k p(x^(i), z^(i) = k|pi, mu, sigma) =
            \sum_k p(x^(i) | z_ik=1, pi, mu, sigma) p(z_ik = 1|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        result = self.get_conditional(x).dot(self._pi).reshape(-1)
        assert(result.shape == (x.shape[0],))

        return result + np.finfo(float).eps

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        N, d = x.shape
        p_theta = self.get_conditional(x)
        assert(p_theta.shape == (N, self._n_components))

        z_ik = p_theta * self._pi.reshape(-1)
        z_ik = z_ik / self.get_marginals(x)[:, np.newaxis]
        assert(z_ik.shape == (N, self._n_components))

        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        d = x.shape[1]
        return multivariate_normal.pdf(x, mu_k, sigma_k + np.diag([self._reg_covar] * d))

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        self.fit(x)

        z_ik = self.get_posterior(x)
        y_hat = np.argmax(z_ik, axis=1)
        assert(y_hat.shape == (x.shape[0],))

        unique_y, y_counts = np.unique(y_hat, return_counts=True)
        self.cluster_label_map = np.zeros(self._n_components)
        for y_i in unique_y:
            self.cluster_label_map[y_i] = mode(y[y_hat == y_i])[0][0]


    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """
        z_ik = self.get_posterior(x)
        y_hat = [self.cluster_label_map[p] for p in np.argmax(z_ik, axis=1)]
        assert(len(y_hat) == x.shape[0])

        return np.array(y_hat)
