import numpy as np
from tqdm import tqdm
import cv2
from .ClusterStatistics import ClustersStatisticsList

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class EM:
    def __init__(self, data_points: np.ndarray, clusters : int, tolerance:float = 0.01,
                 initialization:str = "kmeans", iterations:int = 100):
        self.data_points = data_points
        self.clusters = clusters
        self.tolerance = tolerance
        self.iterations = iterations
        self.loglikelihood = []
        if initialization == "kmeans":
            self.labels = self.__apply_kmeans()
        else:
            self.labels = self.__init_near()

        self.weights = []


    def __apply_kmeans(self):
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(self.data_points, self.clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8
        label = label + 1
        label = np.uint8(label)
        return label


    def __init_near(self):
        min_x = np.min(self.data_points, axis=0)
        max_x = np.max(self.data_points, axis=0)

        space = (max_x - min_x) / (self.clusters + 1)
        distance = np.zeros((self.data_points.shape[0], self.clusters))

        for i in range(self.clusters):
            distance[:, i] = np.linalg.norm(self.data_points - (1 + i) * space, axis=1)

        label = distance.argmin(axis=1) + 1
        label = label.reshape((-1, 1))
        return label


    @staticmethod
    def __compute_membership_weights(p_all, alpha: float):
        p_times_alpha = p_all * alpha
        sum_p_times_alpha = np.sum(p_times_alpha, axis=1).reshape((-1, 1))
        w_i_k = p_times_alpha * sum_p_times_alpha ** -1
        return w_i_k

    @staticmethod
    def __calculate_likelihood(p_all, cluster_stat: ClustersStatisticsList):
        alpha_p = cluster_stat.alpha * p_all
        log_likelihood = np.sum(np.log(np.sum(alpha_p, axis=1)))
        return log_likelihood

    def __compute_mixture_model_p(self, mu_k, sigma_k):
        dimensionality = self.data_points.shape[1]
        # number of techniques we are mixing.

        # Gaussian density equation
        x_mu = self.data_points - mu_k
        sigma_k_inv = np.linalg.inv(sigma_k)
        x_mu_sigma = np.dot(x_mu, sigma_k_inv)
        x_mu_sigma_x = np.sum(x_mu_sigma * x_mu, axis=1)
        det_sigma = np.sqrt(np.linalg.det(sigma_k))
        p = (1 / ((2 * np.pi) ** (dimensionality / 2) * det_sigma)) * np.exp(-0.5 * x_mu_sigma_x)
        return p

    def __compute_all_mm_p(self, cluster_stat: ClustersStatisticsList):
        p_all = np.zeros((self.data_points.shape[0], self.clusters))

        current_k = 0
        for cluster in cluster_stat.cluster_statistics_list:
            p_all[:, current_k] = self.__compute_mixture_model_p(cluster.mu, cluster.sigma)
            current_k += 1

        return p_all

    def __expectation_step(self, cluster_stat: ClustersStatisticsList):
        p_all = self.__compute_all_mm_p(cluster_stat)
        self.weights = self.__compute_membership_weights(p_all, cluster_stat.alpha)

    def fit(self):
        # Create the initial membership of the points
        # Initially this is categorical
        self.weights = one_hot(self.labels - 1, self.clusters)  # with the given initialization
        log_likelihood_prev = 0

        cluster_statistics = ClustersStatisticsList(self.clusters)

        # Calculate the statistic of the cluster
        cluster_statistics.compute_cluster_statistics(self.weights, self.data_points)

        for i in tqdm(range(self.iterations)):

            # E - step:
            self.__expectation_step(cluster_statistics)

            # M - step
            # Calculate the statistic of the cluster
            cluster_statistics.compute_cluster_statistics(self.weights, self.data_points)
            p_all = self.__compute_all_mm_p(cluster_statistics)

            # Check for convergence
            log_likelihood = EM.__calculate_likelihood(p_all, cluster_statistics)
            self.loglikelihood.append(log_likelihood)

            if np.abs(log_likelihood - log_likelihood_prev) < self.tolerance :
                break
            else:
                log_likelihood_prev = log_likelihood

        w_c = self.weights.argmax(axis=1) + 1

        return w_c
