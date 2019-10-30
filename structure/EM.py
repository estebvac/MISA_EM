import numpy as np
from tqdm import tqdm
from .ClusterStatistics import ClustersStatisticsList

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class EM:
    def __init__(self, data_points: np.ndarray, labels: np.ndarray, flat_mask: np.ndarray, shape: tuple):
        self.data_points = data_points
        self.labels = labels
        self.flat_mask = flat_mask
        self.shape = shape
        self.weights = []

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

    def __compute_all_mm_p(self, cluster_stat: ClustersStatisticsList, k: int):
        p_all = np.zeros((self.data_points.shape[0], k))

        current_k = 0
        for cluster in cluster_stat.cluster_statistics_list:
            p_all[:, current_k] = self.__compute_mixture_model_p(cluster.mu, cluster.sigma)
            current_k += 1

        return p_all

    def __expectation_step(self, cluster_stat: ClustersStatisticsList, k: int):
        p_all = self.__compute_all_mm_p(cluster_stat, k)
        self.weights = self.__compute_membership_weights(p_all, cluster_stat.alpha)

    def calculate_em(self, number_of_clusters: int):
        # Create the initial membership of the points
        # Initially this is categorical
        self.weights = one_hot(self.labels - 1, 3)  # initialize with K-means

        log_likelihood_prev = 0

        cluster_statistics = ClustersStatisticsList(number_of_clusters)

        # Calculate the statistic of the cluster
        cluster_statistics.compute_cluster_statistics(self.weights, self.data_points)

        for i in tqdm(range(100)):

            # E - step:
            self.__expectation_step(cluster_statistics, number_of_clusters)

            # M - step
            # Calculate the statistic of the cluster
            cluster_statistics.compute_cluster_statistics(self.weights, self.data_points)
            p_all = self.__compute_all_mm_p(cluster_statistics, number_of_clusters)

            # Check for convergence
            log_likelihood = EM.__calculate_likelihood(p_all, cluster_statistics)

            # print(log_likelihood)
            if np.abs(log_likelihood - log_likelihood_prev) < 0.01:
                break
            else:
                log_likelihood_prev = log_likelihood

        w_c = self.weights.argmax(axis=1) + 1
        np.unique(w_c)

        # MAPPING BACK: Restore original size

        flat_result = np.zeros_like(self.flat_mask)
        flat_result = np.uint8(flat_result)
        flat_result[self.flat_mask == 1] = w_c.flatten()
        segmented = flat_result.reshape(self.shape)

        return segmented
