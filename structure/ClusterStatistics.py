import numpy as np


class ClusterStatistics:

    def __init__(self, w: np.ndarray, data_points: np.ndarray):
        self.w = w
        self.alpha = 0
        self.mu = 0
        self.sigma = 0
        self.data_points = data_points

        self.__compute_mu()
        self.__compute_covariance()

    def __compute_mu(self):
        x_l = self.data_points * self.w.reshape((-1, 1))
        self.mu = np.sum(x_l, axis=0) / np.sum(self.w)

    def __compute_covariance(self):
        xl_mu = self.data_points - self.mu
        wxl_mu = xl_mu * self.w.reshape((-1, 1))
        self.sigma = np.dot(wxl_mu.transpose(), xl_mu) / np.sum(self.w)


class ClustersStatisticsList:

    def __init__(self, clusters: int):
        self.clusters = clusters
        self.cluster_statistics_list = []
        self.alpha = []

    def compute_cluster_statistics(self, weights: np.ndarray, data_points: np.ndarray):
        self.alpha = np.sum(weights, axis=0) / np.sum(weights)
        self.alpha = self.alpha.reshape((1, -1))

        self.cluster_statistics_list = []
        for cluster in range(self.clusters):
            cluster_statistics =\
                ClusterStatistics(weights[:, cluster], data_points)
            self.cluster_statistics_list.append(cluster_statistics)
