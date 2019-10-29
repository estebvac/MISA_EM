import numpy as np
from tqdm import tqdm
import time
from matplotlib import pyplot as plt

K = 3

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def calculate_statistics(x_all, w, K):
    cluster_stat = {}

    # Calculate the effective number of points
    alpha = np.sum(w, axis=0) / np.sum(w)
    cluster_stat['alpha'] = alpha.reshape((1, -1))

    for cluster in range(K):
        # Calculate mean for each cluster
        mu = np.zeros((1, x_all.shape[1]), np.double)
        x_l = x_all * w[:, cluster].reshape((-1, 1))
        mu = np.sum(x_l, axis=0) / np.sum(w[:, cluster])

        # Calculate the covariance of the cluster:
        xl_mu = x_all - mu
        wxl_mu = xl_mu * w[:, cluster].reshape((-1, 1))
        sigma = np.dot(wxl_mu.transpose(), xl_mu) / np.sum(w[:, cluster])

        # Storage the statistics:
        cluster_stat['mu' + str(cluster + 1)] = mu
        cluster_stat['sigma' + str(cluster + 1)] = sigma

    return cluster_stat


'''
    Calculates the mixture model per cluster.
    
    x = data
    mu_k = Mean
    sigma = covariance
'''
def calc_mixture_model_p(x, mu_k, sigma_k):
    dimensionality = x.shape[1]
    #number of techniques we are mixing.

    # Gaussian density equation
    x_mu = x - mu_k
    sigma_k_inv = np.linalg.inv(sigma_k)
    x_mu_sigma = np.dot(x_mu, sigma_k_inv)
    x_mu_sigma_x = np.sum(x_mu_sigma * x_mu, axis=1)
    det_sigma = np.sqrt(np.linalg.det(sigma_k))
    p = (1 / ((2 * np.pi) ** (dimensionality/ 2) * det_sigma)) * np.exp(-0.5 * x_mu_sigma_x)
    return p


def calc_all_mm_p(x, cluster_stat, K):
    p_all = np.zeros((x.shape[0], K))
    for cluster in range(K):
        mu_k = cluster_stat['mu' + str(cluster + 1)]
        sigma_k = cluster_stat['sigma' + str(cluster + 1)]
        p_all[:, cluster] = calc_mixture_model_p(x, mu_k, sigma_k)

    return p_all


def membership_weights(p_all, cluster_stat, K):
    a_k = cluster_stat['alpha']
    p_times_alpha = p_all * a_k
    sum_p_times_alpha = np.sum(p_times_alpha, axis=1).reshape((-1, 1))
    w_i_k = p_times_alpha * sum_p_times_alpha ** -1
    return w_i_k


def expectation_step(x_all, cluster_stat):
    p_all = calc_all_mm_p(x_all, cluster_stat, K)
    w = membership_weights(p_all, cluster_stat, K)
    return w


def calculate_likelihood(p_all, cluster_stat):
    a_k = cluster_stat['alpha']
    alfa_p = a_k * p_all
    log_likelihood = np.sum(np.log(np.sum(alfa_p, axis=1)))
    return log_likelihood


# initialization Near Neigbours
def initialization_near(x_all):
    min_x = np.min(x_all, axis=0)
    max_x = np.max(x_all, axis=0)

    space = (max_x - min_x) / (K + 1)
    distance = np.zeros((x_all.shape[0], K))

    for i in range(K):
        distance[:, i] = np.linalg.norm(x_all - (1 + i) * space, axis=1)

    w = distance.argmin(axis=1)
    w = one_hot(w, K)
    return w

def calculate_EM(label, data_points, mask_flatten, initial_shape):
    # Create the initial membership of the points
    # Initially this is categorical
    w = one_hot(label - 1, 3)  # initialize with K-means
    # w = initialization_near(x_all)
    log_likelihood_prev = 0
    # Calculate the statistic of the cluster
    cluster_stat = calculate_statistics(data_points, w, K)

    for i in tqdm(range(100)):

        # E - step:
        w = expectation_step(data_points, cluster_stat)
        w_c = w.argmax(axis=1)

        # M - step
        # Calculate the statistic of the cluster
        cluster_stat = calculate_statistics(data_points, w, 3)
        p_all = calc_all_mm_p(data_points, cluster_stat, K)

        # Check for convergence
        log_likelihood = calculate_likelihood(p_all, cluster_stat)
        # print(log_likelihood)
        if np.abs(log_likelihood - log_likelihood_prev) < 0.00001:
            print(i)
            break
        else:
            log_likelihood_prev = log_likelihood

    w_c = w.argmax(axis=1) + 1
    np.unique(w_c)

    # MAPPING BACK: Restore original size

    flat_result = np.zeros_like(mask_flatten)
    flat_result = np.uint8(flat_result)
    flat_result[mask_flatten == 1] =  w_c.flatten()
    segmented = flat_result.reshape(initial_shape)

    return segmented