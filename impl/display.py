import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np


def display_results(data_points, clustering, number_of_images, initialization, segmentation, gt):
    # Plot log likelihood
    plot_likelihood(clustering.loglikelihood, number_of_images)

    # Plot the comparison of the images
    plot_result_comparison(initialization[:, :, 20], segmentation[:, :, 20], gt[:, :, 20], number_of_images)

    # Distribution Initialization
    prediction_gmm = initialization.flatten()
    prediction_gmm = prediction_gmm[prediction_gmm != 0]
    plot_data_distribution(data_points, prediction_gmm, 'Clusters K-means')
    plt.savefig('results/distribution_init' + str(number_of_images) + '.png')
    plt.show()

    # Distribution EM Segmentation
    prediction_gmm = segmentation.flatten()
    prediction_gmm = prediction_gmm[prediction_gmm != 0]
    plot_data_distribution(data_points, prediction_gmm, 'Clusters EM')
    statistics = clustering.stats_output
    plot_covariances(statistics)
    plt.savefig('results/distribution_em' + str(number_of_images) + '.png')
    plt.show()

    # Distribution GT
    prediction_gmm = gt.flatten()
    prediction_gmm = prediction_gmm[prediction_gmm != 0]
    plot_data_distribution(data_points, prediction_gmm, 'Clusters GT')
    plt.savefig('results/distribution_gt' + str(number_of_images) + '.png')
    plt.show()


def plot_data_distribution(data_points, prediction_gmm, title='Clusters'):
    plt.figure(figsize=[4, 4])
    # Plot Data distribution
    ind_plt = np.arange(0, len(prediction_gmm), 20)
    data = data_points[ind_plt, :]
    prediction = prediction_gmm[ind_plt]

    ax = plt.subplot(111)
    plt.scatter(data[:, 0], data[:, 1], c=prediction, s=50, cmap='viridis')
    plt.title(title, fontweight='bold')
    plt.xlabel('T1', style='italic', fontsize=10)
    plt.ylabel('T2 Flair', style='italic', fontsize=10)
    plt.grid()

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    plt.grid(linestyle='--', linewidth=1, color=[0.4, 0.4, 0.4])


def plot_covariances(statistics):
    # colors = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    for i in range(len(statistics)):
        if len(statistics[i].mu) == 2:
            draw_ellipse(statistics[i].mu, statistics[i].sigma, alpha=0.4)


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draw an ellipse with a given position and covariance
    https://github.com/DFoly/Gaussian-Mixture-Modelling/blob/master/gaussian-mixture-model.ipynb
    """
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        cmap = colormap()
        e = Ellipse(position, nsig * width, nsig * height, angle, **kwargs)
        e.set_facecolor(cmap.colors[nsig * 60])  # np.random.rand(3))
        ax.add_patch(e)


def plot_likelihood(loglikelihood, number_of_images):
    # let us make the graph
    plt.figure(figsize=[8, 3])
    ax = plt.subplot(111)
    plt.plot(loglikelihood)

    # set the basic properties
    plt.xlabel('Iteration', style='italic', fontsize=10)
    plt.ylabel('Log-likelihood', style='italic', fontsize=10)
    plt.title('Likelihood of GMM', fontweight='bold')

    # set the limits
    min_l = np.min(loglikelihood)
    max_l = np.max(loglikelihood)
    c_max = max_l + 0.1 * np.abs(max_l - min_l)

    ax.set_xlim(0, len(loglikelihood) - 1)
    ax.set_ylim(min_l, c_max)

    # add more ticks
    ax.set_xticks(np.arange(0, len(loglikelihood), int(len(loglikelihood) / 10)))
    ax.set_yticks(np.arange(min_l, c_max, int((c_max - min_l) / 5)))

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    # tweak the title
    plt.grid(linestyle='--', linewidth=1, color=[0.4, 0.4, 0.4])
    plt.savefig('results/likelihood' + str(number_of_images) + '.pdf',
                bbox_inches="tight")
    plt.show()


def colormap():
    top = cm.get_cmap('hot', 128)
    bottom = cm.get_cmap('brg', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 155)),
                           bottom(np.linspace(0, 1, 100))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    return newcmp


def plot_result_comparison(remarked_k, remarked, gt, number_of_images):
    plt.figure(figsize=[4, 4])
    plt.imshow(remarked_k, cmap=colormap())
    plt.title('Initialization', fontweight='bold')
    plt.axis('off')
    plt.savefig('results/Initialization' + str(number_of_images) + '.pdf',
                bbox_inches="tight")

    plt.imshow(remarked, cmap=colormap())
    plt.title('EM Segmentation', fontweight='bold')
    plt.axis('off')
    plt.savefig('results/Segmented' + str(number_of_images) + '.pdf',
                bbox_inches="tight")

    plt.imshow(gt, cmap=colormap())
    plt.title('Ground Truth', fontweight='bold')
    plt.axis('off')
    plt.savefig('results/GT' + str(number_of_images) + '.pdf',
                bbox_inches="tight")

    plt.figure(figsize=[15, 4])
    plt.subplot(1, 3, 1)
    plt.imshow(remarked_k, cmap=colormap())
    plt.title('Initialization')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(remarked, cmap=colormap())
    plt.title('EM algorithm')
    plt.subplot(1, 3, 3)
    plt.imshow(gt, cmap=colormap())
    plt.title('Ground truth')
    plt.show()
