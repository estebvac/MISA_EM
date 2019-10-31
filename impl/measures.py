import numpy as np
import nibabel as nib
import timeit
import matplotlib.pyplot as plt
from structure.Process_Data import Process_Data
from structure.EM import EM


def test(number_of_images: int, iterations:int = 10):

    # Read the data to process:
    T1_path = 'data/' + str(number_of_images) + '/T1.nii'
    T2_path = 'data/' + str(number_of_images) + '/T2_FLAIR.nii'
    groundtruth = 'data/' + str(number_of_images) + '/LabelsForTesting.nii'
    # Define Number of clusters
    K = 3

    start = timeit.timeit()

    data_manager = Process_Data(T1_path, T2_path, groundtruth)
    clustering = EM(data_manager.data_to_cluster, K, iterations= iterations)
    kmeans_result = clustering.labels
    cluster_result = clustering.fit()
    volume = data_manager.restore_size(cluster_result)
    kmeans_volume = data_manager.restore_size(kmeans_result)

    end = timeit.timeit()

    # Save output
    data_manager.create_nifti_mask(volume, number_of_images)

    # Plot log likelihood
    plt.plot(np.asarray(clustering.loglikelihood))
    plt.title('log likelihood')
    plt.show()

    print("____________________________________________")
    print("Image " + str(number_of_images))
    print("Execution time: ", end - start)

    gt = nib.nifti1.load(groundtruth)
    gt = gt.get_fdata()
    remarked, dices = calculate_dice_and_relabel(K, volume, gt)
    _, dices_kmeans = calculate_dice_and_relabel(K, kmeans_volume, gt)

    print('EM:      ', dices)
    print('K-means: ', dices_kmeans)
    print(" ")
    plt.subplot(1, 2, 1)
    plt.imshow(remarked[:, :, 20])
    plt.title('EM algorithm')
    plt.subplot(1, 2, 2)
    plt.imshow(gt[:, :, 20])
    plt.title('Ground truth')
    plt.show()

    return dices, dices_kmeans

def __dice(volume_counter, mask_counter):

    num = 2 * (volume_counter * mask_counter).sum()
    den = volume_counter.sum() + mask_counter.sum()

    dice_tissue = num / den

    return dice_tissue


def calculate_dice_and_relabel(tissues: int, volume, gt ) -> dict:
    shape_1 = volume.shape
    volume = volume.reshape((-1, 1)).flatten()
    gt = gt.reshape((-1, 1)).flatten()
    results = dict()
    matching_labels = np.zeros((tissues, 1))
    tissues_available = list(range(1, tissues + 1))
    for tissue_id in range(1, tissues + 1):

        dices_per_tissue = np.zeros([tissues, 1])
        for current_tissue in tissues_available:
            volume_counter = 1 * (volume == tissue_id)
            mask_counter = 1 * (gt == current_tissue)
            dice_tissue = __dice(volume_counter, mask_counter)
            dices_per_tissue[current_tissue-1] = dice_tissue

        correct_tissue = np.argmax(dices_per_tissue)+1
        matching_labels[tissue_id-1]  = correct_tissue
        tissues_available.remove(correct_tissue)
        results[tissue_id] = dices_per_tissue[correct_tissue-1][0]

    remarked = np.zeros_like(volume)
    for i in range(0, tissues):
        remarked[volume == i+1] = matching_labels[i]

    return remarked.reshape(shape_1), results
