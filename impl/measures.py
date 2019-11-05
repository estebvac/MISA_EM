import numpy as np
import nibabel as nib
from structure.Image_Manager import Image_Manager
from structure.EM import EM
from sklearn import mixture
from impl.display import display_results


def test(number_of_images: int, iterations: int = 10, initialization='kmeans',
         tolerance=0.01, display=False, cmp_scikit=False, preprocess=False,
         modlaity=0):
    # Read the data to process:
    T1_path = 'data/' + str(number_of_images) + '/T1.nii'
    T2_path = 'data/' + str(number_of_images) + '/T2_FLAIR.nii'
    groundtruth = 'data/' + str(number_of_images) + '/LabelsForTesting.nii'
    # Define Number of clusters
    K = 3

    data_manager = Image_Manager(T1_path, T2_path, groundtruth, preprocess=preprocess)
    data_to_cluster = data_manager.data_to_cluster
    if modlaity>0:
        data_to_cluster = data_to_cluster[:,modlaity-1].reshape(-1, 1)


    clustering = EM(data_to_cluster,
                    K, iterations=iterations,
                    initialization=initialization,
                    tolerance=tolerance)

    initialization_result = clustering.labels
    cluster_result = clustering.fit()
    volume = data_manager.restore_size(cluster_result)
    initialization_volume = data_manager.restore_size(initialization_result)

    # Save output
    data_manager.create_nifti_mask(volume, number_of_images)

    print("____________________________________________")
    print("Image " + str(number_of_images))
    print("Execution time: ", clustering.elapsed_time)

    gt = nib.nifti1.load(groundtruth)
    gt = gt.get_fdata()

    remarked, dices = calculate_dice_and_relabel(K, volume, gt)
    remarked_k, dices_kmeans = calculate_dice_and_relabel(K, initialization_volume, gt)

    print('EM:      ', dices)
    print('K-means: ', dices_kmeans)

    if display:
        display_results(data_manager.data_to_cluster, clustering, number_of_images, remarked_k, remarked, gt)

    if cmp_scikit:
        # fit a Gaussian Mixture Model with scikitlearn
        clf = mixture.GaussianMixture(n_components=K, covariance_type='full', init_params='kmeans')
        gmm = clf.fit_predict(data_to_cluster)
        gmm = gmm + 1
        volume_scikit = data_manager.restore_size(gmm)

        remarked_scikit, dices_scikit = calculate_dice_and_relabel(K, volume_scikit, gt)
        print('Sci-kit: ', dices_scikit)

    return dices, dices_kmeans, clustering.elapsed_time, clustering.iterations


def __dice(volume_counter, mask_counter):
    num = 2 * (volume_counter * mask_counter).sum()
    den = volume_counter.sum() + mask_counter.sum()

    dice_tissue = num / den

    return dice_tissue


def calculate_dice_and_relabel(tissues: int, volume, gt) -> dict:
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
            dices_per_tissue[current_tissue - 1] = dice_tissue

        correct_tissue = np.argmax(dices_per_tissue) + 1
        matching_labels[tissue_id - 1] = correct_tissue
        tissues_available.remove(correct_tissue)
        results[tissue_id] = dices_per_tissue[correct_tissue - 1][0]

    remarked = np.zeros_like(volume)
    for i in range(0, tissues):
        remarked[volume == i + 1] = matching_labels[i]

    return remarked.reshape(shape_1), results
