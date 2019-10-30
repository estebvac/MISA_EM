import numpy as np
import nibabel as nib
import timeit
import matplotlib.pyplot as plt
from structure.Pipeline import Pipeline


def test(number_of_images: int):

    i =3
    T1_path = 'data/' + str(i) + '/T1.nii'
    T2_path = 'data/' + str(i) + '/T2_FLAIR.nii'
    groundtruth = 'data/' + str(i) + '/LabelsForTesting.nii'

    start = timeit.timeit()

    execution = Pipeline(T1_path, T2_path, groundtruth)
    volume = execution.apply_EM()

    end = timeit.timeit()

    print("____________________________________________")
    print("Image")
    print("Execution time: ", end - start)

    gt = nib.nifti1.load(groundtruth)
    gt = gt.get_fdata()
    remarked, dices = calculate_dice_and_relabel(3, volume, gt, volume.shape)

    print(dices)
    print(" ")

    plt.subplot(1, 2, 1)
    plt.imshow(gt[:, :, 20])
    plt.subplot(1, 2, 2)
    plt.imshow(remarked[:, :, 20])
    plt.title('EM algorithm')
    plt.show()

def __dice(volume_counter, mask_counter):

    num = 2 * (volume_counter * mask_counter).sum()
    den = volume_counter.sum() + mask_counter.sum()

    dice_tissue = num / den

    return dice_tissue


def calculate_dice_and_relabel(tissues: int, volume, gt, shape_1) -> dict:
    volume = volume.reshape((-1, 1)).flatten()
    gt = gt.reshape((-1, 1)).flatten()
    results = dict();
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
        results[tissue_id] = dices_per_tissue[correct_tissue-1]

    remarked = np.zeros_like(volume)
    remarked[volume == 1] = matching_labels[0]
    remarked[volume == 2] = matching_labels[1]
    remarked[volume == 3] = matching_labels[2]

    return remarked.reshape(shape_1), results
