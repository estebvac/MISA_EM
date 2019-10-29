import numpy as np

def calculate_dice(tissues : int, volume, gt) -> dict:

    volume = volume.reshape((-1, 1)).flatten()
    gt = gt.reshape((-1, 1)).flatten()
    results = dict();

    for tissue_id in range(1, tissues+1):
        volume_counter = 1*(volume == tissue_id)
        mask_counter = 1*(gt == tissue_id)
        dice_tissue= __dice(volume_counter, mask_counter)
        results[tissue_id] = dice_tissue

    return results


def __dice(volume_counter, mask_counter):

    num = 2 * (volume_counter * mask_counter).sum()
    den = volume_counter.sum() + mask_counter.sum()

    dice_tissue = num / den

    return dice_tissue


def relabel(tissues: int, volume, gt, shape_1) -> dict:
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
            dices_per_tissue[current_tissue-1] = (dice_tissue)

        correct_tissue = np.argmax(dices_per_tissue)+1
        matching_labels[tissue_id-1]  = correct_tissue
        tissues_available.remove(correct_tissue )

    remarked = np.zeros_like(volume)
    remarked[volume == 1] = matching_labels[0]
    remarked[volume == 2] = matching_labels[1]
    remarked[volume == 3] = matching_labels[2]

    return remarked.reshape(shape_1)


    return results
