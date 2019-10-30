import nibabel as nib
import numpy as np
import cv2
from .EM import EM

class Pipeline:

    def __init__(self, path_T1: str, path_T2: str, path_GT: str):

        self.T1_path = path_T1;
        self.T2_path = path_T2;
        self.GT_path = path_GT;

    def apply_EM(self, clusters: int = 3):
        segmented, center, T1_T2, label, data_mask_flat, T1_shape =\
            self.__apply_kmeans(clusters)
        em_algorithm = EM(T1_T2, label, data_mask_flat, T1_shape)
        volume = em_algorithm.calculate_em(clusters)
        return volume

    def __apply_kmeans(self, k: int):
        T1 = nib.load(self.T1_path)
        T2 = nib.load(self.T2_path)
        T1_data = T1.get_fdata()
        T2_data = T2.get_fdata()

        T1_shape = T1_data.shape

        # Removing skull and fat according to the GT
        mask = nib.load(self.GT_path)
        data_mask = mask.get_fdata()
        data_mask = data_mask > 0
        data_mask_flat = data_mask.reshape((-1, 1))

        T1_data = T1_data * data_mask
        T2_data = T2_data * data_mask

        # Turn data into vectors
        T1_data = T1_data.reshape((-1, 1))[data_mask_flat == 1]
        T1_data = np.float32(T1_data.reshape((-1, 1)))

        T2_data = T2_data.reshape((-1, 1))[data_mask_flat == 1]
        T2_data = np.float32(T2_data.reshape((-1, 1)))

        T1_T2 = np.concatenate((T1_data, T2_data), axis=1)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, label, center = cv2.kmeans(T1_T2, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image

        label = label + 1

        label = np.uint8(label)
        flat_result = np.zeros_like(data_mask_flat)
        flat_result = np.uint8(flat_result)
        flat_result[data_mask_flat == 1] = label.flatten()
        segmented = flat_result.reshape(T1_shape)

        return segmented, center, T1_T2, label, data_mask_flat, T1_shape
