import nibabel as nib
import numpy as np
import SimpleITK as sitk


class Image_Manager:

    def __init__(self, path_T1:str, path_T2:str, path_GT:str=None, preprocess=False):

        self.T1_path = path_T1
        self.T2_path = path_T2
        self.GT_path = path_GT
        self.shape_data = None
        self.data_mask_flat = None
        self.affine = None
        self.header = None
        self.data_to_cluster = self.__get_data_to_cluster(preprocess)


    def __get_data_to_cluster(self,preprocess):
        T1 = nib.load(self.T1_path)
        T2 = nib.load(self.T2_path)
        T1_data = T1.get_fdata()
        T2_data = T2.get_fdata()
        self.shape_data = T1_data.shape
        self.affine = T1.affine
        self.header = T1.header

        # Removing skull and fat according to the GT
        mask = nib.load(self.GT_path)
        data_mask = mask.get_fdata()
        data_mask = data_mask > 0
        data_mask_flat = data_mask.reshape((-1, 1))
        self.data_mask_flat = data_mask_flat

        # Store the images information without non-brain tissues
        if preprocess:
            corrector = sitk.GradientAnisotropicDiffusionImageFilter()
            corrector.SetTimeStep(0.0625)
            #corrector = sitk.N4BiasFieldCorrectionImageFilter()
            #corrector.SetMaximumNumberOfIterations([100])
            T1_itk = sitk.GetImageFromArray(T1_data.astype(np.float32))
            T1_itk = sitk.Cast(T1_itk, sitk.sitkFloat32)
            T1_itk = corrector.Execute(T1_itk)
            T1_data = sitk.GetArrayFromImage(T1_itk)
            T2_itk = sitk.GetImageFromArray(T2_data.astype(np.float32))
            T2_itk = sitk.Cast(T2_itk, sitk.sitkFloat32)
            T2_itk = corrector.Execute(T2_itk)
            T2_data = sitk.GetArrayFromImage(T2_itk)



        # Turn data into vectors
        T1_data = T1_data.reshape((-1, 1))[data_mask_flat == 1]
        T1_data = np.float32(T1_data.reshape((-1, 1)))

        T2_data = T2_data.reshape((-1, 1))[data_mask_flat == 1]
        T2_data = np.float32(T2_data.reshape((-1, 1)))

        T1_T2 = np.concatenate((T1_data, T2_data), axis=1)
        return T1_T2

    def restore_size(self, labels):
        flat_result = np.zeros_like(self.data_mask_flat)
        flat_result = np.uint8(flat_result)
        flat_result[self.data_mask_flat == 1] = labels.flatten()
        segmented = flat_result.reshape(self.shape_data)
        return segmented

    def create_nifti_mask(self,segmented, number_of_images:int = None):
        built_labels = nib.Nifti1Image(segmented, self.affine, self.header)
        if number_of_images is None:
            nib.save(built_labels, 'built_labels.nii')
        else:
            save_path =  'data/' + str(number_of_images) + '/built_labels.nii'
            nib.save(built_labels, save_path)

