from impl import kmeans, EM, measures
import nibabel as nib
import timeit
import matplotlib.pyplot as plt

T1_path = 'data/3/T1.nii'
T2_path = 'data/3/T2_FLAIR.nii'
groundtruth = 'data/3/LabelsForTesting.nii'
gt = nib.nifti1.load('data/3/LabelsForTesting.nii')
gt = gt.get_fdata()

start = timeit.timeit()

segmentation, center, T1_T2, label, data_mask_flat, shape, mask= kmeans.apply_kmeans(T1_path, T2_path, groundtruth)

remarked2 = measures.relabel(3, segmentation, gt, segmentation.shape)
dices = measures.calculate_dice(3, remarked2, gt)
print("KMEANS", dices)

volume = EM.calculate_EM(label, T1_T2, data_mask_flat, shape)

end = timeit.timeit()

print(end-start)

#t1_original = nib.load(T1_path)
#newImage = nib.Nifti1Image(volume, t1_original.affine, t1_original.header)

#nib.nifti1.save(newImage , "data/EM.nii");

#img = nib.nifti1.load("data/EM.nii")
#volume = img.get_fdata()

remarked = measures.relabel(3, volume, gt, volume.shape)
dices = measures.calculate_dice(3, remarked, gt)
print(dices)


plt.subplot(1,2,1)
plt.imshow(gt[:, :, 20])
plt.subplot(1,2,2)
plt.imshow(remarked2[:, :, 20])
plt.title('K-MEANS')
plt.show()


plt.subplot(1,2,1)
plt.imshow(gt[:, :, 20])
plt.subplot(1,2,2)
plt.imshow(remarked[:, :, 20])
plt.title('EM algorithm')
plt.show()

#measures.calculate_dice(3, volume, gt)
a = 3