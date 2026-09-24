from skimage.transform import rescale
from functools import reduce
import os

import numpy as np
import h5py

from PIL import Image

import nibabel as nib
import torch

from tqdm import tqdm


#conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
#conda install nibabel scikit-image h5py spyder tqdm

def reorient_affine(affine, img_shape):
	offset = affine[:3, :3] @ np.array(img_shape) * np.array([0, -1, -1])
	affine = affine @ np.diag([1, -1, -1, 1])
	affine[:3, 3] -= offset
	affine = swap_axis(affine)
	return affine


def read_hdf5_meta(in_file, verbose=True):
	with h5py.File(in_file) as f:
		root_ds = f.keys()
		if len(root_ds)>1:
			raise Exception(f"should be only one dataset but found: {';'.join(root_ds)}")

		root = next(iter(root_ds))#[0]
		image_dset = root + "/ImageData/Image"
		img = f[image_dset]
		nBytes = img.dtype.itemsize

		if verbose:
			print('\n'.join(f"{a}: {v}" for a, v in zip(
				img.attrs.keys(), [img.attrs[k] for k in img.attrs.keys()])))


		dim_labels = img.attrs['DIMENSION_LABELS']
		dim_list = img.attrs['DIMENSION_LIST']
	
		dim_list = [f[i][()] for d in dim_list for i in d]
		elem_size_m = np.array(dim_list[1:])

		dim_labels = img.attrs['DIMENSION_LABELS']
		elem_size_m = np.array([f[f'{root}/ImageData/DimensionScale{label}'][()]
								 for label in dim_labels[2:]])
		offset_m = np.array([f[f'{root}/ImageData/{label}Offset'][()]
								 for label in dim_labels[2:]])

		origin = img.attrs['DISPLAY_ORIGIN']
	
		return image_dset, img.shape, elem_size_m, offset_m, img.dtype

def swap_axis(a, dims=[2, 1, 0]):
	perm = np.zeros(a.shape)
	for i, d in enumerate(dims):
		perm[i, d] = 1
	return perm @ a


def hdf5_to_nifti(in_file, out_path):

	param = {
		'use_slices':True
	}
	reorient = False
	overwrite = True

	image_dset, shape, elem_size_m, offset_m, img_dtype = read_hdf5_meta(in_file)
	print("shape", shape)

	elem_size_mm = elem_size_m * 1.0e+3
	offset_mm = offset_m * 1.0e+3

	nC, nT, nZ, _,_ = shape

	assert (nT == 1)

	largest_sampling = max(elem_size_mm)
	data_spacing = [largest_sampling]*3
	data_spacing_s2 = [2*largest_sampling]*3

	affine = np.diag(data_spacing + [1])
	affine[0:3,3] += offset_mm
	affine_s2 = np.diag(data_spacing_s2 + [1])
	affine_s2[0:3, 3] += offset_mm


	#use_slices = as_bool(param.get('use_slices', True))
	use_slices = param.get('use_slices', True)
	print("use_slices", use_slices, type(use_slices))


	tt = torch.tensor

	with h5py.File(in_file) as f:
			img = f[image_dset]
			sf = elem_size_mm/data_spacing
			sf = sf[1:]

			vol_s2 = []
			for z in tqdm(range(nZ)):
				sl = tt(img[:, 0, z].squeeze(),dtype=torch.float32)

				if reorient:
						sl = torch.transpose(sl,2, 1)

				# it is actually a factor of about 0.25
				# so I just scale 2 times by 0.5 to avoid frequency artifacts
				tmp = torch.nn.functional.interpolate(sl[None,...],scale_factor = [sf[0],sf[1]],mode="bilinear",align_corners=True)
				#tmp = torch.nn.functional.interpolate(tmp,scale_factor = [0.5,0.5],mode="bilinear",align_corners=True)[0,...,None]
				tmp = torch.nn.functional.interpolate(tmp,scale_factor = [0.5,0.5],mode="bilinear",align_corners=True)[0,:,None,...]
				vol_s2 += [
					tmp
				]

	Vol_s2 = torch.cat(vol_s2,dim=1)
	Vol_s2 = torch.nn.functional.interpolate(Vol_s2[None,...],scale_factor = [0.5,1,1],mode="trilinear",align_corners=True)[0,...]

	if reorient:
		Vol_s2 = Vol_s2.permute([0,2, 3, 1])

	for ch in range(nC):
		
		out_file_s2 = os.path.join(out_path, f'raw_img_channel_{ch}.nii.gz')
		# reorient affine
		affine_s2_ = reorient_affine(affine_s2, Vol_s2[ch,...].shape)
		# data_units = 'mm'
		nifti_img = nib.Nifti1Image(torch.clamp(Vol_s2[ch,...],min=0,max=2**16-1).to(dtype=torch.uint16).numpy(),
									affine=affine_s2_)
		nib.save(img=nifti_img, filename=out_file_s2)

