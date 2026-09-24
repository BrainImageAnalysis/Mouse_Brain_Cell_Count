import ants
import os
from skimage.util import compare_images
import nibabel as nib
import numpy as np
from PIL import Image


# add write_composite_transform=True for h5 file
def ants_registration(fixed, moving, outprefix, type_of_transform, metric='mattes', **kwargs):

	outprefix = outprefix + '/'
	write_composite_transform = kwargs.get('write_composite_transform', False)
	mytx = ants.registration(fixed=fixed, moving=moving,
							 type_of_transform=type_of_transform,
							 verbose=False,
							 aff_metric=metric,
							 syn_metric=metric,
							 outprefix=outprefix,
							 write_composite_transform=write_composite_transform,
							 kwargs=kwargs)
	return mytx


def warp_image(trafo_fn, fixed_fn, moving_fn):

	fixed_img = ants.image_read(fixed_fn)
	moving_img = ants.image_read(moving_fn)

	warped_img = ants.apply_transforms(fixed=fixed_img, moving=moving_img,
													transformlist=[
														trafo_fn,
												],
												whichtoinvert=[False],
												interpolator='multiLabel'
												)
	return warped_img



def write_checkerboard(img1, img2, out_folder, nb_cuts = 3):

	x_cuts = np.round(np.linspace(0, img1.shape[0], nb_cuts + 2)[1:-1]).astype(int).tolist()
	y_cuts = np.round(np.linspace(0, img1.shape[1], nb_cuts + 2)[1:-1]).astype(int).tolist()
	z_cuts = np.round(np.linspace(0, img1.shape[2], nb_cuts + 2)[1:-1]).astype(int).tolist()

	c = 0
	for i in x_cuts:
		img1_slc = img1[i, :, :] / img1[i, :, :].max()
		img2_slc = img2[i, :, :] / img2[i, :, :].max()
		out_img = compare_images(img1_slc, img2_slc, method='checkerboard', n_tiles = (4,4))
		image = Image.fromarray((out_img * 255).astype(np.uint8))
		image.save(out_folder + "registration_visualization_cut_x_" + str(c) + ".png")
		c+=1

	c = 0
	for i in y_cuts:
		img1_slc = img1[:, i, :] / img1[:, i, :].max()
		img2_slc = img2[:, i, :] / img2[:, i, :].max()
		out_img = compare_images(img1_slc, img2_slc, method='checkerboard', n_tiles = (4,4))
		image = Image.fromarray((out_img * 255).astype(np.uint8))
		image.save(out_folder + "registration_visualization_cut_y_" + str(c) + ".png")
		c+=1

	c = 0
	for i in z_cuts:
		img1_slc = img1[:, :, i] / img1[:, :, i].max()
		img2_slc = img2[:, :, i] / img2[:, :, i].max()
		out_img = compare_images(img1_slc, img2_slc, method='checkerboard', n_tiles = (4,4))
		image = Image.fromarray((out_img * 255).astype(np.uint8))
		image.save(out_folder + "registration_visualization_cut_z_" + str(c) + ".png")
		c+=1


def register_to_atlas(fixed_fn, moving_fn, ann_fn, out_folder):

	
	# Register
	fixed_img = ants.image_read(fixed_fn)
	moving_img = ants.image_read(moving_fn)

	
	write_composite_transform = True
	type_of_transform = "SyN"
	trafo = ants_registration(fixed=fixed_img, moving=moving_img, outprefix=out_folder,
		type_of_transform=type_of_transform,
		write_composite_transform=write_composite_transform,
		reg_iterations=(50, 25, 10, 5))

	# Warp img to altas
	ants.image_write(
		trafo['warpedmovout'],
		os.path.join(out_folder, 'warped_img.nii.gz')
		)

	trafo_fn = out_folder + "/InverseComposite.h5"

	# Warp annotations to img
	warped_ann = warp_image(trafo_fn, moving_fn, ann_fn)
	ants.image_write(warped_ann, os.path.join(out_folder, 'warped_annotation.nii.gz'))
	

	data = nib.load(out_folder + "warped_img.nii.gz")
	wraped_img = data.get_fdata()

	data = nib.load(fixed_fn)
	fixed_img = data.get_fdata()


	# Write registration visualization
	write_checkerboard(wraped_img, fixed_img, out_folder, nb_cuts = 3)



