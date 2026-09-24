import os
import numpy as np
from PIL import Image
from math import sqrt
from skimage.feature import blob_log
import nibabel as nib
from numpy.linalg import norm
import matplotlib.pyplot as plt
from skimage.exposure import adjust_gamma


def detect_blobs(img, pix_dim, threshold = 0.01):

	img = (img - img.min()) / (img.max()- img.min())

	max_sigma = 20 * 1e-3
	min_sigma = 8 * 1e-3

	# assume cubic
	max_sigma /= pix_dim[0]
	min_sigma /= pix_dim[0]
	num_sigma = 5

	blobs = blob_log(img,
					max_sigma=max_sigma, min_sigma=min_sigma,
					num_sigma=num_sigma,
					threshold=threshold
					)

	# Compute radii in the 3rd column.
	blobs[:, 3] = blobs[:, 3] * sqrt(3)

	return blobs


def remove_comets(blobs, img_shape):

	# Write binary image of blob centers
	img = np.zeros(img_shape, dtype = int)
	for i in range(len(blobs)):
		img[int(blobs[i,0]), int(blobs[i,1]), int(blobs[i,2])] = i

	iterations = [6,1,1]
	
	print(len(blobs))
	for it in range(iterations[1]):
		for i in np.arange(0, img.shape[1], 2).tolist():
			img[:,i,:] = np.max(img[:,i:i+2,:], axis = 1)
		img = img[:, tuple(np.arange(0, img.shape[1], 2)), :]

	for it in range(iterations[2]):
		for i in np.arange(0, img.shape[2], 2).tolist():
			img[:,:,i] = np.max(img[:,:,i:i+2], axis = 2)
		img = img[:, :, tuple(np.arange(0, img.shape[2], 2))]

	for it in range(iterations[0]):
		for i in np.arange(0, img.shape[0], 2).tolist():
			img[i,:,:] = np.max(img[i:i+2,:,:], axis = 0)
		img = img[tuple(np.arange(0, img.shape[0], 2)), :, :]

	#blobs = np.argwhere(img == 1) # Find center position after fusion
	index = np.unique(img)
	blobs = blobs[tuple(index), :]
	print(len(blobs))

	# Rescale
	#blobs[:,0] = blobs[:,0] * (img_shape[0] / img.shape[0]) # Rescale coordinates
	return blobs


def write_sphere(img, center, rad, val = 1):

	img[int(center[0]), int(center[1]), int(center[2])] = val

	for x in range(int(center[0]) - int(rad), int(center[0]) + int(rad) + 1):
		for y in range(int(center[1]) - int(rad), int(center[1]) + int(rad) + 1):
			for z in range(int(center[2]) - int(rad), int(center[2]) + int(rad) + 1):
				if norm(np.array([x,y,z]) - center[:3]) <= rad and x >= 0 and y >= 0 and z >=0 and x < img.shape[0] and y < img.shape[1] and z < img.shape[2]:
					img[x, y, z] = val
	return img


def write_blob_image(img_dim, blobs, scale = 1):

	ann_img = np.zeros(img_dim)
	val_list = np.arange(0,len(blobs))
	np.random.shuffle(val_list)

	for i in range(blobs.shape[0]):
		coord = blobs[i,:3]
		coord[0] = coord[0] * scale 
		rad = blobs[i, -1]      
		ann_img[int(coord[0]), int(coord[1]), int(coord[2])] = 1                                                                                             
		write_sphere(ann_img, coord, rad, val = 1)
	return ann_img


def write_count_blobs(blobs, ann_fn, label_fn, out_fn):

	data = nib.load(ann_fn)
	ann_img = data.get_fdata().astype(int)
	blobs = blobs[:,:3].astype(int)
	# Get correspondance between region_id and name from label file
	flab = open(label_fn, "r")
	region_names = {}

	for line in flab.readlines():
		line = line.split("|")
		region_names[int(line[0])] = line[3]
	flab.close()

	region_ids = ann_img[blobs[:,0].tolist(), blobs[:,1].tolist(), blobs[:,2].tolist()]
	values, counts = np.unique(region_ids, return_counts=True)
	
	count_dict = {}
	region_size = {}
	for i in range(len(values)):
		if values[i] >= 1 and values[i] <= ann_img.max():
			count_dict[region_names[values[i]]] = counts[i]
			region_size[region_names[values[i]]] = np.sum(ann_img == i)


	# Write count file
	sep = ";"
	f  = open(out_fn, "w")
	f.write("region_name" + sep + "cell_count" + sep + "region_size" + "\n")
	for k in count_dict.keys():
		f.write(k + sep + str(count_dict[k]) + sep + str(region_size[k]) + "\n")
	f.close()

	return count_dict


def overlay(img_slc, cell_slc, coef1, coef2):

	img_slc = (img_slc - img_slc.min()) / (img_slc.max() - img_slc.min())
	img_slc = adjust_gamma(img_slc, 0.5)

	rgb_img = np.dstack((img_slc, img_slc, img_slc)) 
	rgb_cells = np.dstack((cell_slc, np.zeros(cell_slc.shape), np.zeros(cell_slc.shape))) 
	overlay = coef1 * rgb_img + coef2 * rgb_cells
	out_img = np.hstack((rgb_img, overlay))

	return out_img


def write_2dvisu_cells(img, blob_img, out_folder, out_file, nb_cuts = 5, mip_depth = 1):

	
	x_cuts = np.round(np.linspace(0, img.shape[0], nb_cuts + 2)[1:-1]).astype(int).tolist()
	y_cuts = np.round(np.linspace(0, img.shape[1], nb_cuts + 2)[1:-1]).astype(int).tolist()
	z_cuts = np.round(np.linspace(0, img.shape[2], nb_cuts + 2)[1:-1]).astype(int).tolist()

	c = 0
	for i in x_cuts:

		img_slc = img[i, :, :]
		cell_slc = np.max(blob_img[i - mip_depth:i+mip_depth, :, :], axis = 0)
		out_img =  overlay(img_slc, cell_slc, 0.6,0.4)
		image = Image.fromarray((out_img * 255).astype(np.uint8))
		image.save(out_folder + out_file + "_visualization_cut_x_" + str(c) + ".png")
		c+=1

	c = 0
	for i in y_cuts:
		img_slc = img[:, i, :]
		cell_slc = np.max(blob_img[:, i - mip_depth:i+mip_depth, :], axis = 1)
		out_img =  overlay(img_slc, cell_slc, 0.6,0.4)
		image = Image.fromarray((out_img * 255).astype(np.uint8))
		image.save(out_folder + out_file + "_visualization_cut_y_" + str(c) + ".png")
		c+=1

	c = 0
	for i in z_cuts:
		img_slc = img[:, :, i]
		cell_slc = np.max(blob_img[:, :, i - mip_depth:i+mip_depth], axis = 2)
		out_img =  overlay(img_slc, cell_slc, 0.6,0.4)
		image = Image.fromarray((out_img * 255).astype(np.uint8))
		image.save(out_folder + out_file + "_visualization_cut_z_" + str(c) + ".png")
		c+=1



def detect_cells(img_fn, ann_fn, label_fn, out_folder, out_filename, threshold = 0.01, remove_doubles = False):

	# Open image
	data = nib.load(img_fn)
	img = data.get_fdata()
	affine = data.affine
	pix_dim = list(data.header['pixdim'][1:4])
	print(pix_dim)

	if remove_doubles:

		# Option 1
		"""
		img_comp = img[tuple(np.arange(0, img.shape[0],2)), :, :]
		img_comp = img_comp[tuple(np.arange(0, img_comp.shape[0],2)), :, :]
		blobs = detect_blobs(img_comp, pix_dim, threshold = threshold)
		blobs = np.array(blobs)
		scale = img.shape[0] / img_comp.shape[0]
		"""
		

		# Option 2
		blobs = detect_blobs(img, pix_dim, threshold = threshold)
		blobs = np.array(blobs)
		scale = 1
		blobs = remove_comets(blobs, img.shape)

	else:
		# Detect cells
		blobs = detect_blobs(img, pix_dim, threshold = threshold)
		blobs = np.array(blobs)
		scale = 1

	# Write cell coordinates csv file
	np.savetxt(out_folder + out_filename + "_coordinates.csv", blobs, delimiter=",")


	print("Write cell image")
	# Write cell image
	blob_img = write_blob_image(img.shape, blobs, scale)
	nifti_img = nib.Nifti1Image(blob_img, affine=affine)
	nib.save(img=nifti_img, filename= out_folder + out_filename + ".nii.gz")


	# Write visualization 2d images
	write_2dvisu_cells(img, blob_img, out_folder, out_filename, nb_cuts = 3, mip_depth = 2)

	# Write cell count per region
	write_count_blobs(blobs, ann_fn, label_fn, out_folder + out_filename + "_count_per_region.csv")

