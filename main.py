from hdf5_to_nifti import *
from detect_cells import *
from register_to_atlas import *
import os

# Path to the raw images
in_files = #[path_to_img1.h5] 
out_folders = #[path_to_output_folder1]
in_files = out_folders


# Left or Right hemisphere?
side = "RH" #LH

# Path to atlas, annotations and region labels
atlas_fn = #path_to_Allen_Atlas 
annotation_fn = #path_to_Allen_Annotations
label_fn = #path_to_label_list 

for i in range(len(out_folders)):
	# Path to the raw image
	in_file = in_files[i]
	# Path to output folder
	out_folder = out_folders[i]

	# TMP show pixel dim
	img_fn = out_folder + "nifti_converted/raw_img_channel_0.nii.gz"
	data = nib.load(img_fn)
	print(data)


	### STEP 1 : Write the image to nifti
	print("Converting image to nifti")
	sub_folder = "nifti_converted/"
	# Create output folder
	if not os.path.exists(out_folder + sub_folder):
		os.makedirs(out_folder + sub_folder)

	# Convert to nifti + downsample
	hdf5_to_nifti(in_file, out_folder + sub_folder)


	### STEP 2 : Register to atlas
	print("Registering to atlas")
	subfolder = "registration/"
	# Create output folder
	if not os.path.exists(out_folder + subfolder):
		os.makedirs(out_folder + subfolder)

	img_fn = out_folder + "nifti_converted/raw_img_channel_0.nii.gz"

	register_to_atlas(atlas_fn, img_fn, annotation_fn, out_folder + subfolder)

	
	### STEP 3 : Cell detection and counting
	print("Counting cells")
	subfolder = "cell_count/thrs_0.0075_with_size/"
	# Create output folder
	if not os.path.exists(out_folder + subfolder):
		os.makedirs(out_folder + subfolder)

	img_fn = out_folder + "nifti_converted/raw_img_channel_1.nii.gz"
	warped_ann_fn = out_folder + "registration/warped_annotation.nii.gz"

	detect_cells(img_fn, warped_ann_fn, label_fn, out_folder + subfolder, "cell_detection", threshold = 0.0075, remove_doubles = True)
	

