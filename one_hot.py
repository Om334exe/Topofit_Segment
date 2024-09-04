import os
import numpy as np
import nibabel as nib

data_dir = '/local/scratch/v_karthik_mohan/data'
output_dir = '/local/scratch/v_karthik_mohan/seg_files'

# List of subject directories
subject_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Number of labels (0 to 35)
num_labels = 36


os.makedirs(output_dir, exist_ok=True)

for subject in subject_dirs:

    seg_file_path = os.path.join(data_dir, subject, 'seg35.nii.gz')

    # Load the segmentation file
    seg_img = nib.load(seg_file_path)
    seg_data = seg_img.get_fdata().astype(int)

    # Initialize one-hot encoded array
    one_hot_encoded = np.eye(num_labels, dtype=np.uint8)[seg_data]


    # Save the one-hot encoded file
    subject_output_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_output_dir, exist_ok=True)
    np.save(os.path.join(subject_output_dir, 'seg.npy'), one_hot_encoded)

    print(f'Processed {subject}: saved one-hot encoded file to {subject_output_dir}/seg.npy')
