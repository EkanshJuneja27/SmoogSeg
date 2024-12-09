import os
import scipy.io as sio
from PIL import Image
import numpy as np

# Define Kaggle paths
mat_files_path = '/kaggle/input/potsdam/imgs'
gt_files_path = '/kaggle/input/potsdam/gt'
output_original = '/kaggle/working/original'
output_groundtruth = '/kaggle/working/groundtruth'

# Create output directories
os.makedirs(output_original, exist_ok=True)
os.makedirs(output_groundtruth, exist_ok=True)

def process_test_images():
    # Read indices from labelled_test.txt
    with open('/kaggle/input/potsdam/labelled_test.txt', 'r') as f:
        test_indices = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(test_indices)} test images to process")
    
    for idx in test_indices:
        # Process original images
        img_mat_path = os.path.join(mat_files_path, f'{idx}.mat')
        if os.path.exists(img_mat_path):
            try:
                img_data = sio.loadmat(img_mat_path)
                original = img_data['original']
                original = (original * 255).astype(np.uint8)
                Image.fromarray(original).save(os.path.join(output_original, f'{idx}.png'))
                print(f"Processed original image {idx}")
                
                # Process corresponding ground truth
                gt_mat_path = os.path.join(gt_files_path, f'{idx}.mat')
                if os.path.exists(gt_mat_path):
                    gt_data = sio.loadmat(gt_mat_path)
                    groundtruth = gt_data['groundtruth']
                    groundtruth = (groundtruth * 255).astype(np.uint8)
                    Image.fromarray(groundtruth).save(os.path.join(output_groundtruth, f'{idx}.png'))
                    print(f"Processed ground truth {idx}")
            except Exception as e:
                print(f"Error processing file {idx}: {str(e)}")

def main():
    print("Starting test image processing...")
    process_test_images()
    print(f"\nProcessing complete!")
    print(f"Original images saved to: {output_original}")
    print(f"Ground truth images saved to: {output_groundtruth}")

if __name__ == "__main__":
    main()