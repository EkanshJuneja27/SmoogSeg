# import os
# import scipy.io as sio
# from PIL import Image
# import numpy as np

# # Define Kaggle paths
# mat_files_path = '/kaggle/input/potsdam/imgs'
# gt_files_path = '/kaggle/input/potsdam/gt'
# output_original = '/kaggle/working/original'
# output_groundtruth = '/kaggle/working/groundtruth'

# # Create output directories
# os.makedirs(output_original, exist_ok=True)
# os.makedirs(output_groundtruth, exist_ok=True)

# def process_test_images():
#     # Read indices from labelled_test.txt
#     with open('/kaggle/input/potsdam/labelled_test.txt', 'r') as f:
#         test_indices = [line.strip() for line in f.readlines()]
    
#     print(f"Found {len(test_indices)} test images to process")
    
#     for idx in test_indices:
#         # Process original images
#         img_mat_path = os.path.join(mat_files_path, f'{idx}.mat')
#         if os.path.exists(img_mat_path):
#             try:
#                 img_data = sio.loadmat(img_mat_path)
#                 original = img_data['original']
#                 original = (original * 255).astype(np.uint8)
#                 Image.fromarray(original).save(os.path.join(output_original, f'{idx}.png'))
#                 print(f"Processed original image {idx}")
                
#                 # Process corresponding ground truth
#                 gt_mat_path = os.path.join(gt_files_path, f'{idx}.mat')
#                 if os.path.exists(gt_mat_path):
#                     gt_data = sio.loadmat(gt_mat_path)
#                     groundtruth = gt_data['groundtruth']
#                     groundtruth = (groundtruth * 255).astype(np.uint8)
#                     Image.fromarray(groundtruth).save(os.path.join(output_groundtruth, f'{idx}.png'))
#                     print(f"Processed ground truth {idx}")
#             except Exception as e:
#                 print(f"Error processing file {idx}: {str(e)}")

# def main():
#     print("Starting test image processing...")
#     process_test_images()
#     print(f"\nProcessing complete!")
#     print(f"Original images saved to: {output_original}")
#     print(f"Ground truth images saved to: {output_groundtruth}")

# if __name__ == "__main__":
#     main()




# import os
# import scipy.io as sio
# from PIL import Image
# import numpy as np

# # Define paths
# mat_files_path = '/kaggle/input/potsdam/imgs'
# gt_files_path = '/kaggle/input/potsdam/gt'
# output_original = '/kaggle/working/original'
# output_groundtruth = '/kaggle/working/groundtruth'

# # Create output directories
# os.makedirs(output_original, exist_ok=True)
# os.makedirs(output_groundtruth, exist_ok=True)

# def process_test_images():
#     # Read indices from labelled_test.txt
#     try:
#         with open('/kaggle/input/potsdam/labelled_test.txt', 'r') as f:
#             test_indices = [line.strip() for line in f.readlines()]
#         print(f"Found {len(test_indices)} test images to process")
#     except FileNotFoundError:
#         print("Error: labelled_test.txt not found.")
#         return
    
#     for idx in test_indices:
#         # Process original image
#         img_mat_path = os.path.join(mat_files_path, f'{idx}.mat')
#         if os.path.exists(img_mat_path):
#             try:
#                 img_data = sio.loadmat(img_mat_path)
#                 # Get the first array from the .mat file
#                 img_array = next(value for key, value in img_data.items() 
#                                if isinstance(value, np.ndarray) and value.ndim >= 2)
                
#                 # Convert to uint8 if needed
#                 if img_array.dtype != np.uint8:
#                     img_array = (img_array * 255).astype(np.uint8)
#                 Image.fromarray(img_array).save(os.path.join(output_original, f'{idx}.png'))
#                 print(f"Processed original image {idx}")
#             except Exception as e:
#                 print(f"Error processing original file {idx}: {str(e)}")
        
#         # Process ground truth
#         gt_mat_path = os.path.join(gt_files_path, f'{idx}.mat')
#         if os.path.exists(gt_mat_path):
#             try:
#                 gt_data = sio.loadmat(gt_mat_path)
#                 # Get the first array from the .mat file
#                 gt_array = next(value for key, value in gt_data.items() 
#                               if isinstance(value, np.ndarray) and value.ndim >= 2)
                
#                 # Convert to uint8 if needed
#                 if gt_array.dtype != np.uint8:
#                     gt_array = (gt_array * 255).astype(np.uint8)
#                 Image.fromarray(gt_array).save(os.path.join(output_groundtruth, f'{idx}.png'))
#                 print(f"Processed ground truth {idx}")
#             except Exception as e:
#                 print(f"Error processing ground truth file {idx}: {str(e)}")

# def main():
#     print("Starting test image processing...")
#     process_test_images()
#     print("\nProcessing complete!")
#     print(f"Original images saved to: {output_original}")
#     print(f"Ground truth images saved to: {output_groundtruth}")

# if __name__ == "__main__":
#     main()




# import shutil

# # Define paths
# output_dirs = {
#     'original': '/kaggle/working/original',
#     'groundtruth': '/kaggle/working/groundtruth',
#     'predictions': '/kaggle/working/predictions'
# }

# # Create zip files for each directory
# for folder_name, folder_path in output_dirs.items():
#     zip_path = f'/kaggle/working/{folder_name}_images.zip'
#     try:
#         shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
#         print(f"Successfully created {zip_path}")
#     except Exception as e:
#         print(f"Error creating zip for {folder_name}: {str(e)}")





# import os
# import scipy.io as sio
# from PIL import Image
# import numpy as np
# import shutil

# # Define Kaggle paths
# mat_files_path = '/kaggle/input/potsdam/imgs'
# gt_files_path = '/kaggle/input/potsdam/gt'
# output_original = '/kaggle/working/original'
# output_groundtruth = '/kaggle/working/groundtruth'

# # Create output directories
# os.makedirs(output_original, exist_ok=True)
# os.makedirs(output_groundtruth, exist_ok=True)

# def process_test_images():
#     # Read indices from labelled_test.txt
#     with open('/kaggle/input/potsdam/labelled_test.txt', 'r') as f:
#         test_indices = [line.strip() for line in f.readlines()]
#     print(f"Found {len(test_indices)} test images to process")
    
#     for idx in test_indices:
#         # Process original image
#         img_mat_path = os.path.join(mat_files_path, f'{idx}.mat')
#         if os.path.exists(img_mat_path):
#             try:
#                 img_data = sio.loadmat(img_mat_path)
#                 # Get the largest array from the .mat file
#                 img_array = max((value for key, value in img_data.items() 
#                                if isinstance(value, np.ndarray) and value.ndim >= 2 
#                                and key not in ['__header__', '__version__', '__globals__']),
#                               key=lambda x: x.size)
                
#                 # Process and save original image
#                 img_array = (img_array * 255).astype(np.uint8)
#                 Image.fromarray(img_array).save(os.path.join(output_original, f'{idx}.png'))
#                 print(f"Saved original image {idx}")
#             except Exception as e:
#                 print(f"Error processing original file {idx}: {str(e)}")
        
#         # Process ground truth - using same processing as original
#         gt_mat_path = os.path.join(gt_files_path, f'{idx}.mat')
#         if os.path.exists(gt_mat_path):
#             try:
#                 gt_data = sio.loadmat(gt_mat_path)
#                 # Get the largest array from the .mat file
#                 gt_array = max((value for key, value in gt_data.items() 
#                               if isinstance(value, np.ndarray) and value.ndim >= 2 
#                               and key not in ['__header__', '__version__', '__globals__']),
#                              key=lambda x: x.size)
                
#                 # Process ground truth same as original
#                 gt_array = (gt_array * 255).astype(np.uint8)
#                 Image.fromarray(gt_array).save(os.path.join(output_groundtruth, f'{idx}.png'))
#                 print(f"Saved ground truth {idx}")
#             except Exception as e:
#                 print(f"Error processing ground truth file {idx}: {str(e)}")

# def create_zip_files():
#     try:
#         shutil.make_archive('/kaggle/working/original_images', 'zip', output_original)
#         print("Created zip file for original images")
#         shutil.make_archive('/kaggle/working/groundtruth_images', 'zip', output_groundtruth)
#         print("Created zip file for ground truth images")
#     except Exception as e:
#         print(f"Error creating zip files: {str(e)}")

# def main():
#     print("Starting test image processing...")
#     process_test_images()
#     print("\nCreating zip files...")
#     create_zip_files()
#     print("\nProcessing complete!")

# if __name__ == "__main__":
#     main()





# import os
# import scipy.io as sio
# from PIL import Image
# import numpy as np
# import shutil

# # Define Kaggle paths
# mat_files_path = '/kaggle/input/potsdam/imgs'
# gt_files_path = '/kaggle/input/potsdam/gt'
# output_original = '/kaggle/working/original'
# output_groundtruth = '/kaggle/working/groundtruth'

# # Create output directories
# os.makedirs(output_original, exist_ok=True)
# os.makedirs(output_groundtruth, exist_ok=True)

# def process_test_images():
#     with open('/kaggle/input/potsdam/labelled_test.txt', 'r') as f:
#         test_indices = [line.strip() for line in f.readlines()]
#     print(f"Found {len(test_indices)} test images to process")
    
#     for idx in test_indices:
#         # Process original image
#         img_mat_path = os.path.join(mat_files_path, f'{idx}.mat')
#         if os.path.exists(img_mat_path):
#             try:
#                 img_data = sio.loadmat(img_mat_path)
#                 img_array = next(value for key, value in img_data.items() 
#                                if isinstance(value, np.ndarray) and value.ndim >= 2 
#                                and key not in ['__header__', '__version__', '__globals__'])
                
#                 # For original images
#                 if img_array.ndim == 3:  # RGB image
#                     img_array = (img_array * 255).astype(np.uint8)
#                     Image.fromarray(img_array).save(os.path.join(output_original, f'{idx}.png'))
#                     print(f"Saved original image {idx}")
#             except Exception as e:
#                 print(f"Error processing original file {idx}: {str(e)}")
        
#         # Process ground truth
#         gt_mat_path = os.path.join(gt_files_path, f'{idx}.mat')
#         if os.path.exists(gt_mat_path):
#             try:
#                 gt_data = sio.loadmat(gt_mat_path)
#                 gt_array = next(value for key, value in gt_data.items() 
#                               if isinstance(value, np.ndarray) and value.ndim >= 2 
#                               and key not in ['__header__', '__version__', '__globals__'])
                
#                 # For ground truth - preserve original values without scaling
#                 if gt_array.dtype != np.uint8:
#                     gt_array = gt_array.astype(np.uint8)
#                 Image.fromarray(gt_array, mode='L').convert('RGB').save(os.path.join(output_groundtruth, f'{idx}.png'))
#                 print(f"Saved ground truth {idx}")
#             except Exception as e:
#                 print(f"Error processing ground truth file {idx}: {str(e)}")

# def create_zip_files():
#     try:
#         shutil.make_archive('/kaggle/working/original_images', 'zip', output_original)
#         shutil.make_archive('/kaggle/working/groundtruth_images', 'zip', output_groundtruth)
#         print("Created zip files for both original and ground truth images")
#     except Exception as e:
#         print(f"Error creating zip files: {str(e)}")

# def main():
#     print("Starting test image processing...")
#     process_test_images()
#     print("\nCreating zip files...")
#     create_zip_files()
#     print("\nProcessing complete!")

# if __name__ == "__main__":
#     main()