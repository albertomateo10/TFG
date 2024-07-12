import os
import shutil
import random

def copy_random_samples(source_dir, dest_dir, num_samples):
    """Copies a specific number of images and their annotations randomly.

    Args:
        source_dir (str): Source directory containing 'images' and 'annotations'.
        dest_dir (str): Destination directory where the samples will be copied.
        num_samples (int): Number of samples to copy.
    """
    
    image_dir = os.path.join(source_dir, 'images')
    annotation_dir = os.path.join(source_dir, 'annotations')

    # Get all images
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))] 

    # Randomly select images to copy
    selected_images = random.sample(image_files, num_samples)

    # Create destination directories if they do not exist
    os.makedirs(os.path.join(dest_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'annotations'), exist_ok=True)

    # Copy the images and their corresponding annotations
    for image_file in selected_images:
        image_path = os.path.join(image_dir, image_file)
        annotation_file = os.path.splitext(image_file)[0] + '.txt'  # Assume annotations are .txt files
        annotation_path = os.path.join(annotation_dir, annotation_file)

        shutil.copy(image_path, os.path.join(dest_dir, 'images'))
        shutil.copy(annotation_path, os.path.join(dest_dir, 'annotations'))

# Configuration
dataset_path = '/VisDrone/Imagen/'  # Path to your Visdrone dataset
new_dataset_path = '/Visdrone_sampled'  # Path for the new dataset
num_samples_train = 140  # Number of samples for training
num_samples_val = 40    # Number of samples for validation
num_samples_test = 60   # Number of samples for testing

# Copy samples for each set (train, val, test)
copy_random_samples(os.path.join(dataset_path, 'VisDrone2019-DET-train'), os.path.join(new_dataset_path, 'train'), num_samples_train)
copy_random_samples(os.path.join(dataset_path, 'VisDrone2019-DET-val'), os.path.join(new_dataset_path, 'val'), num_samples_val)
copy_random_samples(os.path.join(dataset_path, 'VisDrone2019-DET-test-dev'), os.path.join(new_dataset_path, 'test'), num_samples_test)
