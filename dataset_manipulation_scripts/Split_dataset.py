# SPLIT INTO TRAIN VALID TEST -> 85% 10% 5%
import os
import shutil
import random

def split_data(base_path, train_pct=0.85, valid_pct=0.10, test_pct=0.05):
    images_path = os.path.join(base_path, 'images')
    labels_path = os.path.join(base_path, 'labels')
    
    images = [img for img in os.listdir(images_path) if img.endswith('.jpg')]
    labels = [lbl for lbl in os.listdir(labels_path) if lbl.endswith('.txt')]
    
    # Ensuring corresponding image and label files are aligned
    images.sort()
    labels.sort()

    # Calculate split sizes
    total_images = len(images)
    train_size = int(total_images * train_pct)
    valid_size = int(total_images * valid_pct)
    
    # Random shuffle the files
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images[:], labels[:] = zip(*combined)

    # Create directories for the splits if they do not exist
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(base_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, split, 'labels'), exist_ok=True)

    # Function to move files
    def move_files(files, split):
        for file in files:
            image_src = os.path.join(images_path, file[0])
            label_src = os.path.join(labels_path, file[1])
            image_dest = os.path.join(base_path, split, 'images', file[0])
            label_dest = os.path.join(base_path, split, 'labels', file[1])
            shutil.move(image_src, image_dest)
            shutil.move(label_src, label_dest)

    # Move files to respective directories
    move_files(combined[:train_size], 'train')
    move_files(combined[train_size:train_size + valid_size], 'valid')
    move_files(combined[train_size + valid_size:], 'test')

base_directory = '/Users/kuba/Desktop/assigns/data' 
split_data(base_directory)