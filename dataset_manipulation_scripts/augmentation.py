import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

class DataAugmentor:
    def __init__(self, 
                 input_dirs: dict, 
                 output_dirs: dict = None, 
                 target_size: Tuple[int, int] = (640, 640)):
        self.input_dirs = input_dirs
        self.output_dirs = output_dirs or input_dirs
        self.target_size = target_size
        
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def _augment_image(self, image_path: str, aug_type: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        image = cv2.resize(image, self.target_size)
        
        if aug_type == 'grayscale':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        elif aug_type == 'brightness':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, np.random.randint(-50, 50))
            v = np.clip(v, 0, 255)
            image = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        
        elif aug_type == 'contrast':
            alpha = np.random.uniform(0.5, 1.5)
            beta = np.random.randint(-50, 50)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        elif aug_type == 'blur':
            kernel_size = np.random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image
    
    def augment_dataset(self, num_workers: int = None):
        processed_count = 0
        
        for split in ['train', 'valid', 'test']:
            image_dir = os.path.join(self.input_dirs['images'], split, 'images')
            label_dir = os.path.join(self.input_dirs['labels'], split, 'labels')
            
            # Ensure output directories for this split exist
            output_image_dir = os.path.join(self.output_dirs['images'], split)
            output_label_dir = os.path.join(self.output_dirs['labels'], split)
            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)
            
            print(f"Processing {split} split:")
            print(f"Image directory: {image_dir}")
            print(f"Label directory: {label_dir}")
            
            for filename in os.listdir(image_dir):
                # Skip if not an image file
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    continue
                
                # Construct full paths
                image_path = os.path.join(image_dir, filename)
                label_filename = os.path.splitext(filename)[0] + '.txt'
                label_path = os.path.join(label_dir, label_filename)
                
                # Verify both image and label exist
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue
                if not os.path.exists(label_path):
                    print(f"Label not found: {label_path}")
                    continue
                
                # Augmentation types
                aug_types = ['original', 'grayscale', 'brightness', 'contrast', 'blur']
                
                for aug_type in aug_types:
                    # Augment image
                    if aug_type == 'original':
                        aug_image = cv2.imread(image_path)
                        aug_image = cv2.resize(aug_image, self.target_size)
                        new_filename = filename
                    else:
                        aug_image = self._augment_image(image_path, aug_type)
                        new_filename = f"{os.path.splitext(filename)[0]}_{aug_type}{os.path.splitext(filename)[1]}"
                    
                    # Save augmented image
                    output_image_path = os.path.join(output_image_dir, new_filename)
                    cv2.imwrite(output_image_path, aug_image)
                    
                    # Save corresponding label
                    output_label_path = os.path.join(output_label_dir, os.path.splitext(new_filename)[0] + '.txt')
                    with open(label_path, 'r') as f_in, open(output_label_path, 'w') as f_out:
                        f_out.write(f_in.read())
                    
                    processed_count += 1
        
        print(f"Total images processed: {processed_count}")

def main():
    base_dir = '/Users/kuba/Desktop/assigns/data_new'
    input_dirs = {
        'images': base_dir,
        'labels': base_dir
    }
    
    output_base_dir = os.path.join(base_dir, 'augmented')
    output_dirs = {
        'images': os.path.join(output_base_dir, 'images'),
        'labels': os.path.join(output_base_dir, 'labels')
    }
    
    augmentor = DataAugmentor(
        input_dirs=input_dirs, 
        output_dirs=output_dirs, 
        target_size=(640, 640)
    )
    
    augmentor.augment_dataset()

if __name__ == "__main__":
    main()