import numpy as np
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number_of_files", dest="number_of_files", help="Number of files to split.", default=1000, type=int)
parser.add_argument("-d", "--input", dest="directory", help="Directory to split.", default='')
parser.add_argument("-ds", "--output", dest="directory_sorted", help="Directory to save random dataset.", default='')
args = parser.parse_args()

number_of_files = int(args.number_of_files)

directory = args.directory
directory_sorted = args.directory_sorted

directory_image_path = os.path.join(directory, 'images')
directory_mask_path = os.path.join(directory, 'masks')

directory_sorted_train_image_path = os.path.join(directory_sorted, 'train', 'images')
directory_sorted__train_mask_path = os.path.join(directory_sorted, 'train', 'masks')

directory_sorted_valid_image_path = os.path.join(directory_sorted, 'valid', 'images')
directory_sorted_valid_mask_path = os.path.join(directory_sorted, 'valid', 'masks')

if not os.path.exists(directory_sorted_train_image_path):
    os.makedirs(directory_sorted_train_image_path)
if not os.path.exists(directory_sorted__train_mask_path):
    os.makedirs(directory_sorted__train_mask_path)
if not os.path.exists(directory_sorted_valid_image_path):
    os.makedirs(directory_sorted_valid_image_path)
if not os.path.exists(directory_sorted_valid_mask_path):
    os.makedirs(directory_sorted_valid_mask_path)

files = [file for file in os.listdir(directory_image_path) if file.endswith('.jpg')]
random_files = np.random.choice(files, number_of_files, replace=False)

# Split the images and masks into train and test randomly
train_files = np.random.choice(files, number_of_files, replace=False)

# Random choice of valid_files in files verfying valid_files are not in train_files
valid_files = np.random.choice([file for file in files if file not in train_files], int(len(train_files)*0.2), replace=False)

# Copy the images and masks to the train sorted directory
for file in train_files:
    random_mask = file[:-4]+'.png'
    shutil.copy(os.path.join(directory_image_path, file), os.path.join(directory_sorted_train_image_path, file))
    shutil.copy(os.path.join(directory_mask_path, random_mask), os.path.join(directory_sorted__train_mask_path, random_mask))
    print("{} and {} copied.".format(file, random_mask))

# Copy the images and masks to the valid sorted directory
for file in valid_files:
    random_mask = file[:-4]+'.png'
    shutil.copy(os.path.join(directory_image_path, file), os.path.join(directory_sorted_valid_image_path, file))
    shutil.copy(os.path.join(directory_mask_path, random_mask), os.path.join(directory_sorted_valid_mask_path, random_mask))
    print("{} and {} copied.".format(file, random_mask))

print("Done.")