"""
IMPORTANT: Run this script from its folder (datasets/)

Load all the information from the labels file of the main dataset and the additional dataset in a global json file.
"""
import argparse                     # Read console arguments
import os
import os.path as path
import sys
import json
from tqdm import tqdm, trange       # Use to make the progress bar
import numpy as np
import imageio                      # To import images

class Import_Config:
    def __init__(self):
        # Distribution of the data for each dataset [train, dev, test]
        self.sets_distribution = {
            'main': [0.8, 0.1, 0.1],
            'addi': [1, 0, 0]
            }

        # Better to create a symbolic link in the designated folder than changing the variables

        # Path for the main dataset (also works with symbolic links but not MacOS aliases!)
        self.path_main_images = path.join('main','images')
        self.path_main_labels = path.join('main','labels')

        # Path for the additional dataset (also works with symbolic links but not MacOS aliases!)
        self.path_addi_images = path.join('additional','images')
        self.path_addi_labels = path.join('additional','labels')

        # Extension for images files and labels
        self.img_ext = '.png'
        self.lbl_ext = '.txt'

        # Use to convert the types to number to decrease the size of the file
        self.class_mapping = {
            'Car': 0,
            'Van': 1,
            'Truck': 2,
            'Pedestrian': 3,
            'Person_sitting': 4,
            'Cyclist': 5,
            'Tram': 6,
            'Misc': 7,
            'DontCare': 8
        }

def test_path(path_to_test):
    """
    Test if a path exists and convert a symbolic link in a real path.

    Parameters:
    path_to_test      -- path to get(can be a symbolic link but no aliases)

    Returns:
    path              -- working absolute path
    """
    if not path.exists(path_to_test):
        sys.exit('ERROR: Couldn\'t find ' + path_to_test)
    elif path.islink(path_to_test):
        return path.abspath(os.readlink(path_to_test))
    return path.abspath(path_to_test)

def get_files(folder_path, file_extension):
    """
    Import name of files with a certain extension from specific folder.

    Parameters:
    folder_path      -- path to the folder containing the files (can be a symbolic link)
    file_extension   -- extension of the files to extract

    Returns:
    path             -- path of the used folder
    set_files        -- set containing the name of the files without their extensions
    """
    # Test path to see if data is accessible
    path = test_path(folder_path)

    # Get set of files from the folder
    set_files = set([f[:-len(file_extension)] for f in os.listdir(path) if f.endswith(file_extension)])

    print(' ↳ ' + str(len(set_files)) + ' files with extension \'' + file_extension + '\' in ' + path)

    return path, set_files

def import_bboxes(file_path, class_mapping):
    """
    Import the bboxes of an image in a list of dictionnary

    Argument:
    file_path        -- path to file where to extract the labels
    class_mapping    -- class mapping

    Returns:
    count_classes    -- count the number of labels for the different classes
    list_bboxes      -- list of dictionnary containing the bboxes
    """
    list_bboxes = []
    count_classes = np.zeros((len(class_mapping),), dtype=int)

    # Import
    with open(file_path) as f:
        for line in f:
            # Parse line
            line_parsed = line.strip().split(' ')

            # Fill the dictionary with the values
            bbox_dict = {}

            # Extract class
            class_id = int(class_mapping[line_parsed[0]])
            bbox_dict['class'] = class_id
            count_classes[class_id] += 1

            # Simplify KITTI labels file
            if len(line_parsed) == 5:
                bbox_dict['x1']    = float(line_parsed[1])
                bbox_dict['y1']    = float(line_parsed[2])
                bbox_dict['x2']    = float(line_parsed[3])
                bbox_dict['y2']    = float(line_parsed[4])

            # Classic KITTI labels file
            else:
                bbox_dict['trunc'] = float(line_parsed[1])
                bbox_dict['occlu'] = int(line_parsed[2])
                bbox_dict['alpha'] = float(line_parsed[3])
                bbox_dict['x1']    = float(line_parsed[4])
                bbox_dict['y1']    = float(line_parsed[5])
                bbox_dict['x2']    = float(line_parsed[6])
                bbox_dict['y2']    = float(line_parsed[7])

            list_bboxes.append(bbox_dict)

    return count_classes, list_bboxes

def get_im_info(im_path):
    """
    Get shape and mean of each channel of an image

    Parameters:
    im_path       -- Path of the image to get the info from

    Returns:
    shape          -- Shape of the image
    mean_channels  -- Mean value of the RGB channels
    """
    im = imageio.imread(im_path)

    return im.shape, np.mean(im, axis=(0, 1))

def import_data(list_names, name_dataset, dataset_info):
    """
    Import all the data in a list

    Parameters:
    list_names      -- List containing all the name of the data
    name_dataset    -- Name of the dataset (e.g. 'main', 'addi', ...)
    dataset_info    -- Dictionary containing the info on the dataset

    Returns:
    list_data       -- List of dictionary containing all the info on data and bboxes
    classes_count   -- Repartition of the classes in the dataset
    sets_count      -- Repartition of the sets in the dataset
    mean_channels   -- Array containing all the mean of the images' channels

    """
    # Get useful variables from dataset_info
    img_ext = dataset_info['image_extension']
    lbl_ext = dataset_info['label_extension']
    class_mapping = dataset_info['class_mapping']
    sets_distribution = dataset_info['sets_distribution'][name_dataset]

    path_images = dataset_info[name_dataset]['path_images']
    path_labels = dataset_info[name_dataset]['path_labels']

    print('Importing labels from ' + str(len(list_names)) + ' files:')
    print(' ↳ Sets distribution: ' + str(sets_distribution))

    # Initialize variables to store values to return
    list_data = []
    classes_count = np.zeros((len(class_mapping),), dtype=int)
    sets_count = np.zeros((3,), dtype=int)
    mean_channels = np.zeros((len(list_names),3), dtype=int)

    for idx in trange(len(list_names)):
        img_dict = {'dataset': name_dataset}

        img_dict['filename'] = list_names[idx] + img_ext

        # Get images information
        im_shape, im_mean_channels = get_im_info(path.join(path_images, (list_names[idx] + img_ext)))

        mean_channels[idx, :] = im_mean_channels

        # Add image information
        img_dict['height'] = int(im_shape[0])
        img_dict['width'] = int(im_shape[1])
        img_dict['mean_channels'] = im_mean_channels.tolist()

        # Import the labels relative to the current image
        label_path = path.join(path_labels, (list_names[idx] + lbl_ext))
        img_classes_count, bboxes = import_bboxes(label_path, class_mapping)

        img_dict['bboxes'] = bboxes

        img_dict['classes_count'] = img_classes_count.tolist()
        img_dict['set'] = int(np.random.choice(3, 1, p=sets_distribution)[0])

        # Tracking repartition of classes and sets
        classes_count += img_classes_count
        sets_count[img_dict['set']] += 1

        # Add data to the global list
        list_data.append(img_dict)

    return list_data, classes_count, sets_count, mean_channels

def generate_json(export_file_path):
    """
    Import the labels from the files and generate a .json file to save it.

    Parameters:
    export_file_path    -- Path to save the final .json file
    """
    # Get the variables used for the config
    cfg = Import_Config()

    # Initialize dictionary to hold info about the dataset
    dataset_info = {}

    # Add global variable to dataset_info
    dataset_info['class_mapping'] = cfg.class_mapping
    dataset_info['set_mapping'] = {0: 'train', 1: 'dev', 2: 'test'}
    dataset_info['sets_distribution'] = cfg.sets_distribution
    dataset_info['image_extension'] = cfg.img_ext
    dataset_info['label_extension'] = cfg.lbl_ext

    # --- MAIN DATASET ---
    print('----- IMPORT MAIN DATASET -----')
    # Get sets of images and labels from the folders
    path_main_img, set_main_img = get_files(cfg.path_main_images, cfg.img_ext)
    path_main_lbl, set_main_lbl = get_files(cfg.path_main_labels, cfg.lbl_ext)

    # Remove the images without a label or the contrary
    list_main_names = sorted(list(set_main_img & set_main_lbl))

    # Add data to info
    dataset_info['main'] = {'num_data': len(list_main_names)}
    dataset_info['main']['path_images'] = path_main_img
    dataset_info['main']['path_labels'] = path_main_lbl

    list_data, main_classes_count, main_sets_count, mean_channels = import_data(list_main_names, 'main', dataset_info)

    dataset_info['main']['classes_count'] = main_classes_count.tolist()
    dataset_info['main']['sets_count'] = main_sets_count.tolist()
    dataset_info['main']['mean_channels'] = np.mean(mean_channels, axis=0).tolist()

    print(' ↳ Classes count: ' + str(main_classes_count))
    print(' ↳ Sets count: ' + str(main_sets_count))
    print('SUCCESS: Main dataset imported.')
    print('-------------------------------')

    # --- ADDITIONAL DATASET ---
    print('----- IMPORT ADDITIONAL DATASET -----')
    if not path.exists(cfg.path_addi_images) and not path.exists(cfg.path_addi_labels):
        print('No Additional Dataset found:')
        print(' ↳ Not Found: ' + cfg.path_addi_images)
        print(' ↳ Not Found: ' + cfg.path_addi_labels)

        # --- FINAL INFORMATION ---
        dataset_info['num_data'] = len(list_data)
        dataset_info['classes_count'] = main_classes_count.tolist()
        dataset_info['sets_count'] = main_sets_count.tolist()

        dataset_info['mean_channels'] = np.mean(mean_channels, axis=0).tolist()

    else:
        # Get sets of images and labels from the folders
        path_addi_img, set_addi_img = get_files(cfg.path_addi_images, cfg.img_ext)
        path_addi_lbl, set_addi_lbl = get_files(cfg.path_addi_labels, cfg.lbl_ext)

        # Remove the images without a label or the contrary
        list_addi_names = sorted(list(set_addi_img & set_addi_lbl))

        # Add data to info
        dataset_info['addi'] = {'num_data': len(list_addi_names)}
        dataset_info['addi']['path_images'] = path_addi_img
        dataset_info['addi']['path_labels'] = path_addi_lbl

        list_addi_data, addi_classes_count, addi_sets_count, addi_mean_channels = import_data(list_addi_names, 'addi', dataset_info)

        list_data += list_addi_data

        dataset_info['addi']['classes_count'] = addi_classes_count.tolist()
        dataset_info['addi']['sets_count'] = addi_sets_count.tolist()
        dataset_info['addi']['mean_channels'] = np.mean( addi_mean_channels, axis=0).tolist()

        print(' ↳ Classes count: ' + str(addi_classes_count))
        print(' ↳ Sets count: ' + str(addi_sets_count))
        print('SUCCESS: Additional dataset imported.')

        # --- MERGING INFORMATION ---
        dataset_info['num_data'] = len(list_data)
        dataset_info['classes_count'] = (main_classes_count + addi_classes_count).tolist()
        dataset_info['sets_count'] = (main_sets_count + addi_sets_count).tolist()

        mean_channels = np.append(mean_channels, addi_mean_channels, axis=0)
        dataset_info['mean_channels'] = np.mean(mean_channels, axis=0).tolist()

    print('-------------------------------------')

    # --- SAVE FILE ---
    dataset = {
        'info': dataset_info,
        'data': list_data
    }

    print('Saving...')
    with open(export_file_path, 'w') as outfile:
        json.dump(dataset, outfile)

    print('SUCCESS: Dataset information saved to ' + export_file_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_path', '-e', default='dataset.json', help='Path to save .json file containing the dataset.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    file_path = args.export_path

    if path.exists(file_path):
        sys.exit('ERROR: \'' + file_path + '\' already exists.')

    generate_json(file_path)
