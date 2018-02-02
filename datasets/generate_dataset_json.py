"""
This will load all the information from the labels file of the main dataset and the additional dataset in a global json file.
"""
from os import listdir, readlink
import os.path as path      # To create file path
import sys
import json
from tqdm import tqdm, trange       # Use to make the progress bar
import numpy as np

def test_path(path_to_test):
    """
    Test if a path exists and convert a symbolic link in a real path.

    Parameters:
    path_to_test      -- path to get(can be a symbolic link but no aliases)

    Returns:
    path              -- working path
    """
    if not path.exists(path_to_test):
        sys.exit('ERROR: Couldn\'t find ' + path_to_test)
    elif path.islink(path_to_test):
        return readlink(path_to_test)
    return path_to_test

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
    set_files = set([f[:-len(file_extension)] for f in listdir(path) if f.endswith(file_extension)])

    print('   ↳ ' + str(len(set_files)) + ' files with extension \'' + file_extension + '\' in ' + path)

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

def import_data(list_names, path_labels, name_dataset, dataset_info):
    """
    Import all the data in a list

    Parameters:
    list_names      -- List containing all the name of the data
    path_labels     -- Path to the folder containing the labels' files
    name_dataset    -- Name of the dataset (e.g. 'main', 'plus', ...)
    dataset_info    -- Dictionary containing the info on the dataset

    Returns:
    list_data       -- List of dictionary containing all the info on data and bboxes
    num_class       -- Repartition of the classes in the dataset
    num_set         -- Repartition of the sets in the dataset

    """
    print('Importing labels from ' + str(len(list_names)) + ' files:')

    # Get useful variables from dataset_info
    img_ext = dataset_info['image_extension']
    lbl_ext = dataset_info['label_extension']
    class_mapping = dataset_info['class_mapping']
    set_repartition = dataset_info['set_repartition'][name_dataset]

    # Initialize variables to store values to return
    list_data = []
    num_class = np.zeros((len(class_mapping),), dtype=int)
    num_set = np.zeros((3,), dtype=int)

    for idx in trange(len(list_names)):
        img_dict = {'dataset': name_dataset}

        img_dict['filename'] = list_names[idx] + img_ext

        # Imoprt the labels relative to the current image
        label_path = path.join(path_labels, (list_names[idx] + lbl_ext))
        num_classes, bboxes = import_bboxes(label_path, class_mapping)

        img_dict['bboxes'] = bboxes

        img_dict['num_classes'] = num_classes
        img_dict['set'] = np.random.choice(3, 1, p=set_repartition)[0]

        # Tracking repartition of classes and sets
        num_class += num_classes
        num_set[img_dict['set']] += 1

        # Add data to the global list
        list_data.append(img_dict)

    return list_data, num_class, num_set

def generate_json():
    """
    convert kitti data into a single txt file, with this format:
    Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01

    type, truncated, occluded, alpha,
    :param img_dir_:
    :param label_dir_:
    :return:
    """

    # Extension for images files and labels
    IMG_EXT = '.png'
    LBL_EXT = '.txt'

    # Use to convert the types to number to decrease the size of the file
    CLASS_MAPPING = {
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

    # Repartition of the data for each dataset [train, dev, test]
    SET_REPARTITION = {
        'main': [1/3, 1/3, 1/3],
        'plus': [1, 0, 0]
        }

    # Path for the main dataset (also works with symbolic links but not MacOS aliases!)
    PATH_MAIN_IMAGES = path.join('main','images')
    PATH_MAIN_LABELS = path.join('main','labels')

    # Path for the additional dataset (also works with symbolic links but not MacOS aliases!)
    PATH_ADDITIONAL_IMAGES = path.join('additional','images')
    PATH_ADDITIONAL_LABELS = path.join('additional','labels')

    # Initialize dictionary to hold info about the dataset
    dataset_info = {}

    # Initialize list to contains all the data
    list_data = []

    # Add global variable to dataset_info
    dataset_info['class_mapping'] = CLASS_MAPPING
    dataset_info['set_mapping'] = {0: 'train', 1: 'dev', 2: 'test'}
    dataset_info['set_repartition'] = SET_REPARTITION
    dataset_info['image_extension'] = IMG_EXT
    dataset_info['label_extension'] = LBL_EXT

    # --- MAIN DATASET ---
    print('Importing Main Dataset...')
    # Get sets of images and labels from the folders
    path_main_img, set_main_img = get_files(PATH_MAIN_IMAGES, IMG_EXT)
    path_main_lbl, set_main_lbl = get_files(PATH_MAIN_LABELS, LBL_EXT)

    # Remove the images without a label or the contrary
    list_main_names = sorted(list(set_main_img & set_main_lbl))

    # Add data to info
    dataset_info['num_main_data'] = len(list_main_names)
    dataset_info['path_main_images'] = path_main_img
    dataset_info['path_main_labels'] = path_main_lbl

    list_main_data, num_main_class, num_main_set = import_data(list_main_names, path_main_lbl, 'main', dataset_info)

    dataset_info['num_main_class'] = num_main_class
    dataset_info['num_main_set'] = num_main_set

    print(num_main_class)
    print(num_main_set)
    print(list_main_data[0:3])

    # Check if there is an additional dataset to add

    # TODO count the number of each

    if not os.path.exists(label_dir_):
        print('label dir: {} doest not exist'.format(label_dir_))
        exit(0)
    all_label_files = [i for i in os.listdir(label_dir_) if i.endswith('.txt')]
    print('got {} label files.'.format(len(all_label_files)))

    all_img_lables = []

    target_file = open('kitti_simple_label.txt', 'w')
    for label_file_name in all_label_files:
        label_file = os.path.join(label_dir_, label_file_name)

        with open(label_file, 'r') as f:
            for l in f.readlines():
                class_name, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _ = l.strip().split(' ')
                target_file.write('{},{},{},{},{},{}\n'.format(
                    os.path.join(img_dir_, label_file_name.replace('txt', 'png')),
                                                               x1, y1, x2, y2, class_name))

    target_file.close()
    print('convert finished.')


if __name__ == '__main__':
    generate_json()
