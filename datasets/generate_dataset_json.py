"""
This will load all the information from the labels file of the main dataset and the additional dataset in a global json file.
"""
from os import listdir, readlink
import os.path as path      # To create file path
import sys
import json


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

    # Path for the main dataset (also works with symbolic links but not MacOS aliases!)
    PATH_MAIN_IMAGES = path.join('main','images')
    PATH_MAIN_LABELS = path.join('main','labels')

    # Path for the additional dataset (also works with symbolic links but not MacOS aliases!)
    PATH_ADDITIONAL_IMAGES = path.join('additional','images')
    PATH_ADDITIONAL_LABELS = path.join('additional','labels')

    # Initialize dictionary to hold info about the dataset
    dataset_info = {}

    # --- MAIN DATASET ---
    if not path.exists(PATH_MAIN_IMAGES):
        sys.exit('ERROR: Couldn\'t find ' + PATH_MAIN_IMAGES)
    elif path.islink(PATH_MAIN_IMAGES):
        PATH_MAIN_IMAGES = readlink(PATH_MAIN_IMAGES)

    set_main_images_names = set([f[:-len(IMG_EXT)] for f in listdir(PATH_MAIN_IMAGES) if f.endswith(IMG_EXT)])

    if not path.exists(PATH_MAIN_LABELS):
        sys.exit('ERROR: Couldn\'t find ' + PATH_MAIN_LABELS)
    elif path.islink(PATH_MAIN_LABELS):
        PATH_MAIN_LABELS = readlink(PATH_MAIN_LABELS)

    set_main_labels_names = set([f[:-len(LBL_EXT)] for f in listdir(PATH_MAIN_LABELS) if f.endswith(LBL_EXT)])

    # Remove the images without a label or the contrary
    list_valid_data_names = list(set_main_images_names & set_main_labels_names)

    # Add data to
    dataset_info['num_main_data'] = len(list_valid_data_names)
    dataset_info['path_main_images'] = PATH_MAIN_IMAGES
    dataset_info['path_main_labels'] = PATH_MAIN_LABELS
    dataset_info['image_extension'] = IMG_EXT
    dataset_info['label_extension'] = LBL_EXT

    print(json.dumps(dataset_info, indent=2))

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
