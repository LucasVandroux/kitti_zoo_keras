"""
This will load all the information from the labels file of the main dataset and the additional dataset in a global json file.
"""
from os import listdir, readlink
import os.path as path      # To create file path
import sys
import json


def generate(img_dir_, label_dir_):
    """
    convert kitti data into a single txt file, with this format:
    Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01

    type, truncated, occluded, alpha,
    :param img_dir_:
    :param label_dir_:
    :return:
    """

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

    # Symlinks path for the main dataset
    PATH_MAIN_IMAGES = path.join('main','images')
    PATH_MAIN_LABELS = path.join('main','labels')

    # Symlinks path for the additional dataset (only for training)
    PATH_ADDITIONAL_IMAGES = path.join('additional','images')
    PATH_ADDITIONAL_LABELS = path.join('additional','labels')

    # --- MAIN DATASET ---
    if not path.exists(PATH_MAIN_IMAGES):
        sys.exit('ERROR: Couldn\'t find ' + PATH_MAIN_IMAGES)
    elif path.islink(PATH_MAIN_IMAGES):
        PATH_MAIN_IMAGES = readlink(SYMLINK_MAIN_IMAGES)

    list_main_images = [f for f in listdir(PATH_MAIN_IMAGES) if f.endswith('.png')]

    print()

    try:
        path_main_dataset_images = os.readlink(SYMLINK_MAIN_IMAGES)
    except ValueError:
        sys.exit('ERROR: Main dataset images foler impossible to recover from sysmlink ' + SYMLINK_MAIN_IMAGES)

    try:
        path_main_dataset_images = os.readlink(SYMLINK_MAIN_IMAGES)
    except ValueError:
        sys.exit('ERROR: Main dataset images foler impossible to recover from sysmlink ' + SYMLINK_MAIN_IMAGES)

    path_main_dataset_labels = os.readlink(join('main','labels'))

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
    img_dir = sys.argv[1]
    label_dir = sys.argv[2]
    generate(img_dir, label_dir)
