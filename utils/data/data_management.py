import pandas as pd    # Create and manage Dataframes
import cv2
import numpy as np

def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and not found_bg:
                    print('Found class name with special name bg. Will be treated as a'
                          ' background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)),
                 'y2': int(float(y2))})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping

def aggregate_data(list_data_folder , dataset_export_path, save_to_disk = True):
"""
Create pandas dataframe containing all the information about the dataset and save it to the disk.

Parameters:
list_data_folder        [list]    list of all the folder containing the data and its labels:
                                  [(path_data, path_label, is_val, is_test), ()]
dataset_export_path     [string]  path where to save the pandas dataframe
save_to_disk            [bool]    control if dataframe is saved or not to the save_to_disk

Returns:
dataset_panda           [pandas]  pandas dataframe containing all the dataset
"""


def import_labels(path_label_file):
"""
Import the labels from the .txt file (KITTI style) into a list of dictionaries

Parameters:
path_label_file         [string]  path to the .txt file containing the labels

Returns:
labels                  [list]    list of dictionaries describing the labels
"""
    labels = []

    # Import
    with open(label_path) as f:
        for line in f:
            # Parse line
            line_parsed = line.strip().split(' ')

            # Fill the dictionary with the values
            label_dict = {}
            label_dict['type']        = line_parsed[0]
            label_dict['truncated']   = float(line_parsed[1])
            label_dict['occluded']    = int(line_parsed[2])
            label_dict['alpha']       = float(line_parsed[3])
            label_dict['bbox']        = {'x_min': float(line_parsed[4]),
                                          'y_min': float(line_parsed[5]),
                                          'x_max': float(line_parsed[6]),
                                          'y_max': float(line_parsed[7])}
            label_dict['3D_dim']      = {'height': float(line_parsed[8]),
                                          'width' : float(line_parsed[9]),
                                          'length': float(line_parsed[10])}
            label_dict['3D_loc']      = {'x': float(line_parsed[11]),
                                          'y': float(line_parsed[12]),
                                          'z': float(line_parsed[13])}
            label_dict['rotation_y']  = float(line_parsed[14])

            # Add Score [optional in the file]
            if len(line_parsed) > 15:
                label_dict['score']       = float(line_parsed[15])

            # Append the dictionary to the list of object in the picture
            labels.append(label_dict)

    return labels
