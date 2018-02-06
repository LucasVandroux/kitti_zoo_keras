"""
this code will train on kitti data set
"""
# from __future__ import division      # https://mail.python.org/pipermail/tutor/2008-March/060886.html
import argparse                      # Read console arguments
import json                          # Use to import config file
import os
import os.path as path               # To create file path
import sys
import random
import models.faster_rcnn.train as faster_rcnn


def train(args_):
    """
    Initiate the training of a model based on a config file.

    Parameters:
    model_name     Name of the model/subfolder to train
    config         Config file (.json) in the model's subfolder
    dataset        File (.json) containing the description of the dataset
    """
    # TODO in config file allow the ability to remove Keras warnings

    # ===================================
    # === LOAD & SAVE NEW CONFIG FILE ===
    # ===================================
    # The chosen model should have a subfolder containing all the file related to it in the models folder
    # Plus the .json file should be in the model's subfolder
    config_path = path.join('models', args_.model_name, args_.config)

    list_models = [n for n in os.listdir('models') if path.isdir(path.join('models', n))]

    # Check if the subfolder for the model exist in the folder ./models/
    if not args_.model_name in list_models:
        sys.exit('ERROR: '+ args_.model_name +' doesn\'t correspond to any subfolders in the \"./models/\" folder.')
    # Check if the config file exists
    elif not path.isfile(config_path):
        sys.exit('ERROR: Can\'t find ' + config_path)

    # Import config file from the model subfolder
    try:
        cfg = json.load(open(config_path))
    except ValueError as ve:
        print(ve)
        sys.exit('ERROR: Decoding ' + config_path + ' has failed.')

    # Check if folder to save model exists or create it
    trained_model_folder_path = path.join('models', cfg['model_name'], 'trained_model')
    if not path.isdir(trained_model_folder_path):
        os.makedirs(trained_model_folder_path)
        print('SUCCESS: Created ' + trained_model_folder_path)

    # Test if the folder to save the new model exists already
    export_folder_path = path.join(trained_model_folder_path, (cfg['model_name'] + '-' + cfg['base_network'] + '-' + cfg['folder_descriptor']))
    if path.isdir(export_folder_path):
        sys.exit('ERROR: \'' + export_folder_path + '\' already exists, please change \'folder_descriptor\' in config file.')

    # Add path to export the weights and config file of the model
    cfg['export_folder'] = path.abspath(export_folder_path)
    cfg['model_path'] = path.join(cfg['export_folder'], (cfg['model_name'] + '-' + cfg['base_network'] + '.hdf5'))

    # TODO only print useful information at this moment
    # print(json.dumps(cfg, indent=2))

    # ========================
    # === LOAD THE DATASET ===
    # ========================
    # Import dataset information
    dataset_json_path = path.abspath(args_.dataset)

    try:
        print('Importing dataset information...')
        dataset = json.load(open(dataset_json_path))
        print('SUCCESS: Dataset imported.')
    except ValueError:
        sys.exit('ERROR: Decoding ' + dataset_json_path + ' has failed.')

    cfg['dataset_path'] = dataset_json_path

    all_images = dataset['data']
    classes_count = dataset['info']['classes_count']
    class_mapping = dataset['info']['class_mapping']
    set_mapping = dataset['info']['set_mapping']

    # Add Background class
    if 'bg' not in class_mapping or 'Background' not in class_mapping:
        class_mapping['Background'] = len(class_mapping)
        classes_count.append(0)

    # Add class mapping to the configuration file
    cfg['class_mapping'] = class_mapping
    cfg['classes_count'] = classes_count
    cfg['img']['channel_mean'] = dataset['info']['mean_channels']

    # Shuffling the data
    random.shuffle(all_images)
    num_imgs = len(all_images)
    train_imgs = [s for s in all_images if s['set'] == set_mapping['train'] or s['set'] == set_mapping['dev']]
    test_imgs = [s for s in all_images if s['set'] == set_mapping['test']]

    print(' ↳ Classes count: ' + str(classes_count))
    print(' ↳ Num train+dev: ' + str(len(train_imgs)))
    print(' ↳ Num test: ' + str(len(test_imgs)))

    # TODO add the possibilty to train different models

    # Train Faster R-CNN model
    faster_rcnn.train(cfg, dataset, train_imgs, test_imgs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='Name of the folder containing the model.')
    parser.add_argument('--config', '-c', default='config.json' ,help='Configuration file in the model folder.')
    parser.add_argument('--dataset', '-d', default='dataset.json' ,help='Dataset file in the datasets folder.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
