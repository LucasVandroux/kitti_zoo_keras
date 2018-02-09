import os
import sys
import argparse
import json
import importlib

def use_model(args_):
    """
    Prepare the prediction of the bounding boxes on the image(s) given the chosen model
    """
    print('--- LOAD IMG + CONFIG ---')
    # --- LOAD IMAGE(S) ---
    img_path = args_.img_path
    accetped_img_ext = ('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')

    if os.path.isdir(img_path):
        list_img_path = [os.path.join(img_path, img_name) for img_name in os.listdir(img_path) if img_name.lower().endswith(accetped_img_ext)]

        if not list_img_path:
            sys.exit('ERROR: \'' + img_path + '\' doesn\'t contain images.')

    elif os.path.isfile(img_path) and img_path.lower().endswith(accetped_img_ext):
        list_img_path = [img_path]

    else:
        sys.exit('ERROR: \'' + img_path + '\' is neither an image neither a folder containing images.')

    print('SUCCESS: ' + str(len(list_img_path)) + ' image(s) have been found.')

    # --- LOAD CONFIG ---
    config_path = args_.config_path
    try:
        print('Importing configuration file...')
        cfg = json.load(open(config_path))
        print('SUCCESS: Configuration file imported.')
    except ValueError:
        sys.exit('ERROR: Decoding ' + config_path + ' has failed.')

    # Load module corresponding to model in the configuration file
    module_model = 'models.' + cfg['model_name'] + '.predict'
    try:
        model = importlib.import_module(module_model)
        print('SUCCESS: \'' + module_model + '\' has been imported.')
    except Exception as e:
        print(e)
        sys.exit('ERROR: Impossible to import the module \'' + module_model + '\'.')

    # Set global values for predictions
    cfg['data_augmentation']['use_horizontal_flips'] = False
    cfg['data_augmentation']['use_vertical_flips'] = False
    cfg['data_augmentation']['rot_90'] = False

    cfg['export_folder'] = args_.export_folder

    # Add 'Background' class if doesn't exist already
    if 'Background' not in cfg['class_mapping']:
        cfg['class_mapping']['Background'] = len(cfg['class_mapping'])

    model.predict(list_img_path, cfg)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', help='Path to the configuration file.')
    parser.add_argument('--img_path', '-p', help='Path to images\' folder or file')
    parser.add_argument('--export_folder', '-e', help='Path to folder where to export the image(s)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    use_model(args)
