from utils.mean_average_precision.detection_map import DetectionMAP
from utils.mean_average_precision.show_frame import show_frame
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import os.path as path
import json
import imageio

def import_pred(file_path, class_mapping, im_shape):
    with open(file_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        list_row = [row for row in csvreader]
        row_count = len(list_row)

        pred_bb = np.zeros((row_count, 4))
        pred_cls = np.zeros(row_count)
        pred_conf = np.zeros(row_count)

        for idx, row in enumerate(list_row):
            pred_cls[idx] = int(class_mapping[row[0]][1])

            pred_x1 = float(row[1]) / im_shape[1]
            pred_y1 = float(row[2]) / im_shape[0]
            pred_x2 = float(row[3]) / im_shape[1]
            pred_y2 = float(row[4]) / im_shape[0]
            pred_bb[idx,:] = np.array([pred_x1, pred_y1, pred_x2, pred_y2])

            pred_conf[idx] = float(row[5])

    return pred_bb, pred_cls, pred_conf



class_mapping = {
      "Person_sitting": [4, 3],
      "Cyclist": [5, 5],
      "Pedestrian": [3, 3],
      "Van": [1, 0],
      "Truck": [2, 2],
      "Misc": [7, 7],
      "DontCare": [8, 8],
      "Car": [0, 0],
      "Tram": [6, 6]
    }
# TODO detecting a DontCare doesn't influence the mAP
# TODO convert as not all the classes are the same (Person_sitting and Pedestrian are the same)
# TODO save images
# TODO get the variables in arguments for the main function

if __name__ == '__main__':
    is_show_frame = False
    path_pred_folder = '/Users/lucas/Desktop/test_mAP'
    path_database = '/Users/lucas/Documents/GitHub/kitti_zoo_keras/datasets/dummy_dataset.json'
    path_savefig = '/Users/lucas/Desktop/pr_curve_example.png'

    list_frames = []
    n_class = int(len(set([item[1] for item in class_mapping.values()])))
    class_mapping_int = {str(item[0]): int(item[1]) for item in class_mapping.values()}

    print(n_class)

    # Import Database in Json
    database_info = json.load(open(path_database))

    # Extract .csv file from path_pred_folder
    list_pred = [filename[:-4] for filename in os.listdir(path_pred_folder) if filename.endswith('.csv')]

    # Extract file from dataset_info
    dict_gt = {}
    for img_info in database_info['data']:
        filename = path.basename(img_info['filepath'])[:-4]
        if filename in list_pred:
            dict_gt[filename] = { 'bboxes': img_info['bboxes'], 'filepath': img_info['filepath'], 'im_shape': [img_info['height'], img_info['width'], 3]}

    for filename in list_pred:
        csv_name = filename + '.csv'
        im_shape = dict_gt[filename]['im_shape']

        pred_bb, pred_cls, pred_conf = import_pred(path.join(path_pred_folder, csv_name), class_mapping, im_shape)

        gts = dict_gt[filename]['bboxes']
        gt_bb = np.zeros((len(gts), 4))
        gt_cls = np.zeros(len(gts))

        for idx, gt in enumerate(gts):
            gt_cls[idx] = class_mapping_int[str(gt['class'])]
            gt_bb[idx, :] = np.array([gt['x1']/im_shape[1], gt['y1']/im_shape[0], gt['x2']/im_shape[1], gt['y2']/im_shape[0]])

        list_frames.append((pred_bb, pred_cls, pred_conf, gt_bb, gt_cls, dict_gt[filename]['filepath']))

    mAP = DetectionMAP(n_class)
    for i, frame in enumerate(list_frames):
        print("Evaluate frame {}".format(i))
        background_im = imageio.imread(frame[5]).astype('uint8')
        # background_im = np.zeros((background_im.shape))
        # show_frame(frame[0], frame[1], frame[2], frame[3], frame[4], background=background_im, show_confidence=True)
        if is_show_frame:
            show_frame(*frame[:5], background=background_im, show_confidence=True)
        mAP.evaluate(*frame[:5])

    mAP.plot()
    plt.show()
    plt.savefig(path_savefig)
