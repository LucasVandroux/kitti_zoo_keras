from __future__ import division
import argparse
import os
import os.path as path
import cv2
import numpy as np
import time
import importlib
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from models.faster_rcnn import roi_helpers
from utils.visualize import draw_boxes_and_label_on_image_cv2

#import models.faster_rcnn.resnet as nn

def format_img_size(img, cfg):
    """ formats the image size based on config """
    img_min_side = float(cfg['img']['min_side'])
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, cfg):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= cfg['img']['channel_mean'][0]
    img[:, :, 1] -= cfg['img']['channel_mean'][1]
    img[:, :, 2] -= cfg['img']['channel_mean'][2]
    img /= cfg['img']['scaling_factor']
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, cfg):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, cfg)
    img = format_img_channels(img, cfg)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


def predict_single_image(img_path, model_rpn, model_classifier_only, cfg, class_mapping):

    # Loading image
    img = cv2.imread(img_path)
    if img is None:
        print(' ↳ ERROR: Impossible to read \'' + img_path + '\'.')

    else:
        start_time = time.time()

        X, ratio = format_img(img, cfg)
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        # this is result contains all boxes, which is [x1, y1, x2, y2]
        result = roi_helpers.rpn_to_roi(Y1, Y2, cfg, K.image_dim_ordering(), overlap_thresh=cfg['rpn']['overlap_thresh'])

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        result[:, 2] -= result[:, 0]
        result[:, 3] -= result[:, 1]

        # Spatial Pyramid Pooling to the proposed regions
        boxes = dict()
        for jk in range(result.shape[0] // cfg['rpn']['num_rois'] + 1):
            rois = np.expand_dims(result[cfg['rpn']['num_rois'] * jk:cfg['rpn']['num_rois'] * (jk + 1), :], axis=0)
            if rois.shape[1] == 0:
                break
            if jk == result.shape[0] // cfg['rpn']['num_rois']:
                # pad R
                curr_shape = rois.shape
                target_shape = (curr_shape[0], cfg['rpn']['num_rois'], curr_shape[2])
                rois_padded = np.zeros(target_shape).astype(rois.dtype)
                rois_padded[:, :curr_shape[1], :] = rois
                rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
                rois = rois_padded

            [p_cls, p_regr] = model_classifier_only.predict([F, rois])

            for ii in range(p_cls.shape[1]):
                if np.max(p_cls[0, ii, :]) < cfg['classifier']['bbox_threshold'] or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                    continue

                cls_num = np.argmax(p_cls[0, ii, :])
                if cls_num not in boxes.keys():
                    boxes[cls_num] = []
                (x, y, w, h) = rois[0, ii, :]
                try:
                    (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= cfg['classifier']['regr_std'][0]
                    ty /= cfg['classifier']['regr_std'][1]
                    tw /= cfg['classifier']['regr_std'][2]
                    th /= cfg['classifier']['regr_std'][3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except Exception as e:
                    print(e)
                    pass
                boxes[cls_num].append(
                    [cfg['rpn']['stride'] * x, cfg['rpn']['stride'] * y, cfg['rpn']['stride'] * (x + w), cfg['rpn']['stride'] * (y + h),
                     np.max(p_cls[0, ii, :])])

        # add some nms to reduce many boxes
        for cls_num, box in boxes.items():
            boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=cfg['nms']['overlap_thresh'])
            boxes[cls_num] = boxes_nms
            print(class_mapping[cls_num] + ":")

            for b in boxes_nms:
                b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
                print('{} prob: {}'.format(b[0: 4], b[-1]))

        processing_time = time.time() - start_time
        img = draw_boxes_and_label_on_image_cv2(img, class_mapping, boxes)

        print(' ↳ Processing time: {}'.format(processing_time))

        # cv2.imshow('image', img)

        result_path = path.join(cfg['export_folder'], path.basename(img_path))
        cv2.imwrite(result_path, img)
        print(' ↳ SUCCESS: Output image saved as \'' + result_path + '\'.')

def predict(list_img_path, cfg):
    print('------ LOAD MODEL -------')
    # Inverse the class mapping
    class_mapping = {v: k for k, v in cfg['class_mapping'].items()}

    # --- INPUTS ---
    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(cfg['rpn']['num_rois'], 4))
    feature_map_input = Input(shape=(None, None, 1024))

    # --- MODEL ---
    # Load module corresponding to chosen base network
    module_base_network = 'models.' + cfg['model_name'] + '.base_network.' + cfg['base_network']
    try:
        nn = importlib.import_module(module_base_network)
        print('SUCCESS: \'' + module_base_network + '\' has been imported.')
    except Exception as e:
        print(e)
        sys.exit('ERROR: Impossible to import the module \'' + module_base_network + '\'.')

    shared_layers = nn.nn_base(img_input, trainable=True)

    # Define RPN
    num_anchors = len(cfg['anchor_boxes']['scales']) * len(cfg['anchor_boxes']['ratios'])
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    # Define Classifier
    classifier = nn.classifier(feature_map_input, roi_input, cfg['rpn']['num_rois'], nb_classes=len(class_mapping),
                               trainable=True)

    # Create Models
    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    # --- LOAD WEIGHTS ---
    try:
        print('Loading weights from \'' + cfg['model_path'] + '\'...')
        model_rpn.load_weights(cfg['model_path'], by_name=True)
        print(' ↳ SUCCESS: RPN\'s weights loaded.')
        model_classifier.load_weights(cfg['model_path'], by_name=True)
        print(' ↳ SUCCESS: Classifier\'s weights loaded.')
    except Exception as e:
        print(e)
        sys.exit('ERROR: Impossible to load trained model weights.')

    # --- COMPILE ---
    model_rpn.compile(optimizer='sgd', loss='mse')
    print('SUCCESS: RPN model has been compiled.')
    model_classifier.compile(optimizer='sgd', loss='mse')
    print('SUCCESS: Classifier model has been compiled.')

    print('----- PREDICTION(S) -----')

    for img_path in list_img_path:
        print('Processing: ' + img_path)
        predict_single_image(img_path, model_rpn, model_classifier_only, cfg, class_mapping)
