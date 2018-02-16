from __future__ import division      # https://mail.python.org/pipermail/tutor/2008-March/060886.html
import importlib                    # To import other modules
import sys
import os
import time
import os.path as path
import json
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from models.faster_rcnn import data_generators
from keras.optimizers import Adam as optimizer  # TODO try other optimizers
from models.faster_rcnn import losses as losses_fn
import numpy as np
from keras.utils import generic_utils
import models.faster_rcnn.roi_helpers as roi_helpers

def train(cfg, dataset, train_imgs, test_imgs):
    """
    Train the Faster R-CNN model

    Parameters:
    cfg         -- Configuration Dictionary
    dataset     -- Dataset info Dictionary
    train_imgs  -- List of the images to use for training
    test_imgs   -- List of the images to use for testing
    """
    # Set the GPU(s) to use
    os.environ["CUDA_VISIBLE_DEVICE"] = str(cfg['gpu'])

    print('----- MODEL CREATION -----')
    # =============================
    # === DEFINE NEURAL NETWORK ===
    # =============================
    # Building the classes_count as the last implementation
    classes_count = {}
    for key in dataset['info']['class_mapping']:
        classes_count[key] = dataset['info']['classes_count'][dataset['info']['class_mapping'][key]]

    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))

    # Load module corresponding to chosen base network
    module_base_network = 'models.' + cfg['model_name'] + '.base_network.' + cfg['base_network']
    try:
        nn = importlib.import_module(module_base_network)
        print('SUCCESS: \'' + module_base_network + '\' has been imported.')
    except Exception as e:
        print(e)
        sys.exit('ERROR: Impossible to import the module \'' + module_base_network + '\'.')

    # --- LAYERS ---
    # Define shared layers
    shared_layers = nn.nn_base(img_input, trainable=True)

    # Define RPN
    num_anchors = len(cfg['anchor_boxes']['scales']) * len(cfg['anchor_boxes']['ratios'])
    rpn = nn.rpn(shared_layers, num_anchors)

    # Define Classifier
    classifier = nn.classifier(shared_layers, roi_input, cfg['rpn']['num_rois'], nb_classes=len(classes_count), trainable=True)

    # --- MODELS ---
    # Define Models
    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # This is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    # --- LOAD WEIGHTS ---
    # If a different base net weights is already define in the config file use it.
    if 'path_net_weights' in cfg.keys():
        base_net_weights = cfg['path_net_weights']
    else:
        base_net_weights = path.join('models', cfg['model_name'], 'base_network', 'weights', nn.get_weight_path())
        cfg['path_net_weights'] = cfg['model_path']

    try:
        print('Loading weights from \'' + base_net_weights + '\'...')
        model_rpn.load_weights(base_net_weights, by_name=True)
        print(' \'-> SUCCESS: RPN\'s weights loaded.')
        model_classifier.load_weights(base_net_weights, by_name=True)
        print(' \'-> SUCCESS: Classifier\'s weights loaded.')
    except Exception as e:
        print(e)
        print('ERROR: Impossible to load pretrained model weights.')
        print(' \'-> Pretrained weights can be downloaded from:  https://github.com/fchollet/keras/tree/master/keras/applications')

    # ====================
    # === PREPARE DATA ===
    # ====================
    length_img_output = nn.get_img_output_length

    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, cfg, length_img_output, K.image_dim_ordering(), mode='train')

    data_gen_test = data_generators.get_anchor_gt(test_imgs, classes_count, cfg, length_img_output, K.image_dim_ordering(), mode='val')

    # ======================
    # === INITIALIZATION ===
    # ======================

    # --- Optimizer ---
    optimizer_rpn = optimizer(lr=cfg['rpn']['optimizer_lr'])
    optimizer_classifier = optimizer(lr=cfg['classifier']['optimizer_lr'])

    # --- Compile ---
    model_rpn.compile(optimizer=optimizer_rpn,
                      loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])

    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    # --- Epochs Parameters ---
    epoch_length = int(cfg['train']['epoch_length'])
    num_epochs = int(cfg['train']['num_epochs'])

    iter_num = 0
    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf

    # --- Save Config ---
    if not path.exists(cfg['export_folder']):
        print('Saving configuration file...')
        # Create folder to save the config file
        os.makedirs(cfg['export_folder'])
        print('SUCCESS: Created ' + cfg['export_folder'])

        # Save config file
        path_config_file = path.join(cfg['export_folder'], 'config.json')
        with open(path_config_file, 'w') as outfile:
            json.dump(cfg, outfile, indent=2)

        print('SUCCESS: Configuration file saved to ' + path_config_file)
    else:
        print('\'' + cfg['export_folder'] + '\' already exists.')

    # Create file to save losses
    path_losses_file = path.join(cfg['export_folder'], 'losses.txt')
    str_losses_header = "epoch_num " + \
                 "mean_overlapping_bboxes " + \
                 "loss_rpn_cls " + \
                 "loss_rpn_regr " + \
                 "loss_class_cls " + \
                 "loss_class_regr " + \
                 "curr_loss " + \
                 "time" + "\n"

    if not path.exists(path_losses_file):
        with open(path_losses_file, "a") as text_file:
            text_file.write(str_losses_header)
        num_epoch_done = 0
    else:
        list_epoch = []
        with open(path_losses_file) as f:
            for line in f:
                # Parse line
                line_p = line.strip().split(' ')
                list_epoch.append(line_p[0])

        num_epoch_done = int(list_epoch[-1])
    # class_mapping_inv = {v: k for k, v in class_mapping.items()}

    print('-------- TRAINING --------')

    # vis = True

    for epoch_num in range(num_epochs - num_epoch_done):

        curr_epoch_num = epoch_num + num_epoch_done + 1
        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(curr_epoch_num, num_epochs))

        while True:
            try:

                if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg['verbose']:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(
                        'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap'
                              ' the ground truth boxes. Check RPN settings or keep training.')

                # DEBUG
                # print('DEBUG: Time Repartition...')
                start_time_debug = time.time()

                X, Y, img_data = next(data_gen_train)

                # DEBUG
                end_time_data_gen = time.time()
                #print(' \'-> Total Data Generator Processing time: {}'.format(end_time_data_gen - start_time_debug))

                loss_rpn = model_rpn.train_on_batch(X, Y)

                # DEBUG
                end_time_rpn_train = time.time()
                #print(' \'-> RPN Train Processing time: {}'.format(end_time_rpn_train - end_time_data_gen))

                P_rpn = model_rpn.predict_on_batch(X)

                # DEBUG
                end_time_rpn_predict = time.time()
                # print(' \'-> RPN Predict Processing time: {}'.format(end_time_rpn_predict - end_time_rpn_train))

                result = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
                                                overlap_thresh=cfg['rpn']['overlap_thresh'],
                                                max_boxes=cfg['rpn']['max_boxes'])

                # DEBUG
                end_time_rpn_to_roi = time.time()
                #print(' \'-> RPN To ROI Processing time: {}'.format(end_time_rpn_to_roi - end_time_rpn_predict))

                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(result, img_data, cfg)

                # DEBUG
                end_time_calc_iou = time.time()
                #print(' \'-> Calculation IOU Processing time: {}'.format(end_time_calc_iou - end_time_rpn_to_roi))

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if cfg['rpn']['num_rois'] > 1:
                    if len(pos_samples) < cfg['rpn']['num_rois'] // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, cfg['rpn']['num_rois'] // 2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, cfg['rpn']['num_rois'] - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, cfg['rpn']['num_rois'] - len(selected_pos_samples),
                                                                replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                # DEBUG
                end_time_selected_samples = time.time()
                #print(' \'-> Select Samples Processing time: {}'.format(end_time_selected_samples - end_time_calc_iou))

                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                # DEBUG
                end_time_classifier = time.time()
                # print(' \'-> Classifier Processing time: {}'.format(end_time_classifier - end_time_selected_samples))

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                # Update progress bar
                progbar.update(iter_num,
                               [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                ('detector_cls', np.mean(losses[:iter_num, 2])),
                                ('detector_regr', np.mean(losses[:iter_num, 3])),
                                ('t_data_gen', (end_time_data_gen - start_time_debug)),
                                ('t_rpn_train', (end_time_rpn_train - end_time_data_gen)),
                                ('t_rpn_pred', (end_time_rpn_predict - end_time_rpn_train)),
                                ('t_rpn_2_roi', (end_time_rpn_to_roi - end_time_rpn_predict)),
                                ('t_iou_calc', (end_time_calc_iou - end_time_rpn_to_roi)),
                                ('t_select_samples', (end_time_selected_samples - end_time_calc_iou)),
                                ('t_classifier', (end_time_classifier - end_time_selected_samples))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if cfg['verbose']:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    # Save losses in text file
                    str_losses = str(curr_epoch_num) + " " + \
                                 str(mean_overlapping_bboxes) + " " + \
                                 str(loss_rpn_cls) + " " + \
                                 str(loss_rpn_regr) + " " + \
                                 str(loss_class_cls) + " " + \
                                 str(loss_class_regr) + " " + \
                                 str(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr) + " " + \
                                 str(time.time() - start_time) + "\n"

                    with open(path_losses_file, "a") as text_file:
                        text_file.write(str_losses)

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if cfg['verbose']:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(cfg['model_path'])

                    # Save the model if number of iterations
                    if 'save_iter' in cfg['train']:
                        if curr_epoch_num % cfg['train']['save_iter'] == 0:
                            print('Saving Model...')
                            model_path_epoch = cfg['model_path'][:-5] + '_epoch-' + str(curr_epoch_num) + cfg['model_path'][-5:]
                            model_all.save_weights(model_path_epoch)
                            print('SUCCESS: Model saved in \'' + model_path_epoch + '\'.')
                    elif (curr_epoch_num) % 2 == 0:
                        print('Saving Model...')
                        model_path_epoch = cfg['model_path'][:-5] + '_epoch-' + str(curr_epoch_num) + cfg['model_path'][-5:]
                        model_all.save_weights(model_path_epoch)
                        print('SUCCESS: Model saved in \'' + model_path_epoch + '\'.')

                    break

            except Exception as e:
                print(e)
                # sys.exit('DEBUG: Exit when error to speed up debug.')
                print('Saving Model...')
                model_all.save_weights(cfg['model_path'])
                print('SUCCESS: Model saved in \'' + cfg['model_path'] + '\'.')
                continue
    print('SUCESS: Training complete, exiting.')
