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
import models.faster_rcnn as faster_rcnn


def train(args_):
    """
    Initiate the training of a model based on a config file.

    Parameters:
    model_name     Name of the model/subfolder to train
    config         Config file (.json) in the model's subfolder
    dataset        File (.json) containing the description of the dataset

    Returns:
    TODO
    """
# 1. set the right model to train
# 2. Load the default config file
# 3. Load the files
#    ↳ If don't exist raise error to ask user to do it
# 4. Print information about model which is going to be trained
# 5. Launch the training
#
#  TODO in config file allow the ability to remove Keras warnings

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
    except ValueError:
        sys.exit('ERROR: Decoding ' + config_path + ' has failed.')

    # Check if folder to save model exists or create it
    trained_model_folder_path = path.join('models', cfg['model_name'], 'trained_model')
    if not path.isdir(trained_model_folder_path):
        os.makedirs(trained_model_folder_path)
        print('SUCCESS: Created ' + trained_model_folder_path)

    # Create the folder to save the new model
    export_folder_path = path.join(trained_model_folder_path, (cfg['model_name'] + '-' + cfg['base_network'] + '-' + cfg['folder_descriptor']))
    if path.isdir(export_folder_path):
        sys.exit('ERROR: \'' + export_folder_path + '\' already exists, please change \'folder_descriptor\' in config file.')
    else:
        os.makedirs(export_folder_path)
        print('SUCCESS: Created ' + export_folder_path)

    # Add path to export the weights and config file of the model
    cfg['export_folder'] = path.abspath(export_folder_path)

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

    # Add class mapping to the configuration file
    cfg['class_mapping'] = class_mapping

    # Shuffling the data
    # DEBUG: set the seed for reproductible resutls
    random.seed(1)

    random.shuffle(all_images)
    num_imgs = len(all_images)
    train_imgs = [s for s in all_images if s['set'] == set_mapping['train'] or s['set'] == set_mapping['dev']]
    test_imgs = [s for s in all_images if s['set'] == set_mapping['test']]

    print(' ↳ Classes count: ' + str(classes_count))
    print(' ↳ Num train+dev: ' + str(len(train_imgs)))
    print(' ↳ Num test: ' + str(len(test_imgs)))

    # TODO save the config file in the export folder

    # Train Faster R-CNN model
    faster_rcnn.train(cfg, dataset, train_imgs, test_imgs)

    # all_images, classes_count, class_mapping = get_data(cfg.simple_label_file)

    # if 'bg' not in classes_count:
    #     classes_count['bg'] = 0
    #     class_mapping['bg'] = len(class_mapping)



    # with open(cfg.config_save_file, 'wb') as config_f:
    #     pickle.dump(cfg, config_f)
    #     print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
    #         cfg.config_save_file))

    # inv_map = {v: k for k, v in class_mapping.items()}

    # print('Training images per class:')
    # pprint.pprint(classes_count)
    # print('Num classes (including bg) = {}'.format(len(classes_count)))
    # random.shuffle(all_images)
    # num_imgs = len(all_images)
    # train_imgs = [s for s in all_images if s['imageset'] == 'trainval']
    # val_imgs = [s for s in all_images if s['imageset'] == 'test']
    #
    # print('Num train samples {}'.format(len(train_imgs)))
    # print('Num val samples {}'.format(len(val_imgs)))

    # # TODO
    # # Need to import right package so check if package imported correspond to config file and raise error if not
    # # cfg.base_net_weights = os.path.join('./model/', nn.get_weight_path())
    #
    # data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, cfg, nn.get_img_output_length,
    #                                                K.image_dim_ordering(), mode='train')
    # data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, cfg, nn.get_img_output_length,
    #                                              K.image_dim_ordering(), mode='val')
    #
    # if K.image_dim_ordering() == 'th':
    #     input_shape_img = (3, None, None)
    # else:
    #     input_shape_img = (None, None, 3)
    #
    # img_input = Input(shape=input_shape_img)
    # roi_input = Input(shape=(None, 4))
    #
    # # define the base network (resnet here, can be VGG, Inception, etc)
    # shared_layers = nn.nn_base(img_input, trainable=True)
    #
    # # define the RPN, built on the base layers
    # num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    # rpn = nn.rpn(shared_layers, num_anchors)
    #
    # classifier = nn.classifier(shared_layers, roi_input, cfg.num_rois, nb_classes=len(classes_count), trainable=True)
    #
    # model_rpn = Model(img_input, rpn[:2])
    # model_classifier = Model([img_input, roi_input], classifier)
    #
    # # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    # model_all = Model([img_input, roi_input], rpn[:2] + classifier)
    #
    # try:
    #     print('loading weights from {}'.format(cfg.base_net_weights))
    #     model_rpn.load_weights(cfg.model_path, by_name=True)
    #     model_classifier.load_weights(cfg.model_path, by_name=True)
    # except Exception as e:
    #     print(e)
    #     print('Could not load pretrained model weights. Weights can be found in the keras application folder '
    #           'https://github.com/fchollet/keras/tree/master/keras/applications')
    #
    # optimizer = Adam(lr=1e-5)
    # optimizer_classifier = Adam(lr=1e-5)
    # model_rpn.compile(optimizer=optimizer,
    #                   loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
    # model_classifier.compile(optimizer=optimizer_classifier,
    #                          loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(classes_count) - 1)],
    #                          metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    # model_all.compile(optimizer='sgd', loss='mae')
    #
    # epoch_length = 1000
    # num_epochs = int(cfg.num_epochs)
    # iter_num = 0
    #
    # losses = np.zeros((epoch_length, 5))
    # rpn_accuracy_rpn_monitor = []
    # rpn_accuracy_for_epoch = []
    # start_time = time.time()
    #
    # best_loss = np.Inf
    #
    # class_mapping_inv = {v: k for k, v in class_mapping.items()}
    # print('Starting training')
    #
    # vis = True
    #
    # for epoch_num in range(num_epochs):
    #
    #     progbar = generic_utils.Progbar(epoch_length)
    #     print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
    #
    #     while True:
    #         try:
    #
    #             if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
    #                 mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
    #                 rpn_accuracy_rpn_monitor = []
    #                 print(
    #                     'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
    #                         mean_overlapping_bboxes, epoch_length))
    #                 if mean_overlapping_bboxes == 0:
    #                     print('RPN is not producing bounding boxes that overlap'
    #                           ' the ground truth boxes. Check RPN settings or keep training.')
    #
    #             X, Y, img_data = next(data_gen_train)
    #
    #             loss_rpn = model_rpn.train_on_batch(X, Y)
    #
    #             P_rpn = model_rpn.predict_on_batch(X)
    #
    #             result = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
    #                                             overlap_thresh=0.7,
    #                                             max_boxes=300)
    #             # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
    #             X2, Y1, Y2, IouS = roi_helpers.calc_iou(result, img_data, cfg, class_mapping)
    #
    #             if X2 is None:
    #                 rpn_accuracy_rpn_monitor.append(0)
    #                 rpn_accuracy_for_epoch.append(0)
    #                 continue
    #
    #             neg_samples = np.where(Y1[0, :, -1] == 1)
    #             pos_samples = np.where(Y1[0, :, -1] == 0)
    #
    #             if len(neg_samples) > 0:
    #                 neg_samples = neg_samples[0]
    #             else:
    #                 neg_samples = []
    #
    #             if len(pos_samples) > 0:
    #                 pos_samples = pos_samples[0]
    #             else:
    #                 pos_samples = []
    #
    #             rpn_accuracy_rpn_monitor.append(len(pos_samples))
    #             rpn_accuracy_for_epoch.append((len(pos_samples)))
    #
    #             if cfg.num_rois > 1:
    #                 if len(pos_samples) < cfg.num_rois // 2:
    #                     selected_pos_samples = pos_samples.tolist()
    #                 else:
    #                     selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois // 2, replace=False).tolist()
    #                 try:
    #                     selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
    #                                                             replace=False).tolist()
    #                 except:
    #                     selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
    #                                                             replace=True).tolist()
    #
    #                 sel_samples = selected_pos_samples + selected_neg_samples
    #             else:
    #                 # in the extreme case where num_rois = 1, we pick a random pos or neg sample
    #                 selected_pos_samples = pos_samples.tolist()
    #                 selected_neg_samples = neg_samples.tolist()
    #                 if np.random.randint(0, 2):
    #                     sel_samples = random.choice(neg_samples)
    #                 else:
    #                     sel_samples = random.choice(pos_samples)
    #
    #             loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
    #                                                          [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
    #
    #             losses[iter_num, 0] = loss_rpn[1]
    #             losses[iter_num, 1] = loss_rpn[2]
    #
    #             losses[iter_num, 2] = loss_class[1]
    #             losses[iter_num, 3] = loss_class[2]
    #             losses[iter_num, 4] = loss_class[3]
    #
    #             iter_num += 1
    #
    #             progbar.update(iter_num,
    #                            [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
    #                             ('detector_cls', np.mean(losses[:iter_num, 2])),
    #                             ('detector_regr', np.mean(losses[:iter_num, 3]))])
    #
    #             if iter_num == epoch_length:
    #                 loss_rpn_cls = np.mean(losses[:, 0])
    #                 loss_rpn_regr = np.mean(losses[:, 1])
    #                 loss_class_cls = np.mean(losses[:, 2])
    #                 loss_class_regr = np.mean(losses[:, 3])
    #                 class_acc = np.mean(losses[:, 4])
    #
    #                 mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
    #                 rpn_accuracy_for_epoch = []
    #
    #                 if cfg.verbose:
    #                     print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
    #                         mean_overlapping_bboxes))
    #                     print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
    #                     print('Loss RPN classifier: {}'.format(loss_rpn_cls))
    #                     print('Loss RPN regression: {}'.format(loss_rpn_regr))
    #                     print('Loss Detector classifier: {}'.format(loss_class_cls))
    #                     print('Loss Detector regression: {}'.format(loss_class_regr))
    #                     print('Elapsed time: {}'.format(time.time() - start_time))
    #
    #                 curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
    #                 iter_num = 0
    #                 start_time = time.time()
    #
    #                 if curr_loss < best_loss:
    #                     if cfg.verbose:
    #                         print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
    #                     best_loss = curr_loss
    #                     model_all.save_weights(cfg.model_path)
    #
    #                 break
    #
    #         except Exception as e:
    #             print('Exception: {}'.format(e))
    #             # save model
    #             model_all.save_weights(cfg.model_path)
    #             continue
    # print('Training complete, exiting.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='Name of the folder containing the model.')
    parser.add_argument('--config', '-c', default='config.json' ,help='Configuration file in the model folder.')
    parser.add_argument('--dataset', '-d', default='dataset.json' ,help='Dataset file in the datasets folder.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
