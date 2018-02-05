import importlib                    # To import other modules
import sys
import time
import os.path as path
# import numpy as n
from keras import backend as K
# from keras.optimizers import Adam, SGD, RMSprop
# from keras.layers import Input
# from keras.models import Model
# from keras_frcnn import config, data_generators
# from keras_frcnn import losses as losses_fn
# import keras_frcnn.roi_helpers as roi_helpers
# from keras.utils import generic_utils

# Import available base networks
from models.faster_rcnn.base_network import resnet50 as resnet50
from models.faster_rcnn.base_network import vgg16 as vgg16

def train(cfg, dataset, train_imgs, test_imgs):
    """
    TODO
    """

    # =============================
    # === DEFINE NEURAL NETWORK ===
    # =============================
    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))

    # Load module corresponding to chosen base network
    module_base_network = 'models.' + cfg['model_name'] + '.base_network.' + cfg['base_network']
    try:
        nn = importlib.import_module(module_base_network)
    except Exception as e:
        sys.exit('ERROR: Impossible to load the base_network module \'' + module_base_network + '\'')

    # --- LAYERS ---
    # Define shared layers
    shared_layers = nn.nn_base(img_input, trainable=True)

    # Define RPN
    num_anchors = len(cfg['anchor_boxes']['scales']) * len(cfg['anchor_boxes']['ratios'])
    rpn = nn.rpn(shared_layers, num_anchors)

    # Define Classifier
    classifier = nn.classifier(shared_layers, roi_input, cfg.num_rois, nb_classes=len(classes_count), trainable=True)

    # --- MODELS ---
    # Define Models
    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # This is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    # --- LOAD WEIGHTS ---
    base_net_weights = path.join('model', cfg['model_name'], 'base_network', 'weights', nn.get_weight_path())
    try:
        print('Loading weights from \'' + base_net_weights + '\'...')
        model_rpn.load_weights(cfg.model_path, by_name=True)
        print(' ↳ SUCCESS: RPN\'s weighhts loaded.'')
        model_classifier.load_weights(cfg.model_path, by_name=True)
        print(' ↳ SUCCESS: Classifier\'s weighhts loaded.'')
    except Exception as e:
        print(e)
        print('ERROR: Impossible to load pretrained model weights.')
        print(' ↳ Pretrained weights can be downloaded from:  https://github.com/fchollet/keras/tree/master/keras/applications')

    # TODO save config file
    # with open(cfg.config_save_file, 'wb') as config_f:
    #     pickle.dump(cfg, config_f)
    #     print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
    #         cfg.config_save_file))


    # --- PREPARE DATA ---
    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, cfg, nn.get_img_output_length,
                                                   K.image_dim_ordering(), mode='train')
    data_gen_test = data_generators.get_anchor_gt(test_imgs, classes_count, cfg, nn.get_img_output_length,
                                                 K.image_dim_ordering(), mode='val')







    # --- Optimizer ---
    optimizer_rpn = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)

    model_rpn.compile(optimizer=optimizer_rpn,
                      loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])

    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    epoch_length = int(cfg['epoch_length'])
    num_epochs = int(cfg['num_epochs'])
    iter_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf

    class_mapping_inv = {v: k for k, v in class_mapping.items()}
    print('Starting training')

    vis = True

    for epoch_num in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            try:

                if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(
                        'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap'
                              ' the ground truth boxes. Check RPN settings or keep training.')

                X, Y, img_data = next(data_gen_train)

                loss_rpn = model_rpn.train_on_batch(X, Y)

                P_rpn = model_rpn.predict_on_batch(X)

                result = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
                                                overlap_thresh=0.7,
                                                max_boxes=300)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(result, img_data, cfg, class_mapping)

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

                if cfg.num_rois > 1:
                    if len(pos_samples) < cfg.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois // 2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
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

                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num,
                               [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                ('detector_cls', np.mean(losses[:iter_num, 2])),
                                ('detector_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if cfg.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if cfg.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(cfg.model_path)

                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                # save model
                model_all.save_weights(cfg.model_path)
                continue
    print('Training complete, exiting.')
