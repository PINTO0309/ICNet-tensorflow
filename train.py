"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet 
"""
import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np

from model import ICNet_BN
from utils.config import Config
from utils.visualize import decode_labels
from utils.image_reader import ImageReader, prepare_label
from utils.pca import pca

sys.path.append("/home/sangwon/Projects/aaf")

from Adaptive_Affinity_Fields.utils import general as aaf_general
from Adaptive_Affinity_Fields.network.common import layers as nn
from Adaptive_Affinity_Fields.network.aaf import losses as lossx 

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced ICNet")
    
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--update-mean-var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true",
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument("--dataset", required=True,
                        help="Which dataset to trained with",
                        choices=['cityscapes', 'ade20k', 'others'])
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    parser.add_argument("--snapshot-dir", type=str, default='./snapshots/',
                        help="where to store snapshots, default is ./snapshots/")
    return parser.parse_args()

def get_mask(gt, num_classes, ignore_label):
    less_equal_class = tf.less_equal(gt, num_classes-1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)

    return indices

def create_loss(output, label, num_classes, ignore_label):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1,])

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss

def create_losses(net, label, cfg):
    # Get output from different branches
    sub4_out = net.layers['sub4_out']
    sub24_out = net.layers['sub24_out']
    sub124_out = net.layers['conv6_cls']
    embedding_out = net.layers['conv3_sub1']

    loss_sub4 = create_loss(sub4_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
    loss_sub24 = create_loss(sub24_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
    loss_sub124 = create_loss(sub124_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])

    # Affinity loss
    kld_margin = 3.0
    kld_lambda_1 = 1.0
    kld_lambda_2 = 4.0
    prob = tf.nn.softmax(embedding_out, dim=-1)
    edge_loss, not_edge_loss = lossx.affinity_loss(label, prob, cfg.param['num_classes'], kld_margin)
    aff_loss = tf.reduce_mean(edge_loss)*kld_lambda_1
    aff_loss += tf.reduce_mean(not_edge_loss)*kld_lambda_2

    l2_losses = [cfg.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    
    # Calculate weighted loss of three branches, you can tune LAMBDA values to get better results.
    reduced_loss = cfg.LAMBDA1 * loss_sub4 +  cfg.LAMBDA2 * loss_sub24 + cfg.LAMBDA3 * loss_sub124 + tf.add_n(l2_losses) + aff_loss

    return loss_sub4, loss_sub24, loss_sub124, aff_loss, reduced_loss

class TrainConfig(Config):
    def __init__(self, dataset, is_training,  filter_scale=1, random_scale=None, random_mirror=None):
        Config.__init__(self, dataset, is_training, filter_scale, random_scale, random_mirror)

    # Set pre-trained weights here (You can download weight using `python script/download_weights.py`) 
    # Note that you need to use "bnnomerge" version.
    model_weight = './model/cityscapes/icnet_cityscapes_train_30k_bnnomerge.npy'
    
    # Set hyperparameters here, you can get much more setting in Config Class, see 'utils/config.py' for details.
    LAMBDA1 = 0.16
    LAMBDA2 = 0.4
    LAMBDA3 = 1.0
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-4

def main():
    """Create the model and start the training."""
    args = get_arguments()

    """
    Get configurations here. We pass some arguments from command line to init configurations, for training hyperparameters, 
    you can set them in TrainConfig Class.

    Note: we set filter scale to 1 for pruned model, 2 for non-pruned model. The filters numbers of non-pruned
          model is two times larger than prunde model, e.g., [h, w, 64] <-> [h, w, 32].
    """
    cfg = TrainConfig(dataset=args.dataset, 
                is_training=True,
                random_scale=args.random_scale,
                random_mirror=args.random_mirror,
                filter_scale=args.filter_scale)
    cfg.SNAPSHOT_DIR = args.snapshot_dir
    cfg.display()

    # Setup training network and training samples
    train_reader = ImageReader(cfg=cfg, mode='train')
    train_net = ICNet_BN(image_reader=train_reader, 
                            cfg=cfg, mode='train')

    loss_sub4, loss_sub24, loss_sub124, aff_loss, reduced_loss = create_losses(train_net, train_net.labels, cfg)

    # Using Poly learning rate policy 
    base_lr = tf.constant(cfg.LEARNING_RATE)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / cfg.TRAINING_STEPS), cfg.POWER))
    
    # Set restore variable 
    restore_var = tf.global_variables()
    all_trainable = [v for v in tf.trainable_variables() if ('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma]

    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if args.update_mean_var == False:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        opt_conv = tf.train.MomentumOptimizer(learning_rate, cfg.MOMENTUM)
        grads = tf.gradients(reduced_loss, all_trainable)
        train_op = opt_conv.apply_gradients(zip(grads, all_trainable))
    
    # Process for visualization.
    with tf.device('/cpu:0'):
        # Image summary for input image, ground-truth label and prediction.
        image_batch = train_net.images
        label_batch = train_net.labels
        seg_outputs = train_net.layers['conv6_cls']
        aaf_outputs = train_net.layers['conv3_sub1']
        num_classes = cfg.param['num_classes']

        # visualize semantic segmentation output
        output_vis = tf.image.resize_nearest_neighbor(
            seg_outputs, tf.shape(image_batch)[1:3,])
        output_vis = tf.argmax(output_vis, axis=3)
        output_vis = tf.expand_dims(output_vis, dim=3)
        output_vis = tf.cast(output_vis, dtype=tf.uint8)
        
        # visualize instance embedding output
        aaf_vis = pca(aaf_outputs, 3)
        aaf_vis = tf.image.resize_nearest_neighbor(
            aaf_vis, tf.shape(image_batch)[1:3,])
        aaf_summary = tf.cast(aaf_vis, dtype=tf.uint8)

        labels_vis = tf.cast(label_batch, dtype=tf.uint8)
    
        in_summary = tf.py_func(
            aaf_general.inv_preprocess,
            [image_batch, cfg.IMG_MEAN],
            tf.uint8)
        gt_summary = tf.py_func(
            aaf_general.decode_labels,
            [labels_vis, num_classes],
            tf.uint8)
        out_summary = tf.py_func(
            aaf_general.decode_labels,
            [output_vis, num_classes],
            tf.uint8)
        # Concatenate image summaries in a row.
        total_summary = tf.summary.image(
            'images', 
            tf.concat(axis=2, values=[in_summary, gt_summary, out_summary, aaf_summary]), 
            max_outputs=cfg.BATCH_SIZE)

        # Scalar summary for different loss terms.
        seg_loss_summary = tf.summary.scalar(
            'seg_loss', loss_sub124)
        aff_loss_summary = tf.summary.scalar(
            'aff_loss', aff_loss)
        total_summary = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(cfg.SNAPSHOT_DIR,
                                            graph=tf.get_default_graph())

    # Create session & restore weights (Here we only need to use train_net to create session since we reuse it)
    train_net.create_session()
    train_net.restore(cfg.model_weight, restore_var)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

    # Iterate over training steps.
    for step in range(cfg.TRAINING_STEPS):
        start_time = time.time()
            
        feed_dict = {step_ph: step}
        if step % cfg.SAVE_PRED_EVERY == 0:
            sess_outs = [reduced_loss, loss_sub4, loss_sub24, loss_sub124, aff_loss, total_summary, train_op]
            loss_value, loss1, loss2, loss3, loss4, summary, _ = train_net.sess.run(sess_outs, feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            train_net.save(saver, cfg.SNAPSHOT_DIR, step)
        else:
            sess_outs = [reduced_loss, loss_sub4, loss_sub24, loss_sub124, aff_loss, train_op]
            loss_value, loss1, loss2, loss3, loss4, _ = train_net.sess.run(sess_outs, feed_dict=feed_dict)            

        duration = time.time() - start_time
        print('step {:d} \t total loss = {:.3f}, sub4 = {:.3f}, sub24 = {:.3f}, sub124 = {:.3f}, aff_loss= {:.3f} ({:.3f} sec/step)'.\
                    format(step, loss_value, loss1, loss2, loss3, loss4, duration))
    
    
if __name__ == '__main__':
    main()
