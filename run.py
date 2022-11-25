"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 9:34
# @Author  : Chen Wang
# @Site    :
# @File    : run.py
# @Email   : chen.wang@ahnu.edu.cn
# @details :
*****************************************************
"""
import argparse
from Trainer import Trainer
import tensorflow as tf
import configparser


if __name__ == '__main__':
    MODE = 'train'              # train or test
    DATASET = 'LondonHW'        # LondonHW or ManchesterHW
    DURATION = 60
    config_file = './model/config/{}_{}_PFNet.conf'.format(DATASET, DURATION)
    config = configparser.ConfigParser()
    config.read(config_file)
    config_dataset = config['dataset']
    config_model = config['model']
    config_train = config['train']
    config_setting = config['setting']

    # parser
    args = argparse.ArgumentParser()

    # setting config
    args.add_argument('--save_model', default=bool(config_setting['save_model']), type=bool, help='')
    args.add_argument('--spatial_embedding_path', default=config_setting['spatial_embedding_path'], type=str,
                      help='')
    args.add_argument('--model_root_path', default=config_setting['model_root_path'], type=str,
                      help='The best model parameters and logs save root path')
    args.add_argument('--mode', default=MODE, type=str, help='Train or test mode.')
    args.add_argument('--device', default=int(config_setting['device']), type=int, help='The index of GPUs.')
    args.add_argument('--debug', default=bool(config_setting['debug']), type=bool,
                      help='Whether the screen show DEBUG information.')
    args.add_argument('--cuda', default=bool(config_setting['cuda']), type=bool, help='Whether to use GPU device.')
    args.add_argument('--is_limit', default=bool(config_setting['is_limit']), type=bool,
                      help='Whether to limit the max gpu memory.')
    args.add_argument('--memory_limit', default=int(config_setting['memory_limit']), type=int,
                      help='The max gpu memory to allocate.')

    # dataset config
    args.add_argument('--dataset', default=config_dataset['dataset'], type=str, help='The name of dataset.')
    args.add_argument('--duration', default=int(config_dataset['duration']), type=int, help='The duration of dataset.')
    args.add_argument('--sensor_size', default=int(config_dataset['num_nodes']), type=int,
                      help='The number of sensors in dataset.')
    args.add_argument('--pattern_length', default=int(config_dataset['pattern_length']), type=int,
                      help='The pattern length of three segment pattern.')
    args.add_argument('--num_for_predict', default=int(config_dataset['num_for_predict']), type=int,
                      help='How many steps to predict.')

    # model config
    args.add_argument('--wm_left_depth', default=int(config_model['wm_left_depth']), type=int, help='')
    args.add_argument('--wm_right_depth', default=int(config_model['wm_right_depth']), type=int, help='')
    args.add_argument('--wm_cross_attention_depth', default=int(config_model['wm_cross_attention_depth']),
                      type=int, help='')
    args.add_argument('--wm_depth', default=int(config_model['wm_depth']), type=int, help='')
    args.add_argument('--cp_left_depth', default=int(config_model['cp_left_depth']), type=int, help='')
    args.add_argument('--cp_right_depth', default=int(config_model['cp_right_depth']), type=int, help='')
    args.add_argument('--cp_cross_attention_depth', default=int(config_model['cp_cross_attention_depth']),
                      type=int, help='')
    args.add_argument('--cp_depth', default=int(config_model['cp_depth']), type=int, help='')
    args.add_argument('--heads', default=int(config_model['heads']), type=int,
                      help='The number of multi-head attentions.')
    args.add_argument('--pool', default=config_model['pool'], type=str, help='')
    args.add_argument('--temporal_dropout', default=float(config_model['dropout']), type=float, help='')
    args.add_argument('--embedding_dropout', default=float(config_model['embedding_dropout']), type=float, help='')
    args.add_argument('--scale_dim', default=int(config_model['scale_dim']), type=int, help='')
    args.add_argument('--embedding_size', default=int(config_model['embedding_size']), type=int, help='')
    args.add_argument('--progressive_depth', default=int(config_model['progressive_depth']), type=int, help='')

    # training config
    args.add_argument('--lr_boundaries', default=[int(x) for x in config_train['lr_decay_step'].split(',')],
                      type=list, help='')
    args.add_argument('--decay_rate', default=float(config_train['lr_decay_rate']), type=float, help='')
    args.add_argument('--epochs', default=int(config_train['epochs']), type=int, help='')
    args.add_argument('--learning_rate', default=float(config_train['learning_rate_init']), type=float, help='')
    args.add_argument('--batch_size', default=int(config_train['batch_size']), type=int, help='')
    args.add_argument('--shuffle', default=bool(config_train['is_shuffle']), type=bool,
                      help='whether to shuffle the dataset records.')
    args.add_argument('--early_stop_patience', default=int(config_train['early_stop_patience']), type=int, help='')

    args = args.parse_args()

    # init gpu config
    if args.cuda:
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_visible_devices(devices=gpus[args.device], device_type='GPU')
        if args.is_limit:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.memory_limit)]
            )  # set the max video memory
    else:
        cpus = tf.config.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpus[0], device_type='CPU')

    trainer = Trainer(args)

    # train
    history = trainer.fit()

    # test
    # trainer.evaluate(is_pretrained=True, model_path='./experiments/LondonHW_15/2022-05-12-22-33-53/best_model')

