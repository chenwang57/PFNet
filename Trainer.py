"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 11:03
# @Author  : Chen Wang
# @Site    : 
# @File    : Trainer.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : 
*****************************************************
"""
import tensorflow as tf
import numpy as np
from model.pfnet import PFNet
import pickle
import os
from Data_Generator import Data_Generator
from utils.metrics import masked_mae_np, masked_rmse_np, masked_smape_np
from utils.logger import get_logger
from datetime import datetime


class Trainer:
    def __init__(self, args):
        assert args.mode in ['train', 'test'], 'The mode should be train or test!'
        self.args = args
        self.experiment_root_path = os.path.join(self.args.model_root_path,
                                                 str(self.args.dataset) + '_' + str(self.args.duration),
                                                 datetime.now().strftime(f'%Y-%m-%d-%H-%M-%S'))
        if not os.path.exists(self.experiment_root_path):
            os.makedirs(self.experiment_root_path)

        # init logger object, and tensorboard, logger & model save path
        self.logger = get_logger(root=self.experiment_root_path,
                                 name='{}_{}_{}'.format(self.args.dataset,
                                                        self.args.duration,
                                                        datetime.now().strftime(f'%Y-%m-%d-%H-%M-%S')),
                                 is_debug_in_screen=self.args.debug)
        self.logger.info(f'The best model and running log will save in {self.experiment_root_path}.')

        self.tensorboard_path = os.path.join(self.experiment_root_path, 'tensorboard-logs')

        # init dataset
        self.scaler, self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self.__get_dataset()

        # init mode
        if self.args.mode == 'train':
            self.is_pretrained = False
        else:
            self.is_pretrained = True
        self.model = self.__init_model()

        self.logger.info('============================================================================')
        self.logger.info('Model settings:')
        self.logger.info(str(self.args)[10: -1])
        self.logger.info('============================================================================')

    def fit(self):
        assert self.is_pretrained is False, 'The mode should be trained if you want to train model!'

        boundaries = self.args.lr_boundaries
        values = [self.args.learning_rate]
        for i, epoch in enumerate(self.args.lr_boundaries):
            values.append(values[-1] * self.args.decay_rate)
        scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=values)
        lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

        callbacks = [
            lr,
            tf.keras.callbacks.TensorBoard(self.tensorboard_path),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.experiment_root_path, 'best_model'),
                                               save_best_only=True,
                                               save_weights_only=True,
                                               monitor='val_loss',
                                               mode='min'),
            tf.keras.callbacks.EarlyStopping(patience=self.args.early_stop_patience,
                                             mode='min',
                                             monitor='val_loss')
        ]

        history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val),
                                 batch_size=self.args.batch_size, shuffle=self.args.shuffle,
                                 epochs=self.args.epochs, callbacks=callbacks)
        self.logger.info('Finish training!')

        self.logger.info('Load the best model.')
        self.__load_model_parameters(os.path.join(self.experiment_root_path, 'best_model'))
        self.logger.info('Start evaluating!')

        y_pred = self.model.predict(self.x_test, batch_size=self.args.batch_size)

        y_pred_shape = y_pred.shape
        y_true_shape = self.y_test.shape
        y_pred = tf.reshape(self.scaler.inverse_transform(y_pred.reshape(-1, 1)), shape=y_pred_shape)
        y_true = tf.reshape(self.scaler.inverse_transform(self.y_test.reshape(-1, 1)), shape=y_true_shape)

        np.save(f'{self.experiment_root_path}/{self.args.dataset}_{self.args.duration}_pred', y_pred)
        np.save(f'{self.experiment_root_path}/{self.args.dataset}_{self.args.duration}_true', y_true)

        mae, rmse, smape = self.all_metrics(y_pred, y_true, 0)
        self.logger.info('============================================================================')
        self.logger.info('Finish evaluating, the performance is shown below:')
        self.logger.info(f'MAE Loss: {mae}.')
        self.logger.info(f'RMSE Loss: {rmse}.')
        self.logger.info(f'SMAPE Loss: {smape}.')
        self.logger.info('============================================================================')

        return history

    def evaluate(self, is_pretrained=False, model_path=None):
        if is_pretrained is True:
            assert model_path is not None, 'Please give the best model path!'
            self.logger.info('Load pretrained model parameters!')
            self.__load_model_parameters(model_path)
            self.logger.info('Start evaluating!')

            pred = self.model.predict(self.x_test, batch_size=self.args.batch_size)

            true = self.y_test
            pred_shape = pred.shape
            true_shape = true.shape
            pred = tf.reshape(self.scaler.inverse_transform(pred.reshape(-1, 1)), shape=pred_shape)
            true = tf.reshape(self.scaler.inverse_transform(true.reshape(-1, 1)), shape=true_shape)

            np.save(f'{self.experiment_root_path}/{self.args.dataset}_{self.args.duration}_pred', pred)
            np.save(f'{self.experiment_root_path}/{self.args.dataset}_{self.args.duration}_true', true)

            mae, rmse, smape = self.all_metrics(pred, true, 0)
            self.logger.info('============================================================================')
            self.logger.info('Finish evaluating, the performance is shown below:')
            self.logger.info(f'MAE Loss: {mae}.')
            self.logger.info(f'RMSE Loss: {rmse}.')
            self.logger.info(f'SMAPE Loss: {smape}.')
            self.logger.info('============================================================================')

        elif is_pretrained is False:
            self.fit()
        else:
            raise ValueError('is_pretrained must be True or False!')

    def __init_model(self):
        pfnet = PFNet(self.args)
        optimizer = tf.keras.optimizers.Adam(self.args.learning_rate)
        loss = self.mae_tf
        metrics = [self.rmse_tf]
        pfnet.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return pfnet

    def __get_dataset(self):
        """
            The three input lists must be same by length.
            Dataset list structure:
            0: temporal_month_pattern,
            1: temporal_week_pattern,
            2: temporal_current_pattern,
            3: flow
        """
        data_generator = Data_Generator(dataset_name=self.args.dataset,
                                        duration=self.args.duration,
                                        batch_size=self.args.batch_size)
        train_dataset, val_dataset, test_dataset = data_generator.get_dataset()
        scaler = self.read_pkl(f'./input/{self.args.dataset}_{self.args.duration}_scaler.pkl')
        x_train = train_dataset[0: 3]
        y_train = train_dataset[3]
        x_val = val_dataset[0: 3]
        y_val = val_dataset[3]
        x_test = test_dataset[0: 3]
        y_test = test_dataset[3]
        self.logger.info('============================================================================')
        self.logger.info(f'The setting of dataset(name: {self.args.dataset}, duration: {self.args.duration}):')
        self.logger.info(
            f'x_train: month-segment {x_train[0].shape}, week-segment {x_train[1].shape}, current-segment {x_train[2].shape}')
        self.logger.info(f'y_train: {y_train.shape}')
        self.logger.info(
            f'x_val: month-segment {x_val[0].shape}, week-segment {x_val[1].shape}, current-segment {x_val[2].shape}')
        self.logger.info(f'y_val: {y_val.shape}')
        self.logger.info(
            f'x_test: month-segment {x_test[0].shape}, week-segment {x_test[1].shape}, current-segment {x_test[2].shape}')
        self.logger.info(f'y_test: {y_test.shape}')
        self.logger.info(f'The number of train records is {y_train.shape[0]}.')
        self.logger.info(f'The number of val records is {y_val.shape[0]}.')
        self.logger.info(f'The number of test records is {y_test.shape[0]}.')
        self.logger.info('============================================================================')

        return scaler, x_train, y_train, x_val, y_val, x_test, y_test

    def __load_model_parameters(self, model_path):
        self.logger.info(f'Load pretrained model parameters from "{model_path}".')
        self.model.load_weights(model_path)

    def mae_tf(self, y_true, y_pred):
        y_true = (y_true - self.scaler.min_) / self.scaler.scale_
        y_pred = (y_pred - self.scaler.min_) / self.scaler.scale_
        return tf.keras.losses.mae(y_true, y_pred)

    def rmse_tf(self, y_true, y_pred):
        y_true = (y_true - self.scaler.min_) / self.scaler.scale_
        y_pred = (y_pred - self.scaler.min_) / self.scaler.scale_
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    def huber_tf(self, y_true, y_pred):
        y_true = (y_true - self.scaler.min_) / self.scaler.scale_
        y_pred = (y_pred - self.scaler.min_) / self.scaler.scale_
        return tf.keras.losses.huber(y_true, y_pred, delta=self.args.huber_delta)

    @staticmethod
    def read_pkl(file_path):
        file = open(file_path, 'rb')
        res = pickle.load(file)
        file.close()
        return res

    @staticmethod
    def all_metrics(y_pred, y_true, null_val=np.nan):
        mae = masked_mae_np(y_pred, y_true, null_val)
        rmse = masked_rmse_np(y_pred, y_true, null_val)
        smape = masked_smape_np(y_pred, y_true, null_val)
        return mae, rmse, smape


if __name__ == '__main__':
    pass
