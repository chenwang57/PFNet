"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 9:34
# @Author  : Chen Wang
# @Site    :
# @File    : Data_Generator.py
# @Email   : chen.wang@ahnu.edu.cn
# @details :
*****************************************************
"""
import numpy as np
import tensorflow as tf


class Data_Generator():
    """
    Get final process tfrecord. These tfrecord have three files which called train.tfrecord, validate.tfrecord and
    test.tfrecord. They are in input folder. Each of them only has one element. To call get_dataset() function to get
    train, validate and test dataset. The return object is a list and the list store order is 0: temporal_month_pattern,
    1: temporal_week_pattern, 2: temporal_current_pattern, 3: flow.
    """

    def __init__(self, dataset_name, duration, batch_size):
        """
        Init Data_Generator instance and init some self variable.
        """
        # get all useful config parameter
        self.dataset_name = dataset_name
        self.duration = duration
        self.batch_size = batch_size
        # to delete . from ../
        self.tfrecord_path_root = './input/'
        self.__create_init_dataset(self.duration, self.batch_size)

    def __decode_tfrecord(self, example_proto):
        """
        Define the dicts of tfrecord structure and decord it.
        :param example_proto: protocol of the example
        :return: the decode tfrecord
        """
        dicts = {
            'month_pattern': tf.io.FixedLenFeature((), dtype=tf.string),
            'week_pattern': tf.io.FixedLenFeature((), dtype=tf.string),
            'current_pattern': tf.io.FixedLenFeature((), dtype=tf.string),
            'flow': tf.io.FixedLenFeature((), dtype=tf.string),
            'month_pattern_shape': tf.io.FixedLenFeature((3, ), dtype=tf.int64),
            'week_pattern_shape': tf.io.FixedLenFeature((3, ), dtype=tf.int64),
            'current_pattern_shape': tf.io.FixedLenFeature((3, ), dtype=tf.int64),
            'flow_shape': tf.io.FixedLenFeature((3, ), dtype=tf.int64)
        }
        parsed_example = tf.io.parse_single_example(example_proto, dicts)
        parsed_example['month_pattern'] = tf.io.decode_raw(parsed_example['month_pattern'], tf.float32)
        parsed_example['month_pattern'] = tf.reshape(parsed_example['month_pattern']
                                                     , parsed_example['month_pattern_shape'])
        parsed_example['week_pattern'] = tf.io.decode_raw(parsed_example['week_pattern'], tf.float32)
        parsed_example['week_pattern'] = tf.reshape(parsed_example['week_pattern']
                                                    , parsed_example['week_pattern_shape'])
        parsed_example['current_pattern'] = tf.io.decode_raw(parsed_example['current_pattern'], tf.float32)
        parsed_example['current_pattern'] = tf.reshape(parsed_example['current_pattern']
                                                       , parsed_example['current_pattern_shape'])
        parsed_example['flow'] = tf.io.decode_raw(parsed_example['flow'], tf.float32)
        parsed_example['flow'] = tf.reshape(parsed_example['flow'], parsed_example['flow_shape'])
        return parsed_example

    def __consumed_all_samples(self, iterator, batch_size):
        """
        Use iterator to iterate over all parameters(only one element in tfrecord which contain all records)
        :param iterator: the specific iterator
        :return: temporal_month_pattern, temporal_week_pattern, temporal_current_pattern, flow
        """
        global temporal_month_pattern, temporal_week_pattern, temporal_current_pattern, flow
        for element in iterator:
            # load samples for (record_num, sensor_num, pattern_length) and (record_num, sensor_num, )
            temporal_month_pattern = element['month_pattern']
            temporal_week_pattern = element['week_pattern']
            temporal_current_pattern = element['current_pattern']
            flow = element['flow']
        temporal_month_pattern = np.asarray(temporal_month_pattern, dtype=np.float32)
        temporal_week_pattern = np.asarray(temporal_week_pattern, dtype=np.float32)
        temporal_current_pattern = np.asarray(temporal_current_pattern, dtype=np.float32)
        flow = np.asarray(flow, dtype=np.float32)
        total = flow.shape[0]
        # drop some records to guarantee the total records can be exact division
        request = total // batch_size * batch_size
        return temporal_month_pattern[:request, ::], temporal_week_pattern[:request, ::]\
            , temporal_current_pattern[:request, ::], flow[:request, ::]

    def __create_init_dataset(self, duration, batch_size):
        """
        Read and decode the tfrecord files to tf.Dataset object and use it to create a generator. Then use it as a
        parameter to call __consumed_all_samples function to get a list of different kinds of train_dataset.
        :return: None
        """
        # create tfDataset and point to tfrecord file
        self.train_tfrecord_path = self.tfrecord_path_root + f'train_{duration}_{self.dataset_name}.tfrecord'
        train_data = tf.data.TFRecordDataset(self.train_tfrecord_path).map(self.__decode_tfrecord)
        # create single generator
        self.train_iter = iter(train_data)
        # consumed all samples to memory
        self.train_dataset = self.__consumed_all_samples(self.train_iter, batch_size)

        # create tfDataset and point to tfrecord file
        self.validate_tfrecord_path = self.tfrecord_path_root + f'validate_{duration}_{self.dataset_name}.tfrecord'
        validate_data = tf.data.TFRecordDataset(self.validate_tfrecord_path).map(self.__decode_tfrecord)
        # create single generator
        self.validate_iter = iter(validate_data)
        # consumed all samples to memory
        self.validate_dataset = self.__consumed_all_samples(self.validate_iter, batch_size)

        # create tfDataset and point to tfrecord file
        self.test_tfrecord_path = self.tfrecord_path_root + f'test_{duration}_{self.dataset_name}.tfrecord'
        test_data = tf.data.TFRecordDataset(self.test_tfrecord_path).map(self.__decode_tfrecord)
        # create single generator
        self.test_iter = iter(test_data)
        # consumed all samples to memory
        self.test_dataset = self.__consumed_all_samples(self.test_iter, batch_size)

    def get_dataset(self):
        """
        Get three dataset.
        :return: train, validate, test dataset
        """
        return self.train_dataset, self.validate_dataset, self.test_dataset


if __name__ == '__main__':
    data_generator = Data_Generator(dataset_name='LondonHW', duration=15, batch_size=4)
    train_dataset, validate_dataset, test_dataset = data_generator.get_dataset()
    print(train_dataset[3].shape, test_dataset[3].shape, validate_dataset[3].shape)
    for i in train_dataset:
        print(type(i))
        print(i.shape)






