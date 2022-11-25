"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2022/5/1 17:15
# @Author  : Chen Wang
# @Site    : 
# @File    : logger.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : 
*****************************************************
"""
import logging
import os
import datetime


def get_logger(root, name=None, is_debug_in_screen=True, ):
    """

    :param root: the log file's root path, str
    :param name: the logger's name, str
    :param is_debug_in_screen: whether to show DEBUG in screen, bool
    :return: logging object
    """
    # when is_debug_in_screen is True, show DEBUG and INFO in screen
    # when is_debug_in_screen is False, show DEBUG in file and info in both screen & file
    # INFO will always be in screen
    # DEBUG will always be in file

    # create a logger
    logger = logging.getLogger(name)

    # critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(message)s')

    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if is_debug_in_screen:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # create a handler for write log to file
    logfile = os.path.join(root, 'run.log')
    if not os.path.exists(root):
        os.makedirs(root)
    print('Creat Log File in: ', logfile)
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # add Handler to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    print(time)
    logger = get_logger('./log', is_debug_in_screen=True)
    logger.debug('this is a {} debug message'.format(1))
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')
