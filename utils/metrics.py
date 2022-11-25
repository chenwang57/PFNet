"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2022/5/1 16:12
# @Author  : Chen Wang
# @Site    : 
# @File    : metrics.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : 
*****************************************************
"""
import numpy as np


# for missing data
def mae_np(pred, true):
    mae = np.mean(np.abs(pred - true))
    return mae


# for missing data
def rmse_np(pred, true):
    mse = np.mean(np.square(pred - true))
    rmse = np.sqrt(mse)
    return rmse


def masked_rmse_np(pred, true, null_val=np.nan):
    return np.sqrt(masked_mse_np(pred=pred, true=true, null_val=null_val))


def masked_mse_np(pred, true, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(true)
        else:
            mask = np.not_equal(true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(pred, true)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mae_np(pred, true, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(true)
        else:
            mask = np.not_equal(true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, true)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_smape_np(pred, true, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(true)
        else:
            mask = np.not_equal(true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        smape = np.abs(np.divide(np.subtract(pred, true).astype('float32'), (np.abs(true) + np.abs(pred)) / 2))
        smape = np.nan_to_num(mask * smape)
        return 100 * np.mean(smape)


if __name__ == '__main__':
    pass
    # pred = np.array([1, 2, 3, 4])
    # true = np.array([2, 1, 4, 5])
