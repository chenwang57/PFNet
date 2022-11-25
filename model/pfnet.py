"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 11:25
# @Author  : Chen Wang
# @Site    : 
# @File    : pfnet.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : 
*****************************************************
"""
import tensorflow as tf
import numpy as np
from einops import repeat
import sys
sys.path.append('./model/')
import temporal_module as temporal
import st_fusion_module as st


class PFNet(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = self.args.batch_size

        # init DMVSE module
        self.temporal_module = temporal.TemporalModule(
            pattern_length=self.args.pattern_length,
            num_sensor=self.args.sensor_size,
            wm_left_depth=self.args.wm_left_depth,
            wm_right_depth=self.args.wm_right_depth,
            wm_cross_attention_depth=self.args.wm_cross_attention_depth,
            wm_depth=self.args.wm_depth,
            cp_left_depth=self.args.cp_left_depth,
            cp_right_depth=self.args.cp_right_depth,
            cp_cross_attention_depth=self.args.cp_cross_attention_depth,
            cp_depth=self.args.cp_depth,
            heads=self.args.heads,
            pool=self.args.pool,
            dropout=self.args.temporal_dropout,
            embedding_dropout=self.args.embedding_dropout,
            scale_dim=self.args.scale_dim,
            batch_size=self.batch_size
        )

        # get SKE output
        self.spatial_embedding = repeat(tf.expand_dims(np.load(self.args.spatial_embedding_path), axis=0)
                                        , '() s e -> b s e', b=self.batch_size)

        # init CPA module
        self.st_fusion = st.STFusion(
            pattern_length=self.args.pattern_length,
            embedding_dim=self.args.embedding_size,
            ffn_hidden_dim=self.args.pattern_length * self.args.scale_dim,
            progressive_depth=self.args.progressive_depth,
            batch_size=self.batch_size,
            num_for_predict=self.args.num_for_predict
        )

    def call(self, inputs, training=None, mask=None):
        month, week, current = inputs
        x_temporal = self.temporal_module([month, week, current])
        x_spatial = self.spatial_embedding
        return self.st_fusion([x_temporal, x_spatial])


if __name__ == '__main__':
    pass
