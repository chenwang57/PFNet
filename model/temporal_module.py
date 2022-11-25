"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 9:34
# @Author  : Chen Wang
# @Site    : 
# @File    : temporal_module.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : 
*****************************************************
"""
import tensorflow as tf
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange


# Residual Connection class
class Residual(tf.keras.Model):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# Pre-LayerNormalization class
class PreNorm(tf.keras.Model):
    def __init__(self, fn):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(axis=[1, 2])
        self.fn = fn

    def call(self, x, *args, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# Feed Forward class
class FeedForward(tf.keras.Model):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(dim),
            tf.keras.layers.Dropout(dropout)
        ])

    def call(self, x, *args, **kwargs):
        return self.net(x)


# Attention class
class Attention(tf.keras.Model):
    def __init__(self, dimension, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dimension)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = tf.keras.layers.Dense(inner_dim * 3, use_bias=False)
        self.to_out = tf.keras.Sequential([
            tf.keras.layers.Dense(dimension),
            tf.keras.layers.Dropout(dropout)
        ]) if project_out else tf.identity

    def call(self, x, *args, **kwargs):
        # x.shape = (batch_size, sensor_num, pattern_length)
        b, s, p, h = *x.shape, self.heads
        qkv = tf.split(self.to_qkv(x), axis=-1, num_or_size_splits=3)
        q, k, v = map(lambda t: rearrange(t, 'b s (h p) -> b h s p', h=h), qkv)

        # q, k, v shape: (b h s p)
        dots = tf.einsum('b h i p, b h j p -> b h i j', q, k) * self.scale

        attention = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h s p -> b s (h p)')
        out = self.to_out(out)
        return out


# Cross Attention class
class CrossAttention(tf.keras.Model):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = tf.keras.layers.Dense(inner_dim, use_bias=False)
        self.to_v = tf.keras.layers.Dense(inner_dim, use_bias=False)
        self.to_q = tf.keras.layers.Dense(inner_dim, use_bias=False)

        self.to_out = tf.keras.Sequential([
            tf.keras.layers.Dense(dim),
            tf.keras.layers.Dropout(dropout)
        ]) if project_out else tf.identity

    def call(self, x_qkv, **kwargs):
        # x.shape = (batch_size, sensor_num, pattern_length)
        b, s, p, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b s (h p) -> b h s p', h=h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b s (h p) -> b h s p', h=h)

        q = self.to_q(tf.expand_dims(x_qkv[:, 0], axis=1))
        q = rearrange(q, 'b s (h p) -> b h s p', h=h)

        dots = tf.einsum('b h i p, b h j p -> b h i j', q, k) * self.scale

        attention = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h s p -> b s (h p)')
        out = self.to_out(out)
        return out


# Multi-Layer Transformer Encoder class
class Transformer(tf.keras.Model):
    def __init__(self, dimension, depth, heads, dimension_head, mlp_dim, dropout=0.):
        super().__init__()
        self.trans_layers = tf.keras.Sequential()
        for _ in range(depth):
            self.trans_layers.add(tf.keras.Sequential([
                Residual(PreNorm(Attention(dimension, heads=heads, dim_head=dimension_head, dropout=dropout))),
                Residual(PreNorm(FeedForward(dimension, mlp_dim, dropout=dropout)))
            ]))

    def call(self, x, *args, **kwargs):
        return self.trans_layers(x)


# Multi-View Transformer Encoder Sub-block
class MultiViewTransformerEncoderSub(tf.keras.Model):
    def __init__(self, left_dim, left_depth, left_heads, left_dim_head, left_mlp_dim
                 , right_dim, right_depth, right_heads, right_dim_head, right_mlp_dim
                 , num_sensor, batch_size, cross_attention_depth=1, cross_attention_heads=3, dropout=0.):
        super().__init__()
        self.batch_size = batch_size
        self.num_sensor = num_sensor

        self.transformer_encoder_left = Transformer(left_dim, left_depth, left_heads, left_dim_head, left_mlp_dim)
        self.transformer_encoder_right = Transformer(right_dim, right_depth, right_heads, right_dim_head, right_mlp_dim)

        # Dual-Branch Cross Attention
        self.cross_attention_layers = []
        for _ in range(cross_attention_depth):
            self.cross_attention_layers.append([
                tf.keras.layers.Dense(right_dim),
                tf.keras.layers.Dense(left_dim),
                PreNorm(CrossAttention(right_dim
                                       , heads=cross_attention_heads
                                       , dim_head=right_dim_head
                                       , dropout=dropout)),
                tf.Variable(tf.random.normal([1, num_sensor + 1, right_dim])),
                Attention(left_dim, 1, left_dim),
                tf.keras.layers.Dense(left_dim),
                tf.keras.layers.Dense(right_dim),
                PreNorm(CrossAttention(left_dim
                                       , heads=cross_attention_heads
                                       , dim_head=left_dim_head
                                       , dropout=dropout)),
                tf.Variable(tf.random.normal([1, num_sensor + 1, left_dim])),
                Attention(right_dim, 1, right_dim),
            ])

    def call(self, inputs, *args, **kwargs):
        left, left_pos, right, right_pos = inputs
        left_pos = repeat(left_pos, '() s p -> b s p', b=self.batch_size)   # (batch_size, 1, pattern_length)
        right_pos = repeat(right_pos, '() s p -> b s p', b=self.batch_size)
        left = self.transformer_encoder_left(left)
        right = self.transformer_encoder_right(right)

        for f_rl, g_lr, ca_r, b1, sa1, f_lr, g_rl, ca_l, b2, sa2 in self.cross_attention_layers:
            left_tf = left[:, 0]   # (batch_size, pattern_length)
            x_left = left[:, 1:]    # (batch_size, num_nodes, pattern_length)
            right_tf = right[:, 0]
            x_right = right[:, 1:]
            b1 = repeat(b1, '() s p -> b s p', b=self.batch_size)  # (batch_size, num_sensor + 1, pattern_length)
            b2 = repeat(b2, '() s p -> b s p', b=self.batch_size)

            # Cross attention for left branch
            # block 1
            cal_kv = tf.concat([right_pos, x_right], axis=1)    # (batch_size, num_sensor + 1, pattern_length)
            cal_kv = cal_kv + tf.einsum('ijk, ijk -> ijk', sa2(cal_kv), b2)
            # block 2
            cal_q = f_lr(tf.expand_dims(left_tf, axis=1))
            cal_qkv = tf.concat([cal_q, cal_kv[:, 1:]], axis=1)
            cal_out = cal_q + ca_l(cal_qkv)
            cal_out = g_rl(cal_out)
            left = tf.concat([cal_out, x_left], axis=1)

            # Cross attention for right branch
            # block 1
            car_kv = tf.concat([left_pos, x_left], axis=1)
            car_kv = car_kv + tf.einsum('ijk, ijk -> ijk', sa1(car_kv), b1)
            # block 2
            car_q = f_rl(tf.expand_dims(right_tf, axis=1))
            car_qkv = tf.concat([car_q, car_kv[:, 1:]], axis=1)
            car_out = car_q + ca_r(car_qkv)
            car_out = g_lr(car_out)
            right = tf.concat([car_out, x_right], axis=1)

        return [left, right]


# DMVSE module
class TemporalModule(tf.keras.Model):
    def __init__(self, pattern_length, num_sensor, wm_left_depth, wm_right_depth, wm_cross_attention_depth
                 , wm_depth, cp_left_depth, cp_right_depth, cp_cross_attention_depth, cp_depth, batch_size, heads=3,
                 pool='tf', dropout=0., embedding_dropout=0., scale_dim=4, ):
        super().__init__()
        assert pool in {'tf', 'mean'}, 'pool type must be either tf (tf token) or mean (mean pooling)'
        assert pattern_length % heads == 0, 'pattern_length must be divisible by the heads.'
        self.batch_size = batch_size

        # self.to_embedding_month = tf.keras.Sequential([
        #     tf.keras.layers.Dense(pattern_length)
        # ])

        # self.to_embedding_week = tf.keras.Sequential([
        #     tf.keras.layers.Dense(pattern_length)
        # ])

        # self.to_embedding_current = tf.keras.Sequential([
        #     tf.keras.layers.Dense(pattern_length)
        # ])

        self.pos_embedding_month = tf.Variable(tf.random.normal([1, 1, pattern_length]))
        self.tf_token_month = tf.Variable(tf.random.normal([1, 1, pattern_length]))
        # self.dropout_month = tf.keras.layers.Dropout(embedding_dropout)

        self.pos_embedding_week = tf.Variable(tf.random.normal([1, 1, pattern_length]))
        self.tf_token_week = tf.Variable(tf.random.normal([1, 1, pattern_length]))
        # self.dropout_week = tf.keras.layers.Dropout(embedding_dropout)

        self.pos_embedding_current = tf.Variable(tf.random.normal([1, 1, pattern_length]))
        self.tf_token_current = tf.Variable(tf.random.normal([1, 1, pattern_length]))
        # self.dropout_current = tf.keras.layers.Dropout(embedding_dropout)

        self.pos_embedding_p = tf.Variable(tf.random.normal([1, 1, pattern_length]))

        self.wm_block = []
        self.cp_block = []
        for _ in range(wm_depth):
            self.wm_block.append(MultiViewTransformerEncoderSub(
                left_dim=pattern_length,
                left_depth=wm_left_depth,
                left_heads=heads,
                left_dim_head=pattern_length // heads,
                left_mlp_dim=pattern_length * scale_dim,
                right_dim=pattern_length,
                right_depth=wm_right_depth,
                right_heads=heads,
                right_dim_head=pattern_length // heads,
                right_mlp_dim=pattern_length * scale_dim,
                cross_attention_depth=wm_cross_attention_depth,
                cross_attention_heads=heads,
                dropout=dropout,
                num_sensor=num_sensor,
                batch_size=batch_size
            ))
        for _ in range(cp_depth):
            self.cp_block.append(MultiViewTransformerEncoderSub(
                left_dim=pattern_length,
                left_depth=cp_left_depth,
                left_heads=heads,
                left_dim_head=pattern_length // heads,
                left_mlp_dim=pattern_length * scale_dim,
                right_dim=pattern_length,
                right_depth=cp_right_depth,
                right_heads=heads,
                right_dim_head=pattern_length // heads,
                right_mlp_dim=pattern_length * scale_dim,
                cross_attention_depth=cp_cross_attention_depth,
                cross_attention_heads=heads,
                dropout=dropout,
                num_sensor=num_sensor,
                batch_size=batch_size
            ))

        self.pool = pool

        self.wm_ffn_left = PreNorm(FeedForward(pattern_length, pattern_length * scale_dim))
        self.wm_ffn_right = PreNorm(FeedForward(pattern_length, pattern_length * scale_dim))
        self.cp_ffn_left = PreNorm(FeedForward(pattern_length, pattern_length * scale_dim))
        self.cp_ffn_right = PreNorm(FeedForward(pattern_length, pattern_length * scale_dim))

    def call(self, pattern, *args, **kwargs):
        month_pattern, week_pattern, current_pattern = pattern

        # month_pattern = self.to_embedding_month(month_pattern)
        s, p = month_pattern.shape[1:]

        tf_token_month = repeat(self.tf_token_month, '() s p -> b s p', b=self.batch_size)
        month_pattern = tf.concat([tf_token_month, month_pattern], axis=-2)
        # month_pattern = self.dropout_month(month_pattern)

        # week_pattern = self.to_embedding_week(week_pattern)
        s, p = week_pattern.shape[1:]

        tf_token_week = repeat(self.tf_token_week, '() s p -> b s p', b=self.batch_size)
        week_pattern = tf.concat([tf_token_week, week_pattern], axis=-2)
        # week_pattern = self.dropout_week(week_pattern)

        # current_pattern = self.to_embedding_current(current_pattern)
        s, p = current_pattern.shape[1:]

        tf_token_current = repeat(self.tf_token_current, '() s p -> b s p', b=self.batch_size)
        current_pattern = tf.concat([tf_token_current, current_pattern], axis=-2)
        # current_pattern = self.dropout_current(current_pattern)

        # print('Temporal_Module: \n month_pattern: \n {0} \n week_pattern: \n {1}'.format(month_pattern, week_pattern))
        for wm in self.wm_block:
            month_pattern, week_pattern = wm([month_pattern, self.pos_embedding_month,
                                              week_pattern, self.pos_embedding_week])

        wm_integrate_pattern = self.wm_ffn_left(month_pattern) + self.wm_ffn_right(week_pattern)

        for cp in self.cp_block:
            wm_integrate_pattern, current_pattern = cp([wm_integrate_pattern, self.pos_embedding_p,
                                                        current_pattern, self.pos_embedding_current])

        final_integrate_pattern = self.cp_ffn_left(wm_integrate_pattern) + self.cp_ffn_right(current_pattern)

        # drop tf token
        final_integrate_pattern = final_integrate_pattern[:, 1:]
        # shape:(batch_size, num_sensors, pattern_length)
        return final_integrate_pattern


if __name__ == '__main__':
    pass
