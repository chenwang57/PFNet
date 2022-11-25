"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 17:55
# @Author  : Chen Wang
# @Site    : 
# @File    : st_fusion_module.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : 
*****************************************************
"""
import tensorflow as tf


# FeedForward class
class FeedForward(tf.keras.Model):
    def __init__(self, out_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(out_dim),
            tf.keras.layers.Dropout(dropout)
        ])

    def call(self, x, *args, **kwargs):
        return self.net(x)


"""
# Aggregation class
class Aggregation(tf.keras.Model):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        
        self.to_q = tf.keras.layers.Dense(self.embedding_size)
        self.to_k = tf.keras.layers.Dense(self.embedding_size)
        self.to_v = tf.keras.layers.Dense(self.embedding_size)

    def call(self, inputs, *args, **kwargs):
        x_temporal, x_spatial = inputs
        q = self.to_q(x_spatial)
        k = self.to_k(x_temporal)
        v = self.to_v(x_temporal)
        attn = tf.einsum('b i p, b j p -> b i j', q, k)
        attn = tf.nn.tanh(attn)
        attn = tf.nn.softmax(attn, axis=-1)
        return tf.einsum('b i s, b s j -> b i j', attn, v)
"""


# cross-attention class
class ProgressiveCrossAttention(tf.keras.Model):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.to_q = tf.keras.layers.Dense(self.embedding_size, use_bias=False)
        self.to_k = tf.keras.layers.Dense(self.embedding_size, use_bias=False)
        self.to_v = tf.keras.layers.Dense(self.embedding_size, use_bias=False)

    def call(self, inputs, *args, **kwargs):
        left, right = inputs
        q = self.to_q(left)
        k = self.to_k(right)
        v = self.to_v(right)
        cross_attn = tf.nn.softmax(tf.einsum('b i p, b j p -> b i j', q, k) * (self.embedding_size ** -0.5), axis=-1)
        out = tf.einsum('b i s, b s j -> b i j', cross_attn, v)
        return out + left


# AF process class
class AggregationAndFusionProcess(tf.keras.Model):
    def __init__(self, out_temporal_dim, embedding_dim, ffn_hidden_dim):
        super().__init__()
        self.out_temporal_dim = out_temporal_dim
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.cross_attn = ProgressiveCrossAttention(self.embedding_dim)
        self.cross_fusion = ProgressiveCrossAttention(self.out_temporal_dim)
        self.ffn_1 = FeedForward(self.out_temporal_dim, self.ffn_hidden_dim)
        self.ffn_2 = FeedForward(self.out_temporal_dim, self.ffn_hidden_dim)

    def call(self, inputs, *args, **kwargs):
        x_temporal, x_temporal_embedding, x_spatial_embedding = inputs
        x_fusion_embedding = self.cross_attn([x_temporal_embedding, x_spatial_embedding])
        x_fusion_out = self.ffn_1(x_fusion_embedding)
        x_progressive_fusion = self.cross_fusion([x_temporal, x_fusion_out])
        out = self.ffn_2(x_progressive_fusion)
        return out


# Layer-Wise Progressive Attention class
class LayerWiseProgressiveAttention(tf.keras.Model):
    def __init__(self, embedding_dim, ffn_hidden_dim, pattern_length):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.pattern_length = pattern_length

        self.embedding_layer = tf.keras.layers.Dense(embedding_dim, activation='tanh')
        self.af_process = AggregationAndFusionProcess(
            embedding_dim=self.embedding_dim,
            ffn_hidden_dim=self.ffn_hidden_dim,
            out_temporal_dim=self.pattern_length
        )

    def call(self, inputs, *args, **kwargs):
        x_temporal, x_spatial_embedding = inputs
        x_temporal_embedding = self.embedding_layer(x_temporal)
        x_fusion_features = self.af_process([x_temporal, x_temporal_embedding, x_spatial_embedding])
        return x_fusion_features


# Cascaded Progressive Attention(CPA)
class STFusion(tf.keras.Model):
    def __init__(self, pattern_length, embedding_dim, ffn_hidden_dim, progressive_depth, batch_size, num_for_predict):
        super().__init__()
        self.batch_size = batch_size
        self.pattern_length = pattern_length
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.progressive_depth = progressive_depth
        self.num_for_predict = num_for_predict

        self.layer_wise_progressive_attentions = []
        for i in range(progressive_depth):
            self.layer_wise_progressive_attentions.append(LayerWiseProgressiveAttention(
                embedding_dim=self.embedding_dim,
                ffn_hidden_dim=self.ffn_hidden_dim,
                pattern_length=self.pattern_length
            ))
        self.final_fusion = FeedForward(self.num_for_predict, self.ffn_hidden_dim)

    def call(self, inputs, *args, **kwargs):
        global fusion_features
        # x_temporal shape:(batch_size, sensor_num, pattern_length)
        # x_spatial shape:(batch_size, sensor_num, emb_size)
        x_temporal, x_spatial = inputs
        for i in range(self.progressive_depth):
            if i == 0:
                fusion_features = self.layer_wise_progressive_attentions[i]([x_temporal, x_spatial])
            else:
                fusion_features = self.layer_wise_progressive_attentions[i]([fusion_features, x_spatial])
        fusion_features = self.final_fusion(fusion_features)
        return fusion_features
