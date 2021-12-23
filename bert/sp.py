from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy

import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F

tf.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)

class Similarity(object):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""

    def __init__(self, embedding,logits):

        self.embedding = embedding
        self.logits = logits

    def similarity_loss(self):
        f_s = self.embedding
        f_t = self.logits

        f_s = tf.reshape(f_s,[f_s.shape[0], -1])
        f_t = tf.reshape(f_t,[f_s.shape[0], -1])


        G_s = tf.matmul(f_s, tf.transpose(f_s))
        #G_s = tf.matmul(f_s, f_s)
        # G_s = G_s / G_s.norm(2)
        G_s = tf.math.l2_normalize(G_s, axis=1)
        G_t = tf.matmul(f_t, tf.transpose(f_t))
        #G_t = tf.matmul(f_t, f_t)
        # G_t = G_t / G_t.norm(2)
        G_t = tf.math.l2_normalize(G_t, axis=1)

        G_diff = G_t - G_s
        bsz = tf.cast(f_s.shape[0],dtype=float)


        sp_loss = tf.reduce_sum(tf.reshape(G_diff * G_diff,[-1, 1]),axis=1) / (bsz* bsz)
        print("the type of sp_loss is: ",type(sp_loss))
        print("the shape of sp_loss is: ", type(sp_loss))
        return sp_loss

    '''def get_result(self):
        sp_loss = self.similarity_loss()
        return sp_loss'''
    '''
    def similarity_loss(self, f_s, f_t):

        print("345678")
        bsz = f_s.shape[0]
        print("the bsz is: ", bsz)
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s, dim=1)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t, dim=1)

        G_diff = G_t - G_s
        sp_loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return sp_loss'''
