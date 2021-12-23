import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size,embedded_chars, filter_sizes, num_filters, labels, dropout_rate):

        '''self.embedded_chars = embedded_chars
        self.hidden_sizes = hidden_sizes
        self.labels = labels
        self.num_label = num_label
        self.dropout_rate = dropout_rate
        self.max_len = max_len
        self.embedding_size = embedded_chars.shape[-1].value'''

        # Placeholders for input, output and dropout
        #self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        #self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        #self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        self.sequence_length = sequence_length
        #l2_loss = tf.constant(0.0)
        self.labels = labels
        self.vocab_size = vocab_size
        self.embedding_size = embedded_chars.shape[-1].value
        self.embedded_chars =embedded_chars
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters


    def textcnn(self):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
            #self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_rate)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            #self.l2_loss += tf.nn.l2_loss(W)
            #self.l2_loss += tf.nn.l2_loss(b)
            logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            one_hot_labels = tf.one_hot(self.labels, depth= self.num_classes, dtype=tf.float32)
            #self.predictions = tf.argmax(self.logits , 1, name="predictions")

        return (logits, one_hot_labels)

    '''def gen_result(self, labels, num_classes):
        #output, output_size = self.__init__()
        logits, one_hot_labels = self.__init__(output, output_size, labels, num_classes)
        # loss = self._cal_loss(logits)

        # predictions = self._get_prediction(logits)
        # tf.logging.info("predictions: {}".format(predictions))

        return (logits, one_hot_labels)'''
