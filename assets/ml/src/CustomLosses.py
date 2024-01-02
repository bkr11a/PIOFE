__author__ = "Brad Rice"
__version__ = 0.1

import tensorflow as tf

class EPE_Loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @tf.function
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.norm(y_true-y_pred, ord='euclidean'))