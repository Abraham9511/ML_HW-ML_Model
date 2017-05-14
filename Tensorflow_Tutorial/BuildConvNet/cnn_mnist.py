from __future__ import absolute_import
from __future__ import division
from __futrue__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(feature, labels, model):
    #Input Layer
    input_layer = tf.reshape(feature)

if __name__ == "__main__":
    tf.app.run()
