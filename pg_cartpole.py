import tensorflow as tf
import numpy as np
import gym

num_inputs = 4
num_hidden = 4
num_outputs = 1

learning_rate =  0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeolder(tf.float32, shape=[None, num_inputs])
hidden_layer = tf.layers.dense(X, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)

logits = tf.layers.dense(hidden_layer, num_outputs)
outputs = tf.nn.sigmoid(logits)

probabilites = tf.concat(axis=1, values=[outputs,1-outputs])
action = tf.multinomial(probabilities, num_samples=1)

y = 1.0 - tf.to_float(action)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(lebels=y, logits=logits)
opt = tf.train.AdamOptimizer(learning_rate)

gradients_and_variables = opt.compute_gradients(cross_entropy)

gradients = list()
gradients_placeholders = []
grads_and_vars_feed = []

for gradient, variable in gradients_and_variables:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.get_shape())
    gradients_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = opt.apply_gradients(grads_and_vars_feed)
