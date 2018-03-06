import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import numpy as np
import math
import tqdm

# TODO: Load traffic signs data.
with open('train.p', 'rb') as pfile:
    data = pickle.load(pfile)
    features = data['features']
    labels = data['labels']
    del data
    
n_samples = 10000
features = features[:n_samples]
labels = labels[:n_samples]
n_labels = len(np.unique(labels))
    
# TODO: Split data into training and validation sets.
x_train, x_valid, y_train, y_valid = \
    train_test_split(features, labels, test_size=0.2, random_state=0)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, (None))
x_resized = tf.image.resize_images(x, (227, 227))

learning_rate = 0.001

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x_resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
fc7_shape = fc7.get_shape().as_list()[-1]
fc8_w = tf.Variable(tf.truncated_normal((fc7_shape, n_labels), stddev=1e-2))
fc8_b = tf.Variable(tf.zeros(n_labels))
logits = tf.matmul(fc7, fc8_w) + fc8_b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                               labels=y)
loss = tf.reduce_mean(cross_entropy)

preds = tf.arg_max(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, 
                                  var_list=[fc8_w, fc8_b])
init = tf.global_variables_initializer()

# TODO: Train and evaluate the feature extraction model.
epochs = 10
batch_size = 32
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        x_train0, y_train0 = shuffle(x_train, y_train)
        n_batches = int(math.ceil(x_train.shape[0]/batch_size))
        
        for batch in range(n_batches):
            x_batch = x_train0[batch * batch_size:(batch + 1) * batch_size]
            y_batch = y_train0[batch * batch_size:(batch + 1) * batch_size]
            _, l, acc = sess.run([optimizer, loss, accuracy], 
                                 feed_dict={x: x_batch, y: y_batch})
            print('\rbatch {:>2}/{}, loss: {:>.3f}, accuracy: {:>.3f}'.format(
                    batch+1, n_batches, l, acc), flush=True, end='')
        print()
        acc = sess.run(accuracy, feed_dict={x: x_valid,
                                            y: y_valid})
        print('Epoch {:>1}/{}, acc = {:>.4f}'.format(epoch, epochs, acc))
        print()
        
        