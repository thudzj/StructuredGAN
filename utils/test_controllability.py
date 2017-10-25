import tensorflow as tf
import numpy as np
from mlxtend.preprocessing import one_hot
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../zhijie/data/mnist', validation_size = 0, one_hot=True)

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(tcl.batch_norm(x), alpha)

def relu_batch_norm(x):
    return tf.nn.relu(tcl.batch_norm(x))

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 28,28,1])
dropout_keep = tf.placeholder(tf.float32)

conv1 = tcl.conv2d(
    x, 64, [4, 4], [2, 2],
    weights_initializer=tf.random_normal_initializer(stddev=0.02),
    activation_fn=tf.identity)
conv1 = leaky_relu_batch_norm(conv1)
conv1 = tcl.max_pool2d(conv1, kernel_size = [2, 2], stride = [1, 1])
conv1 = tf.nn.dropout(conv1, keep_prob = dropout_keep)
conv2 = tcl.conv2d(
    conv1, 128, [4, 4], [2, 2],
    weights_initializer=tf.random_normal_initializer(stddev=0.02),
    activation_fn=tf.identity)
conv2 = leaky_relu_batch_norm(conv2)
conv2 = tcl.max_pool2d(conv2, kernel_size = [2, 2], stride = [1, 1])
conv2 = tf.nn.dropout(conv2, keep_prob = dropout_keep)
conv3 = tcl.conv2d(conv2, 256, [4, 4], [2, 2],
    weights_initializer = tf.random_normal_initializer(stddev = 0.01),
    activation_fn = tf.identity)
conv3 = leaky_relu_batch_norm(conv3)
conv3 = tcl.max_pool2d(conv3, kernel_size = [2, 2], stride = [1, 1])
conv3 = tf.nn.dropout(conv3, keep_prob = dropout_keep)
conv4 = tcl.conv2d(conv3, 256, [4, 4], [2, 2],
    weights_initializer = tf.random_normal_initializer(stddev = 0.01),
    activation_fn = tf.identity)
conv4 = leaky_relu_batch_norm(conv4)
conv4 = tcl.flatten(conv4)
fc1 = tcl.fully_connected(
    conv4, 1024,
    weights_initializer=tf.random_normal_initializer(stddev=0.02),
    activation_fn=tf.identity)
fc1 = leaky_relu_batch_norm(fc1)
fc1 = tf.nn.dropout(fc1, keep_prob = dropout_keep)
logits_ = tcl.fully_connected(fc1, 10, activation_fn = None)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=logits_)
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits_,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


lis = ['generations-20-1.npz', 'generations-20-2.npz', 'generations-50-1.npz', 'generations-100-1.npz', '../../triple-gan/generations-triplegan-mnist-20-1.npz', \
 '../../triple-gan/generations-triplegan-mnist-20-2.npz','../../triple-gan/generations-triplegan-mnist-20-3.npz', '../../triple-gan/generations-triplegan-mnist-50-1.npz', '../../triple-gan/generations-triplegan-mnist-50-2.npz',\
 '../../triple-gan/generations-triplegan-mnist-50-3.npz','../../triple-gan/generations-triplegan-mnist-100-1.npz','../../triple-gan/generations-triplegan-mnist-100-2.npz', 'semi-cvae-generationsmnist2.npz',\
 'semi-cvae-generationsmnist5.npz', 'semi-cvae-generationsmnist10.npz', 'generations-fully-cvae.npz']

xs = []
ys = []
for li in lis:
  data = np.load(li)
  xs.append(data['x'])
  ys.append(one_hot(data['y'].astype(np.int32)))

for i in range(50000):
  batch = mnist.train.next_batch(50)
  if i % 1000 == 0:
    print(i)
    print("test set accuracy %g"%accuracy.eval(feed_dict={x: np.reshape(mnist.test.images, [-1, 28, 28, 1]), y_: mnist.test.labels, dropout_keep: 1.0}))
    for xb, yb in zip(xs, ys):
      print("accuracy %g"%accuracy.eval(feed_dict={x: np.reshape(xb, [-1, 28, 28, 1]), y_: yb, dropout_keep: 1.0}))
  train_step.run(feed_dict={x: np.reshape(batch[0], [50, 28, 28, 1]), y_: batch[1], dropout_keep: 0.5})
saver.save(sess, 'logs/mnist_clf_model.ckpt')