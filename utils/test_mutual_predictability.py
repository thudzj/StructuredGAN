import tensorflow as tf
import numpy as np
from mlxtend.preprocessing import one_hot
import tensorflow.contrib as tc
from sklearn.svm import SVC

# z_dim = 100 #100
# y_dim = 10
# z = tf.placeholder(tf.float32, [None, z_dim])
# keep_prob = tf.placeholder(tf.float32)

# fc1 = tf.nn.dropout(z, keep_prob)
# fc1 = tc.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu)
# fc2 = tc.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu)

# fc3 = tc.layers.fully_connected(fc2, 512, activation_fn=tf.nn.relu)
# fc3_drop = tf.nn.dropout(fc3, keep_prob)
# logits_ = tc.layers.fully_connected(fc2, y_dim, activation_fn=None)

# y_ = tf.placeholder(tf.float32, [None, y_dim])
# cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=logits_)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# correct_prediction = tf.equal(tf.argmax(logits_,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#lis = ['../../ALI/features-ali-cifar.npz', '../../triple-gan/features-sgan-cifar.npz']
# lis = ['features-20.npz', 'features-50.npz', 'features-vae.npz', 'features-cvae.npz', 'features-ali.npz']
# lis = ['exp_dis/' + item for item in lis]
lis=[  'semi-cvae-mnist.npz', 'semi-cvae-svhn.npz', 'semi-cvae-cifar10.npz']#, 'vae-mnist.npz', 'vae-svhn.npz', 'vae-cifar10.npz']

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  for li in lis:
    data = np.load(li)
    features, labels = data['x'], data['y']
    labels = labels.astype(np.int32) + 1

    N=labels.shape[0]
    N_train = int(N/10*9)
    perm = np.arange(N)
    np.random.shuffle(perm)
    features = features[perm]
    #labels = one_hot(labels[perm])

    trainz = features[:N_train]
    testz = features[N_train:]
    trainy = labels[:N_train]
    testy = labels[N_train:]

    clf = SVC()
    clf.fit(trainz, trainy) 
    print(clf.score(trainz, trainy), clf.score(testz, testy))

    # sess.run(tf.global_variables_initializer())
    # for epoch in range(100):
    #   perm1 = np.arange(N_train)
    #   np.random.shuffle(perm1)
    #   trainz = trainz[perm1]
    #   trainy = trainy[perm1]
      
    #   for ite in range(int(N_train/100)):
    #     sess.run(train_step, feed_dict={z:trainz[ite*100:(ite+1)*100], y_:trainy[ite*100:(ite+1)*100], keep_prob:0.5})
    #   train_accuracy = sess.run(accuracy, feed_dict={z:trainz[ite*100:(ite+1)*100], y_:trainy[ite*100:(ite+1)*100], keep_prob: 1.0})
    #   test_accuracy = sess.run(accuracy, feed_dict={z:testz, y_:testy, keep_prob: 1.0})
    #   print("step %d, training accuracy %g, testing accuracy %g"%(epoch, train_accuracy, test_accuracy))
    




