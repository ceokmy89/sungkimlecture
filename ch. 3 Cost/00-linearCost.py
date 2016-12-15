import tensorflow as tf
from scipy._lib.six import xrange

# data set

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# try to find values for w and b that compute y_data = W * x_data + b
# range is -100 ~ 100
W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# my hypothesis
hypothesis = W * X

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X), ))    #0.1: learning rate
update = W.assign(descent)

# before starting, initialize the variables
init = tf.initialize_all_variables()

# launch
sess = tf.Session()
sess.run(init)

# fit the line
for step in xrange(100):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))