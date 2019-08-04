#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: hbasriaykol

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data
mnist= input_data.read_data_sets("data/MNIST", one_hot=True, reshape=False)

x=tf.placeholder(tf.float32, [None,28,28,1])
y_true=tf.placeholder(tf.float32,[None,10])

filter_1 = 16
filter_2 = 32
layer_out = 10

w_1=tf.Variable(tf.truncated_normal([5,5,1, filter_1], stddev=0.1))
b_1=tf.Variable(tf.constant(0.1, shape = [filter_1]))
w_2=tf.Variable(tf.truncated_normal([5,5,filter_1,filter_2], stddev=0.1))
b_2=tf.Variable(tf.constant(0.1, shape = [filter_2]))

w_3=tf.Variable(tf.truncated_normal([7*7*filter_2, 256], stddev=0.1))
b_3=tf.Variable(tf.constant(0.1, shape = [256]))

w_4=tf.Variable(tf.truncated_normal([256,10], stddev=0.1))
b_4=tf.Variable(tf.constant(0.1, shape = [10]))


y1 = tf.nn.relu(tf.nn.conv2d(x, w_1, strides=[1,1,1,1], padding = 'SAME')+ b_1) #output = [28,28,16]
y1 = tf.nn.max_pool(y1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME' ) #output=[14,14,16]
y2 = tf.nn.relu(tf.nn.conv2d(y1, w_2, strides=[1,1,1,1], padding = 'SAME')+ b_2) #output = [14,14,32]
y2 = tf.nn.max_pool(y2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME' ) #output=[7,7,32]

flattened = tf.reshape(y2, shape=[-1,7*7*filter_2])
y3=tf.nn.relu(tf.matmul(flattened,w_3)+ b_3)
logits= tf.matmul(y3,w_4)+ b_4
y4=tf.nn.softmax(logits)

correct_prediction=tf.equal(tf.argmax(y4 , 1), tf.argmax(y_true , 1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

xent= tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = y_true)
loss=tf.reduce_mean(xent)

#Optimize , learning Rate = 0.5
optimize = tf.train.AdamOptimizer(5e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size=128


loss_graph= []
def training_step(iterations):
    for i in range(iterations):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true: y_batch}
        [_, train_loss] = sess.run([optimize,loss], feed_dict = feed_dict_train)
        loss_graph.append(train_loss)
        if i % 100 == 0:
            train_acc = sess.run(accuracy, feed_dict=feed_dict_train)
            print('Iteration:', i, 'Training accuracy:', train_acc, 'Training loss:', train_loss)


def test_accuracy():
    feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
    acc= sess.run(accuracy, feed_dict= feed_dict_test)
    print('Testing accuracy:', acc)
    
training_step(1000)
test_accuracy()

plt.plot(loss_graph, 'k-')
plt.title('Loss Graph')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()




