import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets, preprocessing, linear_model, metrics, pipeline, svm
from tensorflow.python.framework import ops
from keras import backend, models, layers, callbacks, optimizers
import requests
import os.path
import csv
import pandas as pd
import sys
import os
import ray
import ray.tune as tune
import argparse
import time
from ray.tune import grid_search, run_experiments, register_trainable, Trainable
from ray.tune.schedulers import HyperBandScheduler


activation_fn = None  # e.g. tf.nn.relu
# Select training and testing sets
# Set seed for reproducible results
seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)
y_vals_bin = pd.read_table("CancerTypes_y_BinaryClass.txt", sep="\t",header=None)
y_vals = pd.read_table("CancerTypes_y_multiClass.txt", sep="\t",header=None)
x_vals_rna = pd.read_table("RNAseq_processed_multiClass.txt", sep="\t",header=0)
x_vals_cnv = pd.read_table("CNV_processed_multiClass.txt", sep="\t",header=0)
x_vals_lnc = pd.read_table("lncRNA_processed_multiClass.txt", sep="\t",header=0)

# remove geneID
x_vals_rna = x_vals_rna.ix[:,1:]
x_vals_cnv = x_vals_cnv.ix[:,1:]
x_vals_lnc = x_vals_lnc.ix[:,1:]
# transpose
x_vals_rna = x_vals_rna.transpose()
x_vals_cnv = x_vals_cnv.transpose()
x_vals_lnc = x_vals_lnc.transpose()

# convert pd data frame to numpy array and remove nan sample
x_vals_cnv = x_vals_cnv.values
x_vals_rna = x_vals_rna.values
x_vals_lnc = x_vals_lnc.values
y_vals = y_vals.values

#
unique, counts = np.unique(y_vals, return_counts=True)
y_dict = dict(zip(unique, counts))

y_class = y_vals
for x in range(0, y_vals.shape[0]):
    for y in range(0, y_vals.shape[1]):
        try:
            y_class[x,y] = y_vals[x, y].split(" ")[0]
        except:
            y_class[x,y] = y_vals[x, y]


unique, counts = np.unique(y_class, return_counts=True)
y_dict_processed = dict(zip(unique, counts))
y_class_list = ["Carcinoma", "Adenocarcinoma", "Melanoma", "Lymphoma", "Leukemia", "Others"]

for x in range(0, y_class.shape[0]):
    for y in range(0, y_class.shape[1]):
        if y_class[x, y] not in y_class_list:
            y_class[x, y] = y_class_list[-1]


y_class_num = y_class
for x in range(0, y_class_num.shape[0]):
    for y in range(0, y_class_num.shape[1]):
        y_class_num[x, y] = y_class_list.index(y_class_num[x, y])


def train_test(data, rate=0.7):
    train_indices = np.random.choice(len(data), round(len(data) * rate), replace=False)
    test_indices = np.array(list(set(range(len(data))) - set(train_indices)))
    data_train = data[train_indices]
    data_test = data[test_indices]
    data_train_x = data_train[:, :-1]
    data_test_x = data_test[:, :-1]
    data_train_y = data_train[:, [-1]]
    data_test_y = data_test[:, [-1]]
    return data_train_x, data_test_x, data_train_y, data_test_y


def pre_processing(data_sets:tuple, threshold=0.5, impute_strategy="median"):
    # merge multi data sets
    new_data = np.concatenate(data_sets, axis=1)
    # remove all zeros column and row
    new_data = new_data[~np.all(new_data == 0, axis=1), :]
    new_data = new_data[:, ~np.all(new_data == 0, axis=0)]
    # remove all column and row contain nan higher than threshold
    # new_data = new_data[np.isnan(new_data).sum(axis=1) < threshold * new_data[:, :-1].shape[1], :]
    # new_data = new_data[:, np.isnan(new_data).sum(axis=0) < threshold * new_data[:, :-1].shape[0]]
    # impute missing value and standard scale
    data_preprocess = pipeline.Pipeline([
        ('imputer', preprocessing.Imputer(strategy=impute_strategy)),
        ('std_scaler', preprocessing.StandardScaler()),
    ])
    data_processed = data_preprocess.fit_transform(new_data[:, :-1])
    data_processed = np.concatenate((data_processed, new_data[:, [-1]]), axis=1)
    return data_processed


# Single data type processed data with binary class
rna_processed_binary = pre_processing((x_vals_rna, y_vals_bin))
cnv_processed_binary = pre_processing((x_vals_cnv, y_vals_bin))
lnc_processed_binary = pre_processing((x_vals_lnc, y_vals_bin))

# Combined processed data with binary class
rna_cnv_binary = pre_processing((x_vals_rna, x_vals_cnv, y_vals_bin))
rna_lnc_binary = pre_processing((x_vals_rna, x_vals_lnc, y_vals_bin))
cnv_lnc_binary = pre_processing((x_vals_cnv, x_vals_lnc, y_vals_bin))
rna_cnv_lnc_binary = pre_processing((x_vals_rna, x_vals_cnv, x_vals_lnc, y_vals_bin))

# Single data type processed data with categorical class
rna_processed_cate = pre_processing((x_vals_rna, y_class_num))
cnv_processed_cate = pre_processing((x_vals_cnv, y_class_num))
lnc_processed_cate = pre_processing((x_vals_lnc, y_class_num))

# Combined processed data with categorical class
rna_cnv_cate = pre_processing((x_vals_rna, x_vals_cnv, y_class_num))
rna_lnc_cate = pre_processing((x_vals_rna, x_vals_lnc, y_class_num))
cnv_lnc_cate = pre_processing((x_vals_cnv, x_vals_lnc, y_class_num))
rna_cnv_lnc_cate = pre_processing((x_vals_rna, x_vals_cnv, x_vals_lnc, y_class_num))

train_x, test_x, train_y, test_y = train_test(rna_cnv_lnc_cate)
nFeatures = train_x.shape[1]
dimResponse = len(y_class_list)
train_y = tf.one_hot(indices=np.squeeze(train_y), depth=dimResponse)
test_y = tf.one_hot(indices=np.squeeze(test_y), depth=dimResponse)


def setupCNN(x,nFeatures, dimResponse, hidden_layer1_nodes):
    """setupCNN builds the graph for a deep net for classifying digits.
    Args:
        x: an input tensor with the dimensions (N_examples, features), where features is
        the number of columns.
    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 6), with
        values equal to the logits of classifying the digit into one of 6
        classes (the digits 0-5). keep_prob is a scalar placeholder for the
        probability of dropout.
    """

    # First layer.
    with tf.name_scope('layer1'):
        W_1 = weight_variable([nFeatures, hidden_layer1_nodes])
        b_1 = bias_variable([hidden_layer1_nodes])
        h_1 = activation_fn(tf.add(tf.matmul(x, W_1), b_1))

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_1_drop = tf.nn.dropout(h_1, keep_prob)

    # Second layer.
    with tf.name_scope('layer2'):
        W_2 = weight_variable([hidden_layer1_nodes, dimResponse])
        b_2 = bias_variable([1, dimResponse])
        y_conv = tf.matmul(h_1_drop, W_2) + b_2

    return y_conv, keep_prob


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class TrainModel(Trainable):
    """Example Model trainable."""

    def _setup(self, config):
        global activation_fn
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.nFeatures = nFeatures
        self.dimResponse = dimResponse
        self.x = tf.placeholder(tf.float32, [None, self.nFeatures])
        self.y_ = tf.placeholder(tf.float32, [None, self.dimResponse])
        self.batch_size = 25
        activation_fn = getattr(tf.nn, config['activation'])

        # Build the graph for the deep net
        y_conv, self.keep_prob = setupCNN(self.x, self.nFeatures, self.dimResponse, config['hidden_layer_nodes'])

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(
                config['learning_rate']).minimize(cross_entropy)

        self.train_step = train_step

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.iterations = 0
        self.saver = tf.train.Saver()

    def _train(self):
        for i in range(10):
            rand_index = np.random.choice(len(self.train_x), size=self.batch_size)
            rand_x = self.train_x[rand_index]
            rand_y = self.train_y.eval()[rand_index]
            self.sess.run(
                self.train_step,
                feed_dict={
                    self.x: rand_x,
                    self.y_: rand_y,
                    self.keep_prob: 0.7
                })

        test_accuracy = self.sess.run(
            self.accuracy,
            feed_dict={
                self.x: self.test_x,
                self.y_: self.test_y,
                self.keep_prob: 1.0
            })

        self.iterations += 1
        return {"mean_accuracy": test_accuracy}

    def _save(self, checkpoint_dir):
        return self.saver.save(
            self.sess, checkpoint_dir + "/save", global_step=self.iterations)

    def _restore(self, path):
        return self.saver.restore(self.sess, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    args, _ = parser.parse_known_args()

    register_trainable("my_class", TrainModel)
    module_spec = {
        'run': 'my_class',
        'stop': {
            'mean_accuracy': 0.99,
            'training_iteration': 100
        },
        'config': {
            'learning_rate': lambda spec: 10**np.random.uniform(-5, -1),
            'activation': grid_search(['relu', 'elu', 'tanh', 'sigmoid', 'softmax']),
            'hidden_layer_nodes': grid_search([8, 16, 32, 64, 128, 256, 512])
        },
        "num_samples": 10
    }

    if args.smoke_test:
        module_spec['stop']['training_iteration'] = 2
        module_spec['num_samples'] = 2

    ray.init()
    hyperband = HyperBandScheduler(
        time_attr="training_iteration", reward_attr="mean_accuracy", max_t=1800)

    run_experiments({'hyperband_test': module_spec}, scheduler=hyperband)



