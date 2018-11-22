import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets, preprocessing, linear_model, metrics, pipeline, impute
from tensorflow.python.framework import ops
import requests
import os.path
import csv
import pandas as pd
import sys
import os
ops.reset_default_graph()
sess = tf.Session()

y_vals = pd.read_table("CancerTypes_y.txt", sep="\t",header=None)
x_vals_rna = pd.read_table("RNAseq_processed.txt", sep="\t",header=0)
x_vals_cnv = pd.read_table("CNV_processed.txt", sep="\t",header=0)
# remove geneID
x_vals_rna = x_vals_rna.ix[:,1:]
x_vals_cnv = x_vals_cnv.ix[:,1:]
# transpose
x_vals_rna  = x_vals_rna.transpose()
x_vals_cnv  = x_vals_cnv.transpose()


# convert pd dataframe to numpy array and remove nan sample
x_vals_cnv = x_vals_cnv.values
x_vals_rna = x_vals_rna.values
y_vals = y_vals.values

cnv_sample_to_keep = ~np.isnan(x_vals_cnv).any(axis=1)
rna_sample_to_keep = ~np.isnan(x_vals_rna).any(axis=1)

x_vals_rna_removed_nan = x_vals_rna[(cnv_sample_to_keep & rna_sample_to_keep)]
x_vals_cnv_removed_nan = x_vals_cnv[(cnv_sample_to_keep & rna_sample_to_keep)]
y_vals_removed_nan = y_vals[(cnv_sample_to_keep & rna_sample_to_keep)]
# normalize data
x_vals_rna_removed_nan = preprocessing.normalize(x_vals_rna_removed_nan, norm="l2")
x_vals_cnv_removed_nan = preprocessing.normalize(x_vals_cnv_removed_nan, norm="l2")

# impute missing value and standard scale
data_preprocess = pipeline.Pipeline([
        ('imputer', impute.SimpleImputer(strategy = "constant",fill_value= 0)),
        ('std_scaler', preprocessing.StandardScaler()),
    ])

x_vals_rna_processed = data_preprocess.fit_transform(x_vals_rna)
x_vals_cnv_processed = data_preprocess.fit_transform(x_vals_cnv)



#before
fig1, (ax1, ax2) = plt.subplots(1,2)
ax1.boxplot(x_vals_rna[0:30,0:30])
ax2.boxplot(x_vals_cnv[0:30,0:30])
plt.show()
#after
fig2, (ax3, ax4) = plt.subplots(1,2)
ax3.boxplot(x_vals_rna_processed[0:30,0:30])
ax4.boxplot(x_vals_cnv_processed[0:30,0:30])
plt.show()


print(len(x_vals_cnv))
print(x_vals_cnv.shape)
print(x_vals_rna.shape)
print(y_vals.shape)



# Select training and testing sets
# Set seed for reproducible results
seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)

def train_test(data, ratio = 0.8):
    train_indices = np.random.choice(len(data), round(len(data) * ratio), replace=False)
    test_indices = np.array(list(set(range(len(data))) - set(train_indices)))
    data_train = data[train_indices]
    data_test = data[test_indices]
    return data_train, data_test


# before imputation
# rna seq
x_vals_rna_train, x_vals_rna_test = train_test(x_vals_rna)

# cnv
x_vals_cnv_train, x_vals_cnv_test = train_test(x_vals_cnv)

# y vals
y_vals_train, y_vals_test = train_test(y_vals)


# removed nan
# rna seq
x_vals_rna_train_removed_nan, x_vals_rna_test_removed_nan = train_test(x_vals_rna_removed_nan)

# cnv
x_vals_cnv_train_removed_nan, x_vals_cnv_test_removed_nan = train_test(x_vals_cnv_removed_nan)

# y vals
y_vals_train_removed_nan, y_vals_test_removed_nan = train_test(y_vals_removed_nan)


# after imputation
# rna seq
x_vals_rna_train_processed, x_vals_rna_test_processed = train_test(x_vals_rna_processed)

# cnv
x_vals_cnv_train_processed, x_vals_cnv_test_processed = train_test(x_vals_cnv_processed)





def plot(loss_vec,train_acc,test_acc):
    # Plot loss over time
    plt.plot(loss_vec, 'k-')
    plt.title('Cross Entropy Loss per Generation  Logistic')
    plt.xlabel('Generation')
    plt.ylabel('Cross Entropy Loss')
    plt.show()

    # Plot train and test accuracy
    plt.plot(train_acc, 'k-', label='Train Set Accuracy')
    plt.plot(test_acc, 'r--', label='Test Set Accuracy')
    plt.title('Train and Test Accuracy Logistic')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

def run_model(train_x, train_y, test_x, test_y,round=500,c=0.1):
    # Initialize placeholders
    nFeatures = x_vals_cnv_train.shape[1]  # number of cnv/rna
    dimResponse = y_vals_train.shape[1]  # cancer or normal

    # Define inputs for session.run
    x_data = tf.placeholder(shape=[None, nFeatures],
                            dtype=tf.float32)  # tensor with nFeature columns. Note None takes any value when computation takes place
    y_target = tf.placeholder(shape=[None, dimResponse], dtype=tf.float32)  # tensor with 1 column
    # Initialize variables for regression: "y=sigmoid(A×x+b)"
    A = tf.Variable(tf.random_normal(shape=[nFeatures, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    # Declare model operations "y = A×x + b"
    model_output = tf.add(tf.matmul(x_data, A), b)
    # Declare loss function (Cross Entropy loss)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
    # Declare optimizer
    my_opt = tf.train.GradientDescentOptimizer(c)
    train_step = my_opt.minimize(loss)
    # Initialize global variables

    init = tf.global_variables_initializer()
    sess.run(init)
    # Actual Prediction
    prediction = tf.round(tf.sigmoid(model_output))  # model_output  y = 1 / (1 + exp(-x))
    predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)
    # Declare batch size
    batch_size = 25

    with tf.Session():
        loss_vec = []
        train_acc = []
        test_acc = []

        # Run training loop
        for i in range(round):  # on delta run range(5000)
            # get random batch from training set
            rand_index = np.random.choice(len(train_x), size=batch_size)
            rand_x = train_x[rand_index]
            rand_y = train_y[rand_index]
            # run train step
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
            # get loss
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            loss_vec.append(temp_loss)
            # get acc for training
            temp_acc_train = sess.run(accuracy, feed_dict={x_data: train_x, y_target: train_y})
            train_acc.append(temp_acc_train)
            # get acc for testing
            temp_acc_test = sess.run(accuracy, feed_dict={x_data: test_x, y_target: test_y})
            test_acc.append(temp_acc_test)
            # print running stat

        mean_test_acc = sum(test_acc[-200:]) / len(test_acc[-200:])
        print(mean_test_acc)
        plot(loss_vec, train_acc, test_acc)
        return loss_vec, train_acc, test_acc, mean_test_acc

def run_combined_model(train_x,train_x2, train_y, test_x, test_x2, test_y, round=500,c=0.01):
    # Initialize placeholders
    nFeatures = x_vals_cnv_train.shape[1]  # number of cnv/rna
    dimResponse = y_vals_train.shape[1]  # cancer or normal

    # Define inputs for session.run
    x_data = tf.placeholder(shape=[None, nFeatures],
                            dtype=tf.float32)  # tensor with nFeature columns. Note None takes any value when computation takes place
    x_data2 = tf.placeholder(shape=[None, nFeatures],
                            dtype=tf.float32)  # tensor with nFeature columns. Note None takes any value when computation takes place

    y_target = tf.placeholder(shape=[None, dimResponse], dtype=tf.float32)  # tensor with 1 column
    # Initialize variables for regression: "y=sigmoid(A×x_rna+C×x_cnv+b)"
    A = tf.Variable(tf.random_normal(shape=[nFeatures, 1]))
    C = tf.Variable(tf.random_normal(shape=[nFeatures, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    # Declare model operations "y = A×x_rna+C×x_cnv+b"
    model_output = tf.add(tf.add(tf.matmul(x_data, A), tf.matmul(x_data2, C)), b)
    # Declare loss function (Cross Entropy loss)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
    # Declare optimizer
    my_opt = tf.train.GradientDescentOptimizer(c)
    train_step = my_opt.minimize(loss)

    # Initialize global variables
    init = tf.global_variables_initializer()
    sess.run(init)
    # Actual Prediction
    prediction = tf.round(tf.sigmoid(model_output))  # model_output  y = 1 / (1 + exp(-x))
    predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)
    # Declare batch size
    batch_size = 25

    with tf.Session():
        loss_vec = []
        train_acc = []
        test_acc = []

        # Run training loop
        for i in range(round):  # on delta run range(5000)
            # get random batch from training set
            rand_index = np.random.choice(len(train_x), size=batch_size)
            rand_x = train_x[rand_index]
            rand_x2 = train_x2[rand_index]
            rand_y = train_y[rand_index]
            # run train step
            sess.run(train_step, feed_dict={x_data: rand_x, x_data2: rand_x2, y_target: rand_y})
            # get loss
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, x_data2: rand_x2, y_target: rand_y})
            loss_vec.append(temp_loss)
            # get acc for training
            temp_acc_train = sess.run(accuracy, feed_dict={x_data: train_x, x_data2: train_x2, y_target: train_y})
            train_acc.append(temp_acc_train)
            # get acc for testing
            temp_acc_test = sess.run(accuracy, feed_dict={x_data: test_x, x_data2: test_x2, y_target: test_y})
            test_acc.append(temp_acc_test)
            # print running stat

        mean_test_acc = sum(test_acc[-200:]) / len(test_acc[-200:])
        print(mean_test_acc)
        plot(loss_vec, train_acc, test_acc)
        return loss_vec, train_acc, test_acc, mean_test_acc


# logistic model for RNA seq removed nan
run_model(x_vals_rna_train_removed_nan, y_vals_train_removed_nan, x_vals_rna_test_removed_nan, y_vals_test_removed_nan)

# logistic model for cnv removed nan
run_model(x_vals_cnv_train_removed_nan, y_vals_train_removed_nan, x_vals_cnv_test_removed_nan, y_vals_test_removed_nan)

# logistic model for combined data removed nan
run_combined_model(x_vals_rna_train_removed_nan, x_vals_cnv_train_removed_nan, y_vals_train_removed_nan, x_vals_rna_test_removed_nan, x_vals_cnv_test_removed_nan, y_vals_test_removed_nan)

# logistic model for RNA seq with imputation
run_model(x_vals_rna_train_processed, y_vals_train_processed, x_vals_rna_test_processed, y_vals_test_processed)

# logistic model for cnv with imputation
run_model(x_vals_cnv_train_processed, y_vals_train_processed, x_vals_cnv_test_processed, y_vals_test_processed)

# logistic model for combined data with imputation
run_combined_model(x_vals_rna_train_processed, x_vals_cnv_train_processed, y_vals_train_processed,
                   x_vals_rna_test_processed, x_vals_cnv_test_processed, y_vals_test_processed)




# Elastic Net model
alpha = 0.1
enet = linear_model.ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(x_vals_cnv_train, y_vals_train).predict(x_vals_cnv_test)
r2_score_enet = metrics.r2_score(y_vals_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.plot(enet.coef_, color='lightgreen', linewidth=2,
         label='Elastic net coefficients')

plt.legend(loc='best')
plt.title("Elastic Net R^2: %f"
          % (r2_score_enet))
plt.show()
