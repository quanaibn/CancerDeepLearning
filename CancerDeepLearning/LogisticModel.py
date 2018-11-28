import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets, preprocessing, linear_model, metrics, pipeline, impute
from tensorflow.python.framework import ops
from keras import backend, models, layers, callbacks, optimizers
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

print(x_vals_cnv.shape)
print(x_vals_rna.shape)
print(y_vals.shape)

# assign class
x_vals_cnv = np.concatenate((x_vals_cnv, y_vals), axis=1)
x_vals_rna = np.concatenate((x_vals_rna, y_vals), axis=1)

# remove nan
cnv_sample_to_keep = ~np.isnan(x_vals_cnv).any(axis=1)
rna_sample_to_keep = ~np.isnan(x_vals_rna).any(axis=1)

x_vals_rna_removed_nan = x_vals_rna[(cnv_sample_to_keep & rna_sample_to_keep)]
x_vals_cnv_removed_nan = x_vals_cnv[(cnv_sample_to_keep & rna_sample_to_keep)]
y_vals_removed_nan = y_vals[(cnv_sample_to_keep & rna_sample_to_keep)]
# normalize data
x_vals_rna_removed_nan = preprocessing.normalize(x_vals_rna_removed_nan, norm="l2")
x_vals_cnv_removed_nan = preprocessing.normalize(x_vals_cnv_removed_nan, norm="l2")

print(x_vals_cnv_removed_nan.shape)

# remove all zeros column and row
column_to_keep_rna = ~np.all(x_vals_rna == 0, axis=1)
column_to_keep_cnv = ~np.all(x_vals_cnv == 0, axis=1)

x_vals_rna_processed = x_vals_rna[column_to_keep_cnv & column_to_keep_rna, :]
x_vals_cnv_processed = x_vals_cnv[column_to_keep_cnv & column_to_keep_rna, :]

row_to_keep_rna = ~np.all(x_vals_rna_processed == 0, axis=0)
row_to_keep_cnv = ~np.all(x_vals_cnv_processed == 0, axis=0)

x_vals_rna_processed = x_vals_rna_processed[:, row_to_keep_cnv & row_to_keep_rna]
x_vals_cnv_processed = x_vals_cnv_processed[:, row_to_keep_cnv & row_to_keep_rna]

# remove all column and row contain nan higher than threshold
threshold = 0.50

cnv_column_to_keep = np.isnan(x_vals_cnv_processed).sum(axis=1) < threshold * x_vals_cnv.shape[1]
rna_column_to_keep = np.isnan(x_vals_rna_processed).sum(axis=1) < threshold * x_vals_rna.shape[1]

x_vals_rna_processed = x_vals_rna_processed[cnv_column_to_keep & rna_column_to_keep, :]
x_vals_cnv_processed = x_vals_cnv_processed[cnv_column_to_keep & rna_column_to_keep, :]

cnv_row_to_keep = np.isnan(x_vals_cnv_processed).sum(axis=0) < threshold * x_vals_cnv.shape[0]
rna_row_to_keep = np.isnan(x_vals_rna_processed).sum(axis=0) < threshold * x_vals_cnv.shape[0]

x_vals_rna_processed = x_vals_rna_processed[:, cnv_row_to_keep & rna_row_to_keep]
x_vals_cnv_processed = x_vals_cnv_processed[:, cnv_row_to_keep & rna_row_to_keep]

print(x_vals_cnv_processed.shape)
print(x_vals_rna_processed.shape)

def impute_scale(data):
    # impute missing value and standard scale
    data_preprocess = pipeline.Pipeline([
        ('imputer', impute.SimpleImputer(strategy="constant", fill_value=0)),
        ('std_scaler', preprocessing.StandardScaler()),
    ])
    data_processed = data_preprocess.fit_transform(data)
    return data_processed


def DataCleanPlot():
    # before
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.boxplot(x_vals_rna[0:30, 0:30])
    ax2.boxplot(x_vals_cnv[0:30, 0:30])
    plt.show()
    # after
    fig2, (ax3, ax4) = plt.subplots(1, 2)
    ax3.boxplot(x_vals_rna_processed[0:30, 0:30])
    ax4.boxplot(x_vals_cnv_processed[0:30, 0:30])
    plt.show()


# Select training and testing sets
# Set seed for reproducible results
seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)


def train_test(data):
    train_indices = np.random.choice(len(data), round(len(data) * 0.7), replace=False)
    test_indices = np.array(list(set(range(len(data))) - set(train_indices)))
    data_train = data[train_indices]
    data_test = data[test_indices]
    data_train_x = data_train[:, :-1]
    data_test_x = data_test[:, :-1]
    data_train_y = data_train[:, [-1]]
    data_test_y = data_test[:, [-1]]
    return data_train_x, data_test_x, data_train_y, data_test_y


# before imputation
# rna seq
x_vals_rna_train, x_vals_rna_test,y_vals_train,y_vals_test = train_test(x_vals_rna)

# cnv
x_vals_cnv_train, x_vals_cnv_test, _, _ = train_test(x_vals_cnv)

# removed nan
# rna seq
x_vals_rna_train_removed_nan, x_vals_rna_test_removed_nan,y_vals_train_removed_nan,y_vals_test_removed_nan = train_test(x_vals_rna_removed_nan)

# cnv
x_vals_cnv_train_removed_nan, x_vals_cnv_test_removed_nan, _, _ = train_test(x_vals_cnv_removed_nan)



# after imputation
# rna seq
x_vals_rna_train_processed, x_vals_rna_test_processed, y_vals_train_processed, y_vals_test_processed = train_test(x_vals_rna_processed)

# cnv
x_vals_cnv_train_processed, x_vals_cnv_test_processed, _, _ = train_test(x_vals_cnv_processed)



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
    nFeatures = train_x.shape[1]  # number of cnv/rna
    dimResponse = train_y.shape[1]  # cancer or normal

    # Define inputs for session.run
    x_data = tf.placeholder(shape=[None, nFeatures],
                            dtype=tf.float32)  # tensor with nFeature columns. Note None takes any value when computation takes place
    y_target = tf.placeholder(shape=[None, dimResponse], dtype=tf.float32)  # tensor with 1 column
    # Initialize variables for regression: "y=sigmoid(Ax+b)"
    A = tf.Variable(tf.random_normal(shape=[nFeatures, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    # Declare model operations "y = Ax + b"
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
        predict_y = []
        true_y = []


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
            # get predicted y value
            temp_prediction = sess.run(prediction, feed_dict={x_data: test_x, y_target: test_y})
            predict_y.append(temp_prediction)
            true_y.append(test_y)

        mean_test_acc = sum(test_acc[-round//3:]) / len(test_acc[-round//3:])
        print("Accuracy:", mean_test_acc)
        plot(loss_vec, train_acc, test_acc)

        return true_y[round-1], predict_y[round-1]

def model_evaluation(true_y, predict_y):
    tn, fp, fn, tp = metrics.confusion_matrix(true_y, predict_y).ravel()
    fpr, tpr, t = metrics.roc_curve(true_y, predict_y)
    _, _, f1, support = metrics.precision_recall_fscore_support(true_y, predict_y)
    precisions, recalls, thresholds = metrics.precision_recall_curve(true_y, predict_y)
    print("True negative:", tn,"\n",
          "False negative:", fn,"\n",
          "False positive:", fp, "\n",
          "True positive:", tp, "\n",
          "False positive rates:", fpr, "\n",
          "True positive rates:", tpr, "\n",
          "Threshold:", thresholds, "\n",
          "Precision:", precisions, "\n",
          "Recall:", recalls, "\n",
          "F1:", f1, "\n",
          "Support:", support, "\n",
          )

    plt.plot(tpr, "b--", label="False positive rates")
    plt.plot(fpr, "g-", label="True positive rates")
    plt.title('False positive rates against true positive rates')
    plt.xlabel('False positive rates')
    plt.ylabel('True positive rates')
    plt.xlim(0, 1)
    plt.show()

    plt.plot(precisions, "b--", label="precision")
    plt.plot(recalls, "g-", label="recall")
    plt.xlim(0, 1)
    plt.show()




def run_combined_model(train_x,train_x2, train_y, test_x, test_x2, test_y, round=500,c=0.01):
    # Initialize placeholders
    nFeatures = train_x.shape[1]  # number of cnv/rna
    dimResponse = train_y.shape[1]  # cancer or normal

    # Define inputs for session.run
    x_data = tf.placeholder(shape=[None, nFeatures],
                            dtype=tf.float32)  # tensor with nFeature columns. Note None takes any value when computation takes place
    x_data2 = tf.placeholder(shape=[None, nFeatures],
                             dtype=tf.float32)  # tensor with nFeature columns. Note None takes any value when computation takes place

    y_target = tf.placeholder(shape=[None, dimResponse], dtype=tf.float32)  # tensor with 1 column
    # Initialize variables for regression: "y=sigmoid(Ax_rna+Cx_cnv+b)"
    A = tf.Variable(tf.random_normal(shape=[nFeatures, 1]))
    C = tf.Variable(tf.random_normal(shape=[nFeatures, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    # Declare model operations "y = Ax_rna+Cx_cnv+b"
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
        predict_y = []
        true_y = []

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
            # get predicted y
            temp_prediction = sess.run(prediction, feed_dict={x_data: test_x, x_data2: test_x2,  y_target: test_y})
            predict_y.append(temp_prediction)
            true_y.append(test_y)
        mean_test_acc = sum(test_acc[-round//3:]) / len(test_acc[-round//3:])
        print("Accuracy:", mean_test_acc)
        plot(loss_vec, train_acc, test_acc)

        return true_y[round-1], predict_y[round-1]


def LogisticModel():
    # logistic model for RNA seq removed nan
    y_true, y_predict = run_model(x_vals_rna_train_removed_nan, y_vals_train_removed_nan,
                                  x_vals_rna_test_removed_nan, y_vals_test_removed_nan, c=0.1)

    model_evaluation(y_true, y_predict)

    # logistic model for cnv removed nan
    y_true, y_predict = run_model(x_vals_cnv_train_removed_nan, y_vals_train_removed_nan,
                                  x_vals_cnv_test_removed_nan, y_vals_test_removed_nan, c=0.1)

    model_evaluation(y_true, y_predict)

    # logistic model for combined data removed nan
    y_true, y_predict = run_combined_model(x_vals_rna_train_removed_nan, x_vals_cnv_train_removed_nan,
                                           y_vals_train_removed_nan, x_vals_rna_test_removed_nan,
                                           x_vals_cnv_test_removed_nan, y_vals_test_removed_nan, c=0.1)

    model_evaluation(y_true, y_predict)

    # logistic model for RNA seq with imputation
    y_true, y_predict = run_model(x_vals_rna_train_processed, y_vals_train, x_vals_rna_test_processed, y_vals_test)

    model_evaluation(y_true, y_predict)

    # logistic model for cnv with imputation
    y_true, y_predict = run_model(x_vals_cnv_train_processed, y_vals_train, x_vals_cnv_test_processed, y_vals_test,
                                  c=0.01)

    model_evaluation(y_true, y_predict)

    # logistic model for combined data with imputation
    y_true, y_predict = run_combined_model(x_vals_rna_train_processed, x_vals_cnv_train_processed, y_vals_train,
                                           x_vals_rna_test_processed, x_vals_cnv_test_processed, y_vals_test, c=0.1)

    model_evaluation(y_true, y_predict)


def run_NN2Layer(train_x,train_x2, train_y, test_x, test_x2, test_y, round=500,c=0.01):
    # Initialize placeholders
    nFeatures = train_x.shape[1]  # number of cnv/rna
    dimResponse = train_y.shape[1]  # cancer or normal

    # Define inputs for session.run
    hidden_layer_nodes = 256
    x_data = tf.placeholder(shape=[None, nFeatures],
                            dtype=tf.float32)  # tensor with nFeature columns. Note None takes any value when computation takes place
    x_data2 = tf.placeholder(shape=[None, nFeatures],
                             dtype=tf.float32)  # tensor with nFeature columns. Note None takes any value when computation takes place

    y_target = tf.placeholder(shape=[None, dimResponse], dtype=tf.float32)  # tensor with 1 column
    # Initialize variables for regression: "y=sigmoid(Ax_rna+Cx_cnv+b)"
    # Layer 1
    A1 = tf.Variable(tf.random_normal(shape=[nFeatures, hidden_layer_nodes]))
    C1 = tf.Variable(tf.random_normal(shape=[nFeatures, hidden_layer_nodes]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
    # Layer 2
    A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
    C2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
    b2 = tf.Variable(tf.random_normal(shape=[1, 1]))
    # Layer 1 output
    hidden_output = tf.nn.relu(tf.add(tf.add(tf.matmul(x_data, A1), tf.matmul(x_data2, C1)), b1))
    # Declare model operations "y = Ax_rna+Cx_cnv+b"
    model_output = tf.add(tf.add(tf.matmul(hidden_output, A2), tf.matmul(hidden_output, C2)), b2)
    # Declare loss function (Cross Entropy loss)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
    # Declare optimizer
    my_opt = tf.train.GradientDescentOptimizer(c)
    train_step = my_opt.minimize(loss)

    # Initialize global variables
    init = tf.global_variables_initializer()
    # Actual Prediction
    prediction = tf.round(tf.sigmoid(model_output))  # model_output  y = 1 / (1 + exp(-x))
    predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)
    # Declare batch size
    batch_size = 25

    with tf.Session() as sess:
        sess.run(init)
        loss_vec = []
        train_acc = []
        test_acc = []
        predict_y = []
        true_y = []

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
            # get predicted y
            temp_prediction = sess.run(prediction, feed_dict={x_data: test_x, x_data2: test_x2, y_target: test_y})
            predict_y.append(temp_prediction)
            true_y.append(test_y)
        mean_test_acc = sum(test_acc[-round // 3:]) / len(test_acc[-round // 3:])
        print("Accuracy:", mean_test_acc)
        plot(loss_vec, train_acc, test_acc)

        return true_y[round - 1], predict_y[round - 1]


def NN2Layer():
    y_true, y_predict = run_NN2Layer(x_vals_rna_train_processed, x_vals_cnv_train_processed, y_vals_train_processed,
                                     x_vals_rna_test_processed, x_vals_cnv_test_processed, y_vals_test_processed,
                                     c=0.1 ** 5, round=2000)

    model_evaluation(y_true, y_predict)


# keras multi-layer
# merge two data set
merged_data = np.concatenate((x_vals_rna_processed[:, :-1], x_vals_cnv_processed), axis=1)
print(merged_data.shape)
x_train, x_test, y_train, y_test = train_test(merged_data)
x_train = impute_scale(x_train)
x_test = impute_scale(x_test)
model = models.Sequential([
    # first layer
    layers.Dense(512, input_dim=x_train.shape[1]),
    layers.Activation("relu"),
    layers.Dropout(0.2),
    # second layer
    layers.Dense(256),
    layers.Activation("relu"),
    layers.Dropout(0.2),
    # third layer
    layers.Dense(1),
    layers.Activation("sigmoid")
])

optimizer = optimizers.RMSprop(lr=0.01**5, rho=0.9, epsilon=None, decay=0.0)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

model.fit(x_train, y_train,
          epochs=50000,
          batch_size=20,
          callbacks=[callback])
score = model.evaluate(x_test, y_test, batch_size=20)
print(score)
# y_true = y_test.reshape((1, y_test.shape[0])).tolist()[0]
y_predict = model.predict_classes(x_test)
model_evaluation(y_test, y_predict)


def plot_NN(history: model.history):
    plt.plot(history.epoch, history.history["loss"], 'k-')
    #plt.plot(history.epoch, history.history["val_loss"], 'r--')
    plt.title('Cross Entropy Loss per epoch')
    plt.xlabel('epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.show()

    plt.plot(history.epoch, history.history["acc"], 'k-')
    #plt.plot(history.epoch, history.history["val_acc"], 'r--')
    plt.title('accuracy per epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


plot_NN(model.history)


