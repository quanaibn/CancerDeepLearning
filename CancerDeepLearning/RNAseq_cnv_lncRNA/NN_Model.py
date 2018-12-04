import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets, preprocessing, linear_model, metrics, pipeline, impute, svm
from tensorflow.python.framework import ops
from keras import backend, models, layers, callbacks, optimizers
import requests
import os.path
import csv
import pandas as pd
import sys
import os

ops.reset_default_graph()

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
print(sorted(y_dict.items(), key=lambda kv: kv[1], reverse=True))

y_class = y_vals
for x in range(0, y_vals.shape[0]):
    for y in range(0, y_vals.shape[1]):
        try:
            y_class[x,y] = y_vals[x, y].split(" ")[0]
        except:
            y_class[x,y] = y_vals[x, y]

unique, counts = np.unique(y_class, return_counts=True)
y_dict_processed = dict(zip(unique, counts))
print(sorted(y_dict_processed.items(), key=lambda kv: kv[1], reverse=True))

y_class_list = ["Carcinoma", "Adenocarcinoma", "Melanoma", "Lymphoma", "Leukemia", "Others"]

for x in range(0, y_class.shape[0]):
    for y in range(0, y_class.shape[1]):
        if y_class[x, y] not in y_class_list:
            y_class[x, y] = y_class_list[-1]
print(y_class)

y_class_num = y_class
for x in range(0, y_class_num.shape[0]):
    for y in range(0, y_class_num.shape[1]):
        y_class_num[x, y] = y_class_list.index(y_class_num[x, y])
print(y_class_num)

print(x_vals_cnv.shape)
print(x_vals_rna.shape)
print(x_vals_lnc.shape)
print(y_vals.shape)
print(y_vals_bin.shape)


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
        ('imputer', impute.SimpleImputer(strategy=impute_strategy)),
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

# Single data type processed data with binary class
rna_processed_cate = pre_processing((x_vals_rna, y_class_num))
cnv_processed_cate = pre_processing((x_vals_cnv, y_class_num))
lnc_processed_cate = pre_processing((x_vals_lnc, y_class_num))

# Combined processed data with binary class
rna_cnv_cate = pre_processing((x_vals_rna, x_vals_cnv, y_class_num))
rna_lnc_cate = pre_processing((x_vals_rna, x_vals_lnc, y_class_num))
cnv_lnc_cate = pre_processing((x_vals_cnv, x_vals_lnc, y_class_num))
rna_cnv_lnc_cate = pre_processing((x_vals_rna, x_vals_cnv, x_vals_lnc, y_class_num))
# Select training and testing sets
# Set seed for reproducible results
seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)


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


def logistic_model(data, rounds=500, lr=0.1**5):
    train_x,test_x, train_y, test_y = train_test(data)
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
    my_opt = tf.train.GradientDescentOptimizer(lr)
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
        loss_vec = []
        train_acc = []
        test_acc = []
        predict_y = []
        true_y = []

        sess.run(init)
        # Run training loop
        for i in range(rounds):  # on delta run range(5000)
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

        mean_test_acc = sum(test_acc[-rounds//3:]) / len(test_acc[-rounds//3:])
        print("Accuracy:", mean_test_acc)
        plot(loss_vec, train_acc, test_acc)

        return true_y[rounds-1], predict_y[rounds-1]


def model_evaluation(true_y, predict_y):
    tn, fp, fn, tp = metrics.confusion_matrix(true_y, predict_y).ravel()
    fpr, tpr, t = metrics.roc_curve(true_y, predict_y)
    _, _, f1, support = metrics.precision_recall_fscore_support(true_y, predict_y)
    precisions, recalls, thresholds = metrics.precision_recall_curve(true_y, predict_y)
    accuracy = metrics.accuracy_score(true_y, predict_y)
    print(" True negative:", tn,"\n",
          "False negative:", fn,"\n",
          "False positive:", fp, "\n",
          "True positive:", tp, "\n",
          "False positive rates:", fpr, "\n",
          "True positive rates:", tpr, "\n\n",
          "Accuracy:", accuracy, "\n",
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


# Logistic model
def run_logistic_model():
    # RNA seq
    y_true, y_pred = logistic_model(rna_processed_binary)
    model_evaluation(y_true, y_pred)

    # cnv
    y_true, y_pred = logistic_model(cnv_processed_binary)
    model_evaluation(y_true, y_pred)

    # lnc
    y_true, y_pred = logistic_model(lnc_processed_binary)
    model_evaluation(y_true, y_pred)

    # RNA seq and cnv
    y_true, y_pred = logistic_model(rna_cnv_binary)
    model_evaluation(y_true, y_pred)

    # RNA seq and lnc
    y_true, y_pred = logistic_model(rna_lnc_binary)
    model_evaluation(y_true, y_pred)

    # cnv and lnc
    y_true, y_pred = logistic_model(cnv_lnc_binary)
    model_evaluation(y_true, y_pred)

    # RNA seq, cnv and lnc
    y_true, y_pred = logistic_model(rna_cnv_lnc_binary)
    model_evaluation(y_true, y_pred)


# run_logistic_model()


def NN_model(data, rounds=500, c=0.01, dropout=0.80):
    train_x, test_x, train_y, test_y = train_test(data)
    # Initialize placeholders
    nFeatures = train_x.shape[1]  # number of cnv/rna
    dimResponse = train_y.shape[1]  # cancer or normal

    # Define inputs for session.run
    hidden_layer_nodes = 256
    x_data = tf.placeholder(shape=[None, nFeatures],
                            dtype=tf.float32)  # tensor with nFeature columns. Note None takes any value when computation takes place
    y_target = tf.placeholder(shape=[None, dimResponse], dtype=tf.float32)  # tensor with 1 column
    # Initialize variables for regression: "y=sigmoid(Ax_rna+Cx_cnv+b)"
    # Layer 1
    A1 = tf.Variable(tf.random_normal(shape=[nFeatures, hidden_layer_nodes]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
    # Layer 2
    A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
    b2 = tf.Variable(tf.random_normal(shape=[1, 1]))
    # Layer 1 output
    hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
    # dropout
    hidden_output = tf.nn.dropout(hidden_output, dropout)
    # Declare model operations "y = Ax_rna+Cx_cnv+b"
    model_output = tf.add(tf.matmul(hidden_output, A2), b2)
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
        for i in range(rounds):  # on delta run range(5000)
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
            # get predicted y
            temp_prediction = sess.run(prediction, feed_dict={x_data: test_x, y_target: test_y})
            predict_y.append(temp_prediction)
            true_y.append(test_y)
        mean_test_acc = sum(test_acc[-rounds // 3:]) / len(test_acc[-rounds // 3:])
        print("Accuracy:", mean_test_acc)
        plot(loss_vec, train_acc, test_acc)
        return true_y[rounds - 1], predict_y[rounds - 1]


def run_NN_model():
    # RNA seq
    y_true, y_pred = NN_model(rna_processed_binary)
    model_evaluation(y_true, y_pred)

    # cnv
    y_true, y_pred = NN_model(cnv_processed_binary)
    model_evaluation(y_true, y_pred)

    # lnc
    y_true, y_pred = NN_model(lnc_processed_binary)
    model_evaluation(y_true, y_pred)

    # RNA seq and cnv
    y_true, y_pred = NN_model(rna_cnv_binary)
    model_evaluation(y_true, y_pred)

    # RNA seq and lnc
    y_true, y_pred = NN_model(rna_lnc_binary)
    model_evaluation(y_true, y_pred)

    # cnv and lnc
    y_true, y_pred = NN_model(cnv_lnc_binary)
    model_evaluation(y_true, y_pred)

    # RNA seq, cnv and lnc
    y_true, y_pred = NN_model(rna_cnv_lnc_binary)
    model_evaluation(y_true, y_pred)


# run_NN_model()


def keras_NN(data, lr=0.01**5, epochs=100, batch_size=20):
    # keras multi-layer
    train_x, test_x, train_y, test_y = train_test(data)
    model = models.Sequential([
        # first layer
        layers.Dense(512, input_dim=train_x.shape[1]),
        layers.Activation("softmax"),
        layers.Dropout(0.2),
        # second layer
        layers.Dense(256),
        layers.Activation("relu"),
        layers.Dropout(0.2),
        # third layer
        layers.Dense(1),
        layers.Activation("sigmoid")
    ])

    optimizer = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

    model.fit(train_x, train_y,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[callback])
    score = model.evaluate(test_x, test_y, batch_size=20)
    print(score)
    # y_true = y_test.reshape((1, y_test.shape[0])).tolist()[0]
    y_predict = model.predict_classes(test_x)
    model_evaluation(test_y, y_predict)
    # plot
    plt.plot(model.history.epoch, model.history.history["loss"], 'k-')
    #plt.plot(history.epoch, history.history["val_loss"], 'r--')
    plt.title('Cross Entropy Loss per epoch')
    plt.xlabel('epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.show()

    plt.plot(model.history.epoch, model.history.history["acc"], 'k-')
    #plt.plot(history.epoch, history.history["val_acc"], 'r--')
    plt.title('accuracy per epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


def run_keras_NN():
    # RNA seq
    keras_NN(rna_processed_binary)

    # cnv
    keras_NN(cnv_processed_binary)

    # lnc
    keras_NN(lnc_processed_binary)

    # RNA seq and cnv
    keras_NN(rna_cnv_binary)

    # RNA seq and lnc
    keras_NN(rna_lnc_binary)

    # cnv and lnc
    keras_NN(cnv_lnc_binary)

    # RNA seq, cnv and lnc
    keras_NN(rna_cnv_lnc_binary)


# run_keras_NN()


def svm_model(data, kernel='sigmoid', nu=0.6):
    model = svm.NuSVC(kernel=kernel, nu=nu)
    train_x, test_x, train_y, test_y = train_test(data)
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    model_evaluation(test_y, pred_y)


def run_svm_model():
    # RNA seq
    svm_model(rna_processed_binary)

    # cnv
    svm_model(cnv_processed_binary)

    # lnc
    svm_model(lnc_processed_binary)

    # RNA seq and cnv
    svm_model(rna_cnv_binary)

    # RNA seq and lnc
    svm_model(rna_lnc_binary)

    # cnv and lnc
    svm_model(cnv_lnc_binary)

    # RNA seq, cnv and lnc
    svm_model(rna_cnv_lnc_binary)


# run_svm_model()

def DNN_model(data):
    train_x, test_x, train_y, test_y = train_test(data)
    # feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(train_x[1]))]
    x_data = tf.placeholder(shape=[None, train_x.shape[1]],
                            dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, train_y.shape[1]], dtype=tf.float32)
    estimator = tf.estimator.DNNClassifier(
        feature_columns=x_data,
        hidden_units=[1024, 512, 256],
        n_classes=6,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

    def input_fn_train():  # returns x, y
        pass

    # Define the training inputs
    input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x={x_data: train_x},
        y={y_target: train_y},
        batch_size=128,
        num_epochs=1,
        shuffle=True,
        queue_capacity=1000,
        num_threads=1
    )
    estimator.train(input_fn=input_fn_train, steps=100)

    def input_fn_eval():  # returns x, y
        return test_x, test_y

    # metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)

    def input_fn_predict():  # returns x, None
        pass

    input_fn_predict = tf.estimator.inputs.numpy_input_fn(
        x={x_data: test_x},
        y=None,
        batch_size=128,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000,
        num_threads=1
    )

    predictions = estimator.predict(input_fn=input_fn_predict)

    model_evaluation(test_y, predictions)


def run_DNN_model():

    DNN_model(rna_cnv_lnc_cate)


# run_DNN_model()

def decode_one_hot(batch_of_vectors):
    nonzero_indices = tf.where(tf.not_equal(
        batch_of_vectors, tf.zeros_like(batch_of_vectors)))
    reshaped_nonzero_indices = tf.reshape(
        nonzero_indices[:, -1], tf.shape(batch_of_vectors)[:-1])
    return reshaped_nonzero_indices


def NN_model_cate(data, rounds=100, c=0.1**3, dropout=0.80):
    train_x, test_x, train_y, test_y = train_test(data)
    train_y = tf.one_hot(indices=np.squeeze(train_y), depth=len(y_class_list))
    test_y = tf.one_hot(indices=np.squeeze(test_y), depth=len(y_class_list))
    # Initialize placeholders
    nFeatures = train_x.shape[1]  # number of cnv/rna
    dimResponse = train_y.shape[1] #

    # Define inputs for session.run
    hidden_layer_nodes = 256
    x_data = tf.placeholder(shape=[None, nFeatures],
                            dtype=tf.float32)  # tensor with nFeature columns. Note None takes any value when computation takes place
    y_target = tf.placeholder(shape=[None, dimResponse], dtype=tf.float32)  # tensor with 1 column
    # Initialize variables for regression: "y=sigmoid(Ax_rna+Cx_cnv+b)"
    # Layer 1
    A1 = tf.Variable(tf.random_normal(shape=[nFeatures, hidden_layer_nodes]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
    # Layer 2
    A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 6]))
    b2 = tf.Variable(tf.random_normal(shape=[1, 6]))
    # Layer 1 output
    hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
    # dropout
    hidden_output = tf.nn.dropout(hidden_output, dropout)
    # Declare model operations "y = Ax_rna+Cx_cnv+b"
    model_output = tf.add(tf.matmul(hidden_output, A2), b2)
    # Declare loss function (Cross Entropy loss)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))
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
        for i in range(rounds):  # on delta run range(5000)
            # get random batch from training set
            rand_index = np.random.choice(len(train_x), size=batch_size)
            rand_x = train_x[rand_index]
            rand_y = train_y.eval()[rand_index]
            # run train step
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
            # get loss
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            loss_vec.append(temp_loss)
            # get acc for training
            temp_acc_train = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
            train_acc.append(temp_acc_train)
            # get acc for testing
            temp_acc_test = sess.run(accuracy, feed_dict={x_data: test_x, y_target: test_y.eval()})
            test_acc.append(temp_acc_test)
            # get predicted y
            temp_prediction = sess.run(prediction, feed_dict={x_data: test_x, y_target: test_y.eval()})
            predict_y.append(temp_prediction)
            true_y.append(test_y)
        mean_test_acc = sum(test_acc[-rounds // 3:]) / len(test_acc[-rounds // 3:])
        print("Accuracy:", mean_test_acc)
        plot(loss_vec, train_acc, test_acc)
        print(true_y[rounds - 1].shape)
        print(predict_y[rounds - 1].shape)

        return decode_one_hot(true_y[rounds - 1]).eval(), decode_one_hot(predict_y[rounds - 1]).eval()


def run_NN_model_cate():
    test_y, predictions = NN_model_cate(rna_cnv_lnc_cate)
    print(test_y)
    print(predictions)
    model_evaluation(test_y, predictions)

run_NN_model_cate()
