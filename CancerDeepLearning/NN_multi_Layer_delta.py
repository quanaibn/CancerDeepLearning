import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets, preprocessing, linear_model, metrics, pipeline
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
        ('imputer', preprocessing.Imputer(strategy="mean")),
        ('std_scaler', preprocessing.StandardScaler()),
    ])
    data_processed = data_preprocess.fit_transform(data)
    return data_processed


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


def plot_NN(history):
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


# keras multi-layer
# merge two data set
merged_data = np.concatenate((x_vals_rna_processed[:, :-1], x_vals_cnv_processed), axis=1)
# remove nan
merged_data = merged_data[~np.isnan(merged_data).any(axis=1)]
print(merged_data.shape)
x_train, x_test, y_train, y_test = train_test(merged_data)
# impute nan
x_train = impute_scale(x_train)
x_test = impute_scale(x_test)


model = models.Sequential([
    # first layer
    layers.Dense(1024, input_dim=x_train.shape[1]),
    layers.Activation("softmax"),
    layers.Dropout(0.4),
    # second layer
    layers.Dense(256),
    layers.Activation("relu"),
    layers.Dropout(0.4),
    # third layer
    layers.Dense(1),
    layers.Activation("tanh")
])

optimizer = optimizers.RMSprop(lr=0.1**5, rho=0.9, epsilon=None, decay=0.0)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

model.fit(x_train, y_train,
          epochs=500,
          batch_size=20,
          callbacks=[callback])
score = model.evaluate(x_test, y_test, batch_size=20)
print(score)
# y_true = y_test.reshape((1, y_test.shape[0])).tolist()[0]
y_predict = model.predict_classes(x_test)
model_evaluation(y_test, y_predict)

plot_NN(model.history)

