import subprocess
subprocess.call("tensorflowpython", shell = True)

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

y = pd.read_table('CancerTypes_y.txt', sep = '\t', header = None)
x_cnv = pd.read_table('CNV_processed.txt', sep = '\t', header = 0)
x_rna = pd.read_table('RNAseq_processed.txt', sep = '\t', header = 0)

# Remove GeneID Column
x_cnv = x_cnv.drop('GeneID', axis = 1)
x_rna = x_rna.drop('GeneID', axis = 1)

# Transpose
x_cnv = x_cnv.transpose()
x_rna = x_rna.transpose()
print('x_cnv shape is:', x_cnv.shape)
print('x_rna shape is:', x_rna.shape)
print('y shape is:', y.shape)

#DropNa from columns with at least 50% NaN
x_cnv_dropped = x_cnv.dropna(thresh = 0.5*len(x_cnv.index))

#Columns dropped
print('Dropped {0} Columns'.format(len(x_cnv.columns) - len(x_cnv_dropped.columns)))

#Impute
impute_median = Imputer(strategy = 'median')
x_cnv_imputed = impute_median.fit_transform(x_cnv_dropped)
x_cnv_imputed = pd.DataFrame(x_cnv_imputed)

#Check for NaN values
np.isnan(x_cnv_imputed).all().any()

def count_all_zeros (df, _axis_ = 0):
    return len(df.columns) - np.count_nonzero(df.sum(axis = _axis_), axis = _axis_)

#Count number of columns with all zeros
print('RNA: {0}  CNV: {1}'.format(count_all_zeros(x_rna, 0), count_all_zeros(x_cnv_imputed, 0)))

#Delete columns with all zeros
x_cnv_nozero = x_cnv_imputed.loc[:, (x_cnv_imputed != 0).any(axis = 0)]
x_rna_nozero = x_rna.loc[:, (x_rna != 0).any(axis = 0)]

#Count number of column deletions
print('RNA:', x_rna.shape[1] - x_rna_nozero.shape[1])
print('CNV:', x_cnv_imputed.shape[1] - x_cnv_nozero.shape[1])

#Count number of columns with all zeros
print('RNA: {0}  CNV: {1}'.format(count_all_zeros(x_rna_nozero, 0), count_all_zeros(x_cnv_nozero, 0)))

#Scale data
zscore = lambda x: (x-x.mean())/ x.std()

x_rna_processed, x_cnv_processed = x_rna_nozero.transform(zscore), x_cnv_nozero.transform(zscore)

#Check for NaN values
print('RNA: {0}  CNV: {1}'.format(x_rna_processed.isnull().any().any(), x_cnv_processed.isnull().any().any()))

#Combine CNV and RNA data
x_combined = pd.concat([x_cnv_processed.reset_index(drop = True), x_rna_processed.reset_index(drop = True)] , axis = 1)
y_combined = y
x_combined.shape

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

#Split data into training and test sets
x_combined_train, x_combined_test, y_combined_train, y_combined_test = train_test_split(
    x_combined, y_combined.values.flatten(), test_size = 0.25, random_state = 0)

model = Sequential()
model.add(Dense(512, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_combined_train.values, y_combined_train, epochs = 20, batch_size = 128)

print(model.evaluate(x_combined_test, y_combined_test))

print(model.metrics_names)
