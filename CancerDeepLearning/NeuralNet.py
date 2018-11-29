import pandas as pd
import numpy as np

y_combined = pd.read_table('CancerTypes_y.txt', sep = '\t', header = None)
x_combined = pd.read_csv('Combined_processed.csv', header = 0)

from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt

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
