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

#Model Creation function
def create_model(n1 = 128, n2 = 128, dropout = 0.2, activation = 'relu'):
    model = Sequential()
    model.add(Dense(n1, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n2, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

#Create Model
model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 32)

neurons = [16,32,64,128]
dropouts = [0, 0.1, 0.2, 0.4]
activations = ['relu','linear','tanh']
param_grid = dict(n1 = neurons, n2 = neurons, dropout = dropouts, activation = activations)
grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = 1, verbose = 1)
grid_result = grid.fit(x_combined_train.values, y_combined_train)



print(grid_result.score(x_combined_test, y_combined_test))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

"""
print(model.metrics_names)
plt.plot(model.history.epoch, model.history.history['loss'])
plt.show()
"""
