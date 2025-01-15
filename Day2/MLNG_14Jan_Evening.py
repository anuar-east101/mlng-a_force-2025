# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:23:05 2025

@author: sah
"""

url = 'https://raw.githubusercontent.com/anuar-east101/mlng-a_force-2025/refs/heads/main/Day2/valve_data.csv'

import pandas as pd

from plotly.offline import plot
import plotly.graph_objs as go
import plotly.express as px

my_data = pd.read_csv(url)


fig = px.scatter_3d(my_data, x="OP", y="dP", z="F", color="Cv")
plot(fig, auto_open=True)

#%% preprocess data

X = pd.DataFrame(columns=['OP', 'dP', 'Cv'])
X['OP'] = my_data['OP'] / max(my_data['OP'])
X['Cv'] = my_data['Cv'] / max(my_data['Cv'])
X['dP'] = my_data['dP'] / max(my_data['dP'])

y = my_data['F'] / max(my_data['F'])

# split and randomise data using train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%% build neural network 
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# configure some settings that we will use for model training later
my_optimizer = keras.optimizers.Adam(learning_rate=0.01)
my_metric = tf.keras.metrics.MeanAbsoluteError()

# configure and build the neural network
model = Sequential()
model.add(Dense(1, input_dim=3, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mae', optimizer=my_optimizer, metrics=my_metric)

# train the model, 
# we assign the output to "history" so that we can monitor it's performance history
history = model.fit(X_train, y_train, verbose=1, epochs=10, validation_split=0.2)

#%% plot loss
import matplotlib.pyplot as plt
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()

#%% use the trained model to predict y given X for train and test dataset
predicted_y_train = model.predict(X_train)
predicted_y_test = model.predict(X_test)

plt.figure()
plt.plot(y_train, predicted_y_train,'o', alpha=0.5, label='Training dataset')
plt.plot(y_test, predicted_y_test,'o', alpha=0.5, label='Testing dataset')
plt.plot([0,2],[0,2],'--k', label='Reference line') # 45 degree reference line
plt.xlabel('Flow from valve equation')
plt.ylabel('Flow predicted by neural network')
plt.legend()

#%%
weights = model.get_weights()
