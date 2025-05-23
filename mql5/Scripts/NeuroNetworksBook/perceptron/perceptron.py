# -------------------------------------------------------#
# Script for creating and testing different models of    #
# fully connected perceptron with the same dataset.      #
# The script creates three models:                       #
# - fully connected perceptron with one hidden layer,    #
# - fully connected perceptron with three hidden layers, #
# - fully connected perceptron with three hidden layers  #
#   & regularization.                                    #
# When training models, from training dataset, script    #
# allocates 20% to validate the outputs.                 #
# After training, the script tests the performance       #
# of the model on a test dataset (separate data file)    #
# -------------------------------------------------------#
# Import Libraries
import os
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import MetaTrader5 as mt5

# Add fonts
font_list=fm.findSystemFonts()
for f in font_list:
    if(f.__contains__('ClearSans')):
        fm.fontManager.addfont(f)

# Set parameters for output graphs
mp.rcParams.update({'font.family':'serif',
                    'font.serif':'Clear Sans',
                    'axes.titlesize': 'x-large',
                    'axes.labelsize':'medium',
                    'xtick.labelsize':'small',
                    'ytick.labelsize':'small',
                    'legend.fontsize':'small',
                    'figure.figsize':[6.0,4.0],
                    'axes.titlecolor': '#707070',
                    'axes.labelcolor': '#707070',
                    'axes.edgecolor': '#707070',
                    'xtick.labelcolor': '#707070',
                    'ytick.labelcolor': '#707070',
                    'xtick.color': '#707070',
                    'ytick.color': '#707070',
                    'text.color': '#707070',
                    'lines.linewidth': 0.8,
                    'axes.linewidth': 0.5
                   })

# Load training dataset
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

path=os.path.join(mt5.terminal_info().data_path,r'MQL5\Files')
mt5.shutdown()
filename = os.path.join(path,'study_data.csv')
data = np.asarray( pd.read_table(filename,
                   sep=',',
                   header=None,
                   skipinitialspace=True,
                   encoding='utf-8',
                   float_precision='high',
                   dtype=np.float64,
                   low_memory=False))

# Split training dataset to input data and target
inputs=data.shape[1]-2
targerts=2
train_data=data[:,0:inputs]
train_target=data[:,inputs:]

# Create the first model with one hidden layer
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
model1 = keras.Sequential([keras.layers.InputLayer(input_shape=inputs),
                           keras.layers.Dense(40, activation=tf.nn.swish), 
                           keras.layers.Dense(targerts, activation=tf.nn.tanh) 
                         ])
model1.compile(optimizer='Adam', 
               loss='mean_squared_error', 
               metrics=['accuracy'])
history1 = model1.fit(train_data, train_target,
                      epochs=500, batch_size=1000,
                      callbacks=[callback],
                      verbose=2,
                      validation_split=0.2,
                      shuffle=True)
model1.save(os.path.join(path,'perceptron1.h5'))

# Create a model with three hidden layers
model2 = keras.Sequential([keras.layers.InputLayer(input_shape=inputs),
                           keras.layers.Dense(40, activation=tf.nn.swish), 
                           keras.layers.Dense(40, activation=tf.nn.swish), 
                           keras.layers.Dense(40, activation=tf.nn.swish), 
                           keras.layers.Dense(targerts, activation=tf.nn.tanh) 
                         ])
model2.compile(optimizer='Adam', 
               loss='mean_squared_error', 
               metrics=['accuracy'])
history2 = model2.fit(train_data, train_target,
                      epochs=500, batch_size=1000,
                      callbacks=[callback],
                      verbose=2,
                      validation_split=0.2,
                      shuffle=True)
model2.save(os.path.join(path,'perceptron2.h5'))


# Add regularization to the model with three hidden layers
model3 = keras.Sequential([keras.layers.InputLayer(input_shape=inputs),
                           keras.layers.Dense(40, activation=tf.nn.swish, kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)), 
                           keras.layers.Dense(40, activation=tf.nn.swish, kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)), 
                           keras.layers.Dense(40, activation=tf.nn.swish, kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)), 
                           keras.layers.Dense(targerts, activation=tf.nn.tanh) 
                         ])
model3.compile(optimizer='Adam', 
               loss='mean_squared_error', 
               metrics=['accuracy'])
history3 = model3.fit(train_data, train_target,
                      epochs=500, batch_size=1000,
                      callbacks=[callback],
                      verbose=2,
                      validation_split=0.2,
                      shuffle=True)
model3.save(os.path.join(path,'perceptron3.h5'))

# Plotting the first model training results

plt.plot(history1.history['loss'], label='Train 1 hidden layer')
plt.plot(history1.history['val_loss'], label='Validation 1 hidden layer')
plt.ylabel('$MSE$ $loss$')
plt.xlabel('$Epochs$')
plt.title('Model training dynamics')
plt.legend(loc='upper right')

plt.figure()
plt.plot(history1.history['accuracy'], label='Train 1 hidden layer')
plt.plot(history1.history['val_accuracy'], label='Validation 1 hidden layer')
plt.ylabel('$Accuracy$')
plt.xlabel('$Epochs$')
plt.title('Model training dynamics')
plt.legend(loc='lower right')

# Plotting the training results of the second model
plt.figure()
plt.plot(history1.history['loss'], label='Train 1 hidden layer')
plt.plot(history1.history['val_loss'], label='Validation 1 hidden layer')
plt.plot(history2.history['loss'], label='Train 3 hidden layers')
plt.plot(history2.history['val_loss'], label='Validation 3 hidden layers')
plt.ylabel('$MSE$ $loss$')
plt.xlabel('$Epochs$')
plt.title('Model training dynamics')
plt.legend(loc='lower left',ncol=2)

plt.figure()
plt.plot(history1.history['accuracy'], label='Train 1 hidden layer')
plt.plot(history1.history['val_accuracy'], label='Validation 1 hidden layer')
plt.plot(history2.history['accuracy'], label='Train 3 hidden layers')
plt.plot(history2.history['val_accuracy'], label='Validation 3 hidden layers')
plt.ylabel('$Accuracy$')
plt.xlabel('$Epochs$')
plt.title('Model training dynamics')
plt.legend(loc='lower left',ncol=2)

# Plotting the training results of the third model
plt.figure()
plt.plot(history1.history['loss'], label='Train 1 hidden layer')
plt.plot(history1.history['val_loss'], label='Validation 1 hidden layer')
plt.plot(history2.history['loss'], label='Train 3 hidden layers')
plt.plot(history2.history['val_loss'], label='Validation 3 hidden layers')
plt.plot(history3.history['loss'], label='Train 3 hidden layers\nvs regularization')
plt.plot(history3.history['val_loss'], label='Validation 3 hidden layers\nvs regularization')
plt.ylabel('$MSE$ $Loss$')
plt.xlabel('$Epochs$')
plt.title('Model training dynamics')
plt.legend(loc='lower left',ncol=2)

plt.figure()
plt.plot(history1.history['accuracy'], label='Train 1 hidden layer')
plt.plot(history1.history['val_accuracy'], label='Validation 1 hidden layer')
plt.plot(history2.history['accuracy'], label='Train 3 hidden layers')
plt.plot(history2.history['val_accuracy'], label='Validation 3 hidden layers')
plt.plot(history3.history['accuracy'], label='Train 3 hidden layers\nvs regularization')
plt.plot(history3.history['val_accuracy'], label='Validation 3 hidden layers\nvs regularization')
plt.ylabel('$Accuracy$')
plt.xlabel('$Epochs$')
plt.title('Model training dynamics')
plt.legend(loc='upper left',ncol=2)

# Load testing dataset
test_filename = os.path.join(path,'test_data.csv')
test = np.asarray( pd.read_table(test_filename,
                   sep=',',
                   header=None,
                   skipinitialspace=True,
                   encoding='utf-8',
                   float_precision='high',
                   dtype=np.float64,
                   low_memory=False))
# Split test dataset to input data and target
test_data=test[:,0:inputs]
test_target=test[:,inputs:]

# Check model results on a test dataset
test_loss1, test_acc1 = model1.evaluate(test_data, test_target, verbose=2) 
test_loss2, test_acc2 = model2.evaluate(test_data, test_target, verbose=2) 
test_loss3, test_acc3 = model3.evaluate(test_data, test_target, verbose=2) 

# Log testing results
print('Model 1 hidden layer')
print('Test accuracy:', test_acc1)
print('Test loss:', test_loss1)

print('Model 3 hidden layers')
print('Test accuracy:', test_acc2)
print('Test loss:', test_loss2)

print('Model 3 hidden layers vs regularization')
print('Test accuracy:', test_acc3)
print('Test loss:', test_loss3)

plt.figure()
plt.bar(['1 hidden layer','3 hidden layers', '3 hidden layers\nvs regularization'],[test_loss1,test_loss2,test_loss3])
plt.ylabel('$MSE$ $loss$')
plt.title('Test results')
plt.figure()
plt.bar(['1 hidden layer','3 hidden layers', '3 hidden layers\nvs regularization'],[test_acc1,test_acc2,test_acc3])
plt.ylabel('$Accuracy$')
plt.title('Test results')

plt.show()
