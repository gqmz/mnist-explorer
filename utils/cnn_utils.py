#tf/keras imports
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

#math
import numpy as np
import math

#plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=15)
mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15)

def clear_backend():
    """
    utility function to clear keras backend, sets seeds for reproducibility
    """
    K.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

def get_data_splits(val_split=0.1):
    """
    Load MNIST data from keras & split into train, valid, test sets
    Input:
        val_split: fraction of train set to be used as validation
    Return:
        train, valid, test: (X, y) tuples
    """

    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    #split into train, validate, test sets
    split_index = int(x_train_full.shape[0]*val_split)
    x_train, y_train = x_train_full[split_index:]/255, y_train_full[split_index:]
    x_valid, y_valid = x_train_full[:split_index]/255, y_train_full[:split_index]
    x_test = x_test/255

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def build_model_cnn(lr=1e-3):
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(64, 7, activation='relu', padding='same', 
                                input_shape=[28,28,1]),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])

    optimizer = keras.optimizers.Nadam(learning_rate=lr)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics=['accuracy']
    )
    return model

class sweep_learning_rate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        """
        1. save current batch's losses & learning rate
        2. update learning rate for next batch
        """
        self.losses.append(logs['loss'])
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        K.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate*self.factor)

    def plot_lr_sweep(self):
        """
        Plot train loss vs learning rate
        """
        plt.figure(figsize=(8,8))
        plt.scatter(x=self.rates, y=self.losses, label='Training Loss')
        plt.hlines(y=min(self.losses), xmin=min(self.rates), xmax=max(self.rates), color='r', linestyle='dashed')
        plt.xscale('log')
        plt.xlabel('Model learning rate', fontsize=15)
        plt.ylabel('Training loss', fontsize=15)
        plt.grid('on')
        plt.show()

def plot_result(X, y_true, y_pred=None, nrows=3, ncols=3):
    """
    plot nrows x ncols grid of data samples from X with ground truth & predicted labels
    """
    assert len(X)==len(y_true), "Features & labels must have same length"
    if y_pred is not None:
        assert len(y_true)==len(y_pred), "Labels & predictions must have the same length"

    fig, ax = plt.subplots(nrows, ncols)

    for r in range(nrows):
        for c in range(ncols):
            idx = r*nrows + c #index of grid point

            ax[r][c].imshow(X[idx], cmap='binary')
            ax[r][c].grid('off') #turn off grid
            #turn off ticks
            ax[r][c].set_xticks([])
            ax[r][c].set_yticks([])

            #title based on input data
            if y_pred is not None:
                ax[r][c].set_title("Actual: " + str(y_true[idx]) + "\nPredicted: " + str(y_pred[idx]))
            else:
                ax[r][c].set_title("Actual: " + str(y_true[idx]))

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
        