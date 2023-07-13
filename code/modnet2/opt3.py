import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import optuna
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from keras.optimizers import SGD
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.activations import relu, softmax
from sklearn.metrics import precision_score, recall_score, f1_score


def modnet2(filters, learning_rate):
    # Sequential model
    model = Sequential()
    model.add(Conv2D(filters=filters[0], kernel_size=3, kernel_initializer='lecun_normal', activation=relu, padding="same", input_shape=(32, 32, 3)))
    model.add(Conv2D(filters=filters[1], kernel_size=3, kernel_initializer='lecun_normal', activation=relu))
    # 2x2 Max Pooling
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # Dropout regularization of 0.5
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=filters[2], kernel_size=3, kernel_initializer='lecun_normal', activation=relu, padding="same"))
    model.add(Conv2D(filters=filters[3], kernel_size=3, kernel_initializer='lecun_normal', activation=relu))
    # 2x2 Max Pooling
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # Dropout regularization of 0.5
    model.add(Dropout(0.5))
    # Reshape to one-dimensional tensor
    model.add(Flatten())
    # 1st Dense Layer: 512 neurons
    model.add(Dense(512, activation=relu))
    # Dropout regularization of 0.5
    model.add(Dropout(0.5))
    # Output Layer: 100 neurons for the CIFAR-100 dataset, with softmax activation
    model.add(Dense(100, activation=softmax))
    # Compile the model
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    # Multi-class classification, so categorical cross-entropy loss function is selected
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def objective(trial):
    filters = [
        trial.suggest_int('filter1', 64, 128, step=32),
        trial.suggest_int('filter2', 64, 128, step=32),
        trial.suggest_int('filter3', 128, 256, step=32),
        trial.suggest_int('filter4', 128, 256, step=32)
    ]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    model = modnet2(filters, learning_rate)

    # Load the dataset
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    # Preprocess
    x_train = x_train.reshape((50000, 32, 32, 3)).astype('float32') / 255
    x_test = x_test.reshape((10000, 32, 32, 3)).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Train the model
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8, restore_best_weights=True)
    history = model.fit(x_train, y_train, batch_size=32, epochs=200, validation_data=(x_test, y_test),
                        verbose=1, callbacks=[early_stop])

    score1 = model.evaluate(x_train, y_train)
    score = model.evaluate(x_test, y_test)
    accuracy = score[1]
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Print the best hyperparameters and their values
print('Best Hyperparameters:')
print(study.best_params)
print()


