import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten,  Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.activations import tanh, softmax
from sklearn.metrics import precision_score, recall_score, f1_score


def lenet(initializer):
    # Linear stack of layers in LeNet, so Sequential
    model = Sequential()
    # 5x5 kernel size, with 1 stride and 6 featured maps
    model.add(Conv2D(filters=6, kernel_size=5, kernel_initializer=initializer, activation=tanh, input_shape=(28, 28, 1)))
    # 2x2 pooling
    model.add(AveragePooling2D(pool_size=2, strides=2))
    # 5x5 kernel size, with 1 stride and 16 featured maps
    model.add(Conv2D(filters=16, kernel_size=5, kernel_initializer=initializer, activation=tanh))
    # 2x2 pooling
    model.add(AveragePooling2D(pool_size=2, strides=2))
    # Reshape to one dimensional tensor
    model.add(Flatten())
    # Output
    model.add(Dense(120, activation=tanh))
    model.add(Dense(84, activation=tanh))
    model.add(Dense(10, activation=softmax))
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    # Multi class classification, so categorical cross entropy loss function is selected
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# Timer
start_time = time.time()

initializers_dict = {
    'orthogonal': keras.initializers.Orthogonal(gain=1.0, seed=1),
    'random_normal': keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
    'random_uniform': keras.initializers.RandomUniform(seed=1),
    'glorot_normal': keras.initializers.GlorotNormal(seed=1),
    'glorot_uniform': keras.initializers.GlorotUniform(seed=1),
    'he_normal': keras.initializers.HeNormal(seed=1),
    'he_uniform': keras.initializers.HeUniform(seed=1),
    'lecun_normal': keras.initializers.LecunNormal(seed=1),
    'lecun_uniform': keras.initializers.LecunUniform(seed=1),
}

# Initialize dictionaries to store metrics for each algorithm
train_accuracies = {}
train_losses = {}
train_precisions = {}
train_recalls = {}
train_f_scores = {}
test_accuracies = {}
test_losses = {}
test_precisions = {}
test_recalls = {}
test_f_scores = {}

# Execute the code 5 times
for _ in range(5):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for initializer in initializers_dict:
        if initializer not in train_precisions:
            train_precisions[initializer] = []
            train_recalls[initializer] = []
            train_f_scores[initializer] = []
            train_accuracies[initializer] = []
            train_losses[initializer] = []
            test_precisions[initializer] = []
            test_recalls[initializer] = []
            test_f_scores[initializer] = []
            test_losses[initializer] = []
            test_accuracies[initializer] = []

        model = lenet(initializer)

        # Load the dataset
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        # Preprocess
        # 50000 training samples
        x_train = x_train.reshape((60000, 28, 28, 1))
        x_train = x_train.astype('float32') / 255
        # 10000 test samples
        x_test = x_test.reshape((10000, 28, 28, 1))
        x_test = x_test.astype('float32') / 255
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Train the models
        print()
        print(f'Training the LeNet-FMNIST model with a {initializer} weight initializer:')
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8, restore_best_weights=True)
        history = model.fit(x_train, y_train, batch_size=32, epochs=200, validation_data=(x_test, y_test),
                            verbose=1, callbacks=[early_stop])

        score1 = model.evaluate(x_train, y_train)
        print('Train Loss:', score1[0])
        print('Train Accuracy:', score1[1])

        score = model.evaluate(x_test, y_test)
        print('Test Loss:', score[0])
        print('Test Accuracy:', score[1])

        # Make predictions
        y_pred1 = np.argmax(model.predict(x_train), axis=-1)
        y_pred = np.argmax(model.predict(x_test), axis=-1)

        # Calculate precision, recall, and f-score for train
        precision1 = precision_score(np.argmax(y_train, axis=-1), y_pred1, average='macro', zero_division=0)
        recall1 = recall_score(np.argmax(y_train, axis=-1), y_pred1, average='macro')
        f_score1 = f1_score(np.argmax(y_train, axis=-1), y_pred1, average='macro')
        print('Train Precision:', precision1)
        print('Train Recall:', recall1)
        print('Train F-Score:', f_score1)
        print()
        # Calculate precision, recall, and f-score for test
        precision = precision_score(np.argmax(y_test, axis=-1), y_pred, average='macro', zero_division=0)
        recall = recall_score(np.argmax(y_test, axis=-1), y_pred, average='macro')
        f_score = f1_score(np.argmax(y_test, axis=-1), y_pred, average='macro')
        print('Test Precision:', precision)
        print('Test Recall:', recall)
        print('Test F-Score:', f_score)

        # Collect metrics
        train_accuracies[initializer].append(score1[1])
        train_losses[initializer].append(score1[0])
        train_precisions[initializer].append(precision1)
        train_recalls[initializer].append(recall1)
        train_f_scores[initializer].append(f_score1)
        test_precisions[initializer].append(precision)
        test_recalls[initializer].append(recall)
        test_f_scores[initializer].append(f_score)
        test_losses[initializer].append(score[0])
        test_accuracies[initializer].append(score[1])

        # Plot the accuracy and loss
        ax1.plot(history.history['accuracy'], label=initializer)
        ax2.plot(history.history['loss'], label=initializer)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy on LeNet-FMNIST with Different Weight Initializers')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss on LeNet-FMNIST with Different Weight Initializers')
    plt.legend()

    # Save metrics to a file
    with open(f'metrics_loop_{_+1}.txt', 'w') as file:
        file.write(f'Metrics for Loop {_+1}:\n\n')
        for initializer in initializers_dict:
            file.write(f'{initializer} weight initializer:\n')
            file.write(f'Train Loss: {train_losses[initializer][_]}\n')
            file.write(f'Train Accuracy: {train_accuracies[initializer][_]}\n')
            file.write(f'Train Precision: {train_precisions[initializer][_]}\n')
            file.write(f'Train Recall: {train_recalls[initializer][_]}\n')
            file.write(f'Train F-Score: {train_f_scores[initializer][_]}\n')
            file.write(f'Test Loss: {test_losses[initializer][_]}\n')
            file.write(f'Test Accuracy: {test_accuracies[initializer][_]}\n')
            file.write(f'Test Precision: {test_precisions[initializer][_]}\n')
            file.write(f'Test Recall: {test_recalls[initializer][_]}\n')
            file.write(f'Test F-Score: {test_f_scores[initializer][_]}\n')
            file.write('\n')

    # Set the desired window size (width and height in pixels)
    desired_width = 1920
    desired_height = 1080

    # Get the current figure and adjust the window size
    fig = plt.gcf()
    fig.set_size_inches(desired_width / 100, desired_height / 100)  # Convert pixels to inches

    # Save the plot
    plt.savefig(f'lenet_fmnist_{_+1}.png', dpi='figure')
    plt.close()


# Print average metrics for each algorithm
print('Average Metrics for Each Initializer:')
for initializer in initializers_dict:
    avg_train_accuracy = np.mean(train_accuracies[initializer])
    avg_train_loss = np.mean(train_losses[initializer])
    avg_train_precision = np.mean(train_precisions[initializer])
    avg_train_recall = np.mean(train_recalls[initializer])
    avg_train_f_score = np.mean(train_f_scores[initializer])
    avg_test_precision = np.mean(test_precisions[initializer])
    avg_test_recall = np.mean(test_recalls[initializer])
    avg_test_f_score = np.mean(test_f_scores[initializer])
    avg_test_accuracy = np.mean(test_accuracies[initializer])
    avg_test_loss = np.mean(test_losses[initializer])

    print(f'\n{initializer} weight initializer:')
    print('Average Train Loss:', avg_train_loss)
    print('Average Train Accuracy:', avg_train_accuracy)
    print('Average Train Precision:', avg_train_precision)
    print('Average Train Recall:', avg_train_recall)
    print('Average Train F-Score:', avg_train_f_score)
    print()
    print('Average Test Loss:', avg_test_loss)
    print('Average Test Accuracy:', avg_test_accuracy)
    print('Average Test Precision:', avg_test_precision)
    print('Average Test Recall:', avg_test_recall)
    print('Average Test F-Score:', avg_test_f_score)

# Print total execution time
end_time = time.time()
total_time = end_time - start_time
minutes = int(total_time // 60)
seconds = int(total_time % 60)
print('\nTotal Execution Time: {} minutes and {} seconds'.format(minutes, seconds))
