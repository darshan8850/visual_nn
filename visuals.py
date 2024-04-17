from neural_net.nn import Model,Layer_Dense,Layer_Dropout,Activation_ReLU,Layer_Input,Activation_Softmax,Optimizer_Adam,Loss,Loss_CategoricalCrossentropy,Activation_Softmax_Loss_CategoricalCrossentropy,Accuracy,Accuracy_Categorical,train_dict
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle
import pickle
import os
from matplotlib import animation
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("dark_background")

def load_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    y_train = y_train.astype('uint8')
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    y_test = y_test.astype('uint8')
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    return X_train, y_train, X_test, y_test

X, y, X_test, y_test = load_mnist_dataset()

model=Model.load("digits_mnist.model")

def visualize_layers(sample_num, fig):
    input_data = X_test[sample_num]
    output_truth = y_test[sample_num]
    prediction = model.predict(input_data)

    if np.argmax(prediction) == output_truth:
        title_text = f"Correct, Prediction: {np.argmax(prediction)}, Truth: {output_truth}"
        fig.suptitle(title_text, color="g", fontsize=20)

    else:
        title_text = f"Incorrect, Prediction: {np.argmax(prediction)}, Truth: {output_truth}"
        fig.suptitle(title_text, color="r", fontsize=20)

    ax0 = plt.subplot2grid((1,7), (0,0), rowspan=1, colspan=1)
    ax1 = plt.subplot2grid((1,7), (0,1), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((1,7), (0,2), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((1,7), (0,3), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((1,7), (0,4), rowspan=1, colspan=1)
    ax5 = plt.subplot2grid((1,7), (0,5), rowspan=1, colspan=1)
    ax6 = plt.subplot2grid((1,7), (0,6), rowspan=1, colspan=1)

    ax0.imshow(input_data.reshape(28, 28), cmap='gray')

    layer_1 = np.rot90(model.layers[0].output, k=3, axes=(0,1))
    layer_1_activated = np.rot90(model.layers[1].output, k=3, axes=(0,1))

    ax1.imshow(layer_1, cmap='RdYlGn')#, aspect='auto')
    ax2.imshow(layer_1_activated, cmap='YlGn')#, aspect='auto')

    ax1.set_title("L1 Out")
    ax2.set_title("L1 Activated")

    layer_2 = np.rot90(model.layers[2].output, k=3, axes=(0,1))
    layer_2_activated = np.rot90(model.layers[3].output, k=3, axes=(0,1))

    ax3.imshow(layer_2, cmap='RdYlGn')#, aspect='auto')
    ax4.imshow(layer_2_activated, cmap='YlGn')#, aspect='auto')

    ax3.set_title("L2 Out")
    ax4.set_title("L2 Activated")

    layer_3 = np.rot90(model.layers[4].output, k=3, axes=(0,1))
    layer_3_activated = np.rot90(model.layers[5].output, k=3, axes=(0,1))

    ax5.imshow(layer_3, cmap='RdYlGn')#, aspect='auto')
    ax6.imshow(layer_3_activated, cmap='YlGn')#, aspect='auto')

    ax5.set_title("L3 Out")
    ax6.set_title("L3 Activated")


LIMIT = 1000

d = "layer_outs"
if not os.path.exists(d):
    os.makedirs(d)

fig = plt.figure(figsize=(12, 12))

def animate(i):
    fig.clear()
    visualize_layers(i, fig)
    fig.savefig(f"{d}/{y_test[i]}-{i}.png")

ani = animation.FuncAnimation(fig, animate, frames=LIMIT, repeat=False)
plt.show()

layer_data_by_class = {
    0: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    1: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    2: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    3: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    4: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    5: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    6: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    7: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    8: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    9: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
}

for data_n in tqdm(range(len(X_test))):
    model.predict(X_test[data_n].reshape(1, 28, 28))
    truth = y_test[data_n]

    for layer_n, layer in enumerate(model.layers):
        layer_data_by_class[truth][layer_n].append(layer.output)

layer_data_by_class_averages = {
    0: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    1: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    2: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    3: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    4: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    5: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    6: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    7: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    8: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
    9: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
}

for class_n in range(10):
    for layer_n in range(6):
        layer_data_by_class_averages[class_n][layer_n] = np.mean(layer_data_by_class[class_n][layer_n], axis=0)


dname = "class_avgs"
if not os.path.exists(dname):
    os.makedirs(dname)

style.use("dark_background")

class_sample_dict = {}

# Select one sample from each class to represent
for class_n in range(10):
    class_sample_dict[class_n] = X_test[np.argmax(y_test == class_n)]

class_description_dict = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
}

for class_n in range(10):
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f"Average Values for Class {class_n} ({class_description_dict[class_n]})", color="w", fontsize=20)

    ax0 = plt.subplot2grid((1, 7), (0, 0), rowspan=1, colspan=1)
    ax1 = plt.subplot2grid((1, 7), (0, 1), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((1, 7), (0, 2), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((1, 7), (0, 3), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((1, 7), (0, 4), rowspan=1, colspan=1)
    ax5 = plt.subplot2grid((1, 7), (0, 5), rowspan=1, colspan=1)
    ax6 = plt.subplot2grid((1, 7), (0, 6), rowspan=1, colspan=1)

    ax0.imshow(class_sample_dict[class_n], cmap='gray')

    for layer_idx in range(3):
        layer_out = np.rot90(layer_data_by_class_averages[class_n][2 * layer_idx], k=3, axes=(0, 1))
        layer_activated = np.rot90(layer_data_by_class_averages[class_n][2 * layer_idx + 1], k=3, axes=(0, 1))

        ax = [ax1, ax2, ax3, ax4, ax5, ax6][layer_idx * 2]
        ax_activated = [ax1, ax2, ax3, ax4, ax5, ax6][layer_idx * 2 + 1]

        ax.imshow(layer_out, cmap='RdYlGn')
        ax_activated.imshow(layer_activated, cmap='YlGn')

        ax.set_title(f"L{layer_idx + 1} Out")
        ax_activated.set_title(f"L{layer_idx + 1} Activated")

    plt.savefig(f"{dname}/{class_n}.png")