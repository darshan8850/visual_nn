from neural_net.nn import Model,Layer_Dense,Layer_Dropout,Activation_ReLU,Layer_Input,Activation_Softmax,Optimizer_Adam,Loss,Loss_CategoricalCrossentropy,Activation_Softmax_Loss_CategoricalCrossentropy,Accuracy,Accuracy_Categorical,train_dict
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle
import pickle

def load_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    y_train = y_train.astype('uint8')
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    y_test = y_test.astype('uint8')
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    return X_train, y_train, X_test, y_test

X, y, X_test, y_test = load_mnist_dataset()

model = Model()

model.add(Layer_Dense(X.shape[1], 32))
model.add(Activation_ReLU())
model.add(Layer_Dense(32, 32))
model.add(Activation_ReLU())
model.add(Layer_Dense(32, 10))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test),
            epochs=5, batch_size=128, print_every=100)

model.save("digits_mnist.model")

with open("train_dict.pkl", "wb") as f:
    pickle.dump(train_dict, f)
